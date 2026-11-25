import dataclasses
import enum
import math
from collections.abc import Iterator
from typing import Any, Literal, overload, override

import jaxtyping as jt
import tinycudann as tcnn
import torch
from torch import nn
from typing_extensions import deprecated

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.sdf_field import SDFField, SDFFieldConfig
from ssdp.utils.function import signature_of
from ssdp.utils.jaxtyping import jaxtyped


@jaxtyped()
def _log_lerp(
    a: jt.Float[torch.Tensor, " *B "],
    b: jt.Float[torch.Tensor, " *B "],
    w: jt.Float[torch.Tensor, " *B "] | float,
) -> jt.Float[torch.Tensor, " *B "]:
    return torch.exp(torch.lerp(torch.log(a), torch.log(b), w))


class RaySampleLocation(enum.StrEnum):
    START = enum.auto()
    MID = enum.auto()
    END = enum.auto()


@dataclasses.dataclass
class NeuSFieldConfig(SDFFieldConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: NeuSField,
    )
    numerical_gradient: bool = False
    cos_anneal_end_ratio: float = 0.5
    learnable_sdf_scale: bool = True
    sdf_scale_max_value: float = 0.05
    sdf_scale_min_value: float = 0.0005
    sdf_scale_anneal_end_ratio: float = 0.5


class NeuSField(SDFField):
    config: NeuSFieldConfig
    encoding: tcnn.Encoding
    _sdf_scale_max_value: torch.Tensor
    _sdf_scale_min_value: torch.Tensor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer(
            name="_sdf_scale_max_value",
            tensor=torch.as_tensor(self.config.sdf_scale_max_value),
            persistent=False,
        )
        self.register_buffer(
            name="_sdf_scale_min_value",
            tensor=torch.as_tensor(self.config.sdf_scale_min_value),
            persistent=False,
        )
        self.update_gradient_epsilon(self.config.max_res)
        self.progress_ratio = 1.0

    @property
    def progress_ratio(self) -> float:
        return self._progress_ratio

    @progress_ratio.setter
    def progress_ratio(self, progress_ratio: float) -> None:
        self._progress_ratio = progress_ratio

    def _get_linear_anneal_ratio(self, start_ratio: float = 0.0, end_ratio: float = 1.0) -> float:
        numerator = self.progress_ratio - start_ratio
        denominator = max(end_ratio - start_ratio, 1.0e-6)
        linear_anneal_ratio = numerator / denominator
        linear_anneal_ratio = min(max(linear_anneal_ratio, 0.0), 1.0)
        return linear_anneal_ratio

    def _get_cosine_anneal_ratio(self, start_ratio: float = 0.0, end_ratio: float = 1.0) -> float:
        linear_anneal_ratio = self._get_linear_anneal_ratio(start_ratio, end_ratio)
        cosine_anneal_ratio = math.cos(linear_anneal_ratio * math.pi + math.pi) * 0.5 + 0.5
        return cosine_anneal_ratio

    @property
    def _cos_anneal_ratio(self) -> float:
        return self._get_linear_anneal_ratio(end_ratio=self.config.cos_anneal_end_ratio)

    @_cos_anneal_ratio.setter
    def _cos_anneal_ratio(self, cos_anneal_ratio: float) -> None:
        self.set_cos_anneal_ratio(cos_anneal_ratio)

    @deprecated(
        "`SDF.set_cos_anneal_ratio` is deprecated. "
        "Please specify `NeuSConfig.cos_anneal_end_ratio` instead."
    )
    @override
    def set_cos_anneal_ratio(self, _: float) -> None:
        pass

    @property
    def _sdf_scale_anneal_ratio(self) -> float:
        return self._get_linear_anneal_ratio(end_ratio=self.config.sdf_scale_anneal_end_ratio)

    @property
    @jaxtyped()
    def sdf_scale(self) -> jt.Float[torch.Tensor, " "]:
        if self.config.learnable_sdf_scale:
            sdf_inv_scale = self.deviation_network.get_variance().squeeze()
            sdf_scale = 1.0 / sdf_inv_scale
        else:
            sdf_scale = _log_lerp(
                self._sdf_scale_max_value,
                self._sdf_scale_min_value,
                self._sdf_scale_anneal_ratio,
            )
        return sdf_scale

    def _get_field_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.parameters()

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        param_groups = dict(fields=list(self._get_field_parameters()))
        return param_groups

    @jaxtyped()
    def _forward_position_encoder(
        self,
        positions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B {self.position_encoding.get_out_dim()} "]:
        position_encodings = self.position_encoding(positions)
        return position_encodings

    @jaxtyped()
    def _forward_direction_encoder(
        self,
        directions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B {self.direction_encoding.get_out_dim()} "]:
        direction_encodings = self.direction_encoding(directions)
        return direction_encodings

    @jaxtyped()
    def _forward_hash_grid_encoder(
        self,
        positions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B {self.encoding.n_output_dims} "]:
        if self.use_grid_feature:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)
            positions = torch.clamp(positions, 0.0, 1.0)
            hash_grid_encodings: torch.Tensor = self.encoding(positions.flatten(0, -2))
            hash_grid_encodings = hash_grid_encodings.unflatten(0, positions.shape[:-1])
        else:
            hash_grid_encodings = positions.new_zeros(
                *positions.shape[:-1],
                self.encoding.n_output_dims,
            )
        return hash_grid_encodings

    @jaxtyped()
    def _forward_camera_encoder(
        self,
        camera_indices: jt.Int[torch.Tensor, " *B 1 "],
    ) -> jt.Float[torch.Tensor, " *B {self.embedding_appearance.get_out_dim()} "]:
        if self.training:
            if self.config.use_appearance_embedding:
                camera_encodings = self.embedding_appearance(camera_indices.squeeze(-1))
                return camera_encodings
        else:
            if self.use_average_appearance_embedding:
                camera_encodings = self.embedding_appearance.mean(dim=0)
                camera_encodings = camera_encodings.expand(*camera_indices.shape[:-1], -1)
                return camera_encodings

        camera_encodings = camera_indices.new_zeros(
            *camera_indices.shape[:-1],
            self.embedding_appearance.get_out_dim(),
            dtype=torch.float32,
        )

        return camera_encodings

    @jaxtyped()
    def _forward_geo_network(
        self,
        inputs: jt.Float[torch.Tensor, " *B C "],
    ) -> jt.Float[torch.Tensor, " *B 1+{self.config.geo_feat_dim} "]:
        outputs = inputs
        for index in range(self.num_layers - 1):
            layer = getattr(self, f"glin{index}")
            if index in self.skip_in:
                outputs = torch.cat([outputs, inputs], dim=-1)
                outputs = outputs / math.sqrt(2.0)
            outputs = layer(outputs)
            if index < self.num_layers - 2:
                outputs = self.softplus(outputs)
        return outputs

    @deprecated(
        "`SDF.forward_geonetwork` is deprecated. Please use `SDF.forward_geo_network` instead."
    )
    @override
    def forward_geonetwork(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward_geo_network(*args, **kwargs)

    @jaxtyped()
    def forward_geo_network(
        self,
        positions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B 1+{self.config.geo_feat_dim} "]:
        position_encodings = self._forward_position_encoder(positions)
        hash_grid_encodings = self._forward_hash_grid_encoder(positions)
        inputs = torch.cat(
            [
                positions,
                position_encodings,
                hash_grid_encodings,
            ],
            dim=-1,
        )
        outputs = self._forward_geo_network(inputs)
        return outputs

    @jaxtyped()
    def _forward_color_network(
        self,
        inputs: jt.Float[torch.Tensor, " *B C "],
    ) -> jt.Float[torch.Tensor, " *B 3 "]:
        outputs = inputs
        for index in range(self.num_layers_color - 1):
            layer = getattr(self, f"clin{index}")
            outputs = layer(outputs)
            if index < self.num_layers_color - 2:
                outputs = self.relu(outputs)
        outputs = self.sigmoid(outputs)
        return outputs

    @deprecated("`SDF.get_colors` is deprecated. Please use `SDF.forward_color_network` instead.")
    @override
    def get_colors(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward_color_network(*args, **kwargs)

    @jaxtyped()
    def forward_color_network(
        self,
        positions: jt.Float[torch.Tensor, " *B 3 "],
        directions: jt.Float[torch.Tensor, " *B 3 "],
        sdf_normals: jt.Float[torch.Tensor, " *B 3 "],
        geo_features: jt.Float[torch.Tensor, " *B {self.config.geo_feat_dim} "],
        camera_indices: jt.Int[torch.Tensor, " *B 1 "],
    ) -> jt.Float[torch.Tensor, " *B 3 "]:
        direction_encodings = self._forward_direction_encoder(directions)
        camera_encodings = self._forward_camera_encoder(camera_indices)
        inputs = torch.cat(
            [
                positions,
                direction_encodings,
                sdf_normals,
                geo_features,
                camera_encodings,
            ],
            dim=-1,
        )
        outputs = self._forward_color_network(inputs)
        return outputs

    @jaxtyped()
    def sdf(
        self,
        positions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        geo_outputs = self.forward_geo_network(positions)
        sdf_values, _ = torch.split(geo_outputs, [1, self.config.geo_feat_dim], dim=-1)
        return sdf_values

    @jaxtyped()
    @override
    def get_sdf(self, ray_samples: RaySamples) -> jt.Float[torch.Tensor, " *B 1 "]:
        positions = ray_samples.frustums.get_start_positions()
        sdf_values = self.sdf(positions)
        return sdf_values

    @jaxtyped()
    def get_opacity(
        self,
        ray_samples: RaySamples,
        sdf_values: jt.Float[torch.Tensor, " *B 1 "],
        sdf_gradients: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        return self.get_alpha(ray_samples, sdf_values, sdf_gradients)

    @jaxtyped()
    def analytical_gradient(
        self,
        positions: jt.Float[torch.Tensor, " *B 3 "],
        sdf_values: jt.Float[torch.Tensor, " *B 1 "] | None = None,
    ) -> jt.Float[torch.Tensor, " *B 3 "]:
        if sdf_values is None:
            positions.requires_grad_(True)
            with torch.enable_grad():
                sdf_values = self.sdf(positions)
        [sdf_gradients] = torch.autograd.grad(
            outputs=sdf_values,
            inputs=positions,
            grad_outputs=torch.ones_like(sdf_values),
            create_graph=self.training,
            retain_graph=self.training,
        )
        return sdf_gradients

    @overload
    def numerical_gradient(
        self,
        positions: jt.Float[torch.Tensor, " *B 3 "],
        sdf_values: jt.Float[torch.Tensor, " *B 1 "] | None = ...,
        *,
        return_laplacian: Literal[False] = ...,
    ) -> jt.Float[torch.Tensor, " *B 3 "]: ...

    @overload
    def numerical_gradient(
        self,
        positions: jt.Float[torch.Tensor, " *B 3 "],
        sdf_values: jt.Float[torch.Tensor, " *B 1 "] | None = ...,
        *,
        return_laplacian: Literal[True],
    ) -> tuple[
        jt.Float[torch.Tensor, " *B 3 "],
        jt.Float[torch.Tensor, " *B 1 "],
    ]: ...

    @jaxtyped()
    def numerical_gradient(
        self,
        positions: jt.Float[torch.Tensor, " *B 3 "],
        sdf_values: jt.Float[torch.Tensor, " *B 1 "] | None = None,
        *,
        return_laplacian: bool = False,
    ) -> (
        jt.Float[torch.Tensor, " *B 3 "]
        | tuple[
            jt.Float[torch.Tensor, " *B 3 "],
            jt.Float[torch.Tensor, " *B 1 "],
        ]
    ):
        gradient_epsilons = torch.full_like(positions, self._gradient_epsilon)
        gradient_epsilons = torch.diag_embed(gradient_epsilons, dim1=0, dim2=-1)
        pos_sdf_values = self.sdf(positions + gradient_epsilons / 2.0)
        neg_sdf_values = self.sdf(positions - gradient_epsilons / 2.0)
        sdf_gradients = (pos_sdf_values - neg_sdf_values) / self._gradient_epsilon
        sdf_gradients = sdf_gradients.squeeze(-1).movedim(0, -1)
        if not return_laplacian:
            return sdf_gradients
        if sdf_values is None:
            sdf_values = self.sdf(positions)
        pos_sdf_gradients = (pos_sdf_values - sdf_values) / (self._gradient_epsilon / 2.0)
        neg_sdf_gradients = (sdf_values - neg_sdf_values) / (self._gradient_epsilon / 2.0)
        sdf_hessians = (pos_sdf_gradients - neg_sdf_gradients) / (self._gradient_epsilon / 2.0)
        sdf_laplacians = torch.sum(sdf_hessians, dim=0)
        return sdf_gradients, sdf_laplacians

    @jaxtyped()
    def gradient(
        self,
        positions: jt.Float[torch.Tensor, " *B 3 "],
        sdf_values: jt.Float[torch.Tensor, " *B 1 "] | None = None,
    ) -> jt.Float[torch.Tensor, " *B 3 "]:
        return (
            self.numerical_gradient(positions)
            if self.config.numerical_gradient
            else self.analytical_gradient(positions, sdf_values)
        )

    @override
    def get_outputs(
        self,
        ray_samples: RaySamples,
        return_alphas: bool = True,
        return_colors: bool = True,
        return_gradients: bool | None = None,
        ray_sample_location: RaySampleLocation = RaySampleLocation.START,
    ) -> dict[FieldHeadNames | str, torch.Tensor]:
        directions = ray_samples.frustums.directions

        match ray_sample_location:
            case RaySampleLocation.START:
                positions = ray_samples.frustums.get_start_positions()
            case RaySampleLocation.MID:
                positions = ray_samples.frustums.get_positions()
            case RaySampleLocation.END:
                positions = ray_samples.frustums.get_start_positions()
                assert ray_samples.deltas is not None
                intervals = torch.clamp(ray_samples.deltas, min=0.0)
                positions = positions + directions * intervals

        # NOTE: `RaySamples.camera_indices.dtype` is set to `torch.float32` by `nerfstudio.exporter.texture_utils.export_textured_mesh`.
        # https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/exporter/texture_utils.py#L388
        assert ray_samples.camera_indices is not None
        camera_indices = ray_samples.camera_indices.long()

        with torch.set_grad_enabled(not self.config.numerical_gradient or self.training):
            positions.requires_grad_(not self.config.numerical_gradient)
            geo_outputs = self.forward_geo_network(positions)
            sdf_values, geo_features = torch.split(
                tensor=geo_outputs,
                split_size_or_sections=[1, self.config.geo_feat_dim],
                dim=-1,
            )

        outputs: dict[FieldHeadNames | str, torch.Tensor] = {FieldHeadNames.SDF: sdf_values}
        outputs.update(geo_features=geo_features)

        if return_alphas or return_colors or return_gradients:
            if self.config.numerical_gradient:
                sdf_gradients, sdf_laplacians = self.numerical_gradient(
                    positions=positions,
                    sdf_values=sdf_values,
                    return_laplacian=True,
                )
                outputs.update(laplacian=sdf_laplacians)

            else:
                sdf_gradients = self.analytical_gradient(
                    positions=positions,
                    sdf_values=sdf_values,
                )

            sdf_normals = torch.nn.functional.normalize(sdf_gradients, p=2, dim=-1)

            outputs |= {
                FieldHeadNames.NORMALS: sdf_normals,
                FieldHeadNames.GRADIENT: sdf_gradients,
            }

            if return_colors:
                colors = self.forward_color_network(
                    positions=positions,
                    directions=directions,
                    # NOTE: `SDFField` passes the gradients instead of normals.
                    sdf_normals=sdf_gradients,
                    geo_features=geo_features,
                    camera_indices=camera_indices,
                )
                outputs |= {FieldHeadNames.RGB: colors}

            if return_alphas:
                opacities = self.get_opacity(
                    ray_samples=ray_samples,
                    sdf_values=sdf_values,
                    sdf_gradients=sdf_gradients,
                )
                outputs |= {FieldHeadNames.ALPHA: opacities}

        return outputs

    @signature_of(get_outputs)
    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.get_outputs(*args, **kwargs)

    @signature_of(forward)
    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)

    @torch.no_grad()
    def on_train_iteration_start(self, step: int) -> None:
        if not self.config.learnable_sdf_scale:
            sdf_scale_logit = -torch.log(self.sdf_scale) / 10.0
            self.deviation_network.variance.copy_(sdf_scale_logit)

    def on_train_iteration_end(self, step: int) -> None:
        pass

    def update_gradient_epsilon(self, grid_resolution: float) -> None:
        aabb_min, aabb_max = self.aabb
        grid_range = max(aabb_max - aabb_min)
        gradient_epsilon = grid_range / grid_resolution
        self._gradient_epsilon = gradient_epsilon.item()
