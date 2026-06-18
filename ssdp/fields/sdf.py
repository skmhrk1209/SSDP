import dataclasses
import functools
import math
from typing import Any, override

import jaxtyping as jt
import tinycudann as tcnn
import torch
from typing_extensions import deprecated

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.sdf_field import SDFField, SDFFieldConfig
from ssdp.utils.jaxtyping import jaxtyped


def _variance_to_beta(variance: float) -> float:
    return math.log(variance * 3.0 / math.pi**2.0) / -20.0


@dataclasses.dataclass
class SDFConfig(SDFFieldConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: SDF,
    )
    beta_init: float = dataclasses.field(
        default_factory=functools.partial(
            _variance_to_beta,
            variance=1.0e-3,
        ),
    )
    enable_finite_diff_grad: bool = False


class SDF(SDFField):
    config: SDFConfig
    encoding: tcnn.Encoding
    grid_encoding_mask: torch.Tensor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        grid_encoding_mask = torch.ones(self.config.num_levels * self.config.features_per_level)
        self.register_buffer("grid_encoding_mask", grid_encoding_mask)
        self.update_finite_diff_epsilon(self.config.max_res)
        self.set_progress_ratio(1.0)

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
    def _forward_grid_encoder(
        self,
        positions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B {self.encoding.n_output_dims} "]:
        if self.use_grid_feature:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)
            positions = torch.clamp(positions, 0.0, 1.0)
            grid_encodings: torch.Tensor = self.encoding(positions.flatten(0, -2))
            grid_encodings = grid_encodings.unflatten(0, positions.shape[:-1])
            grid_encodings = grid_encodings * self.grid_encoding_mask
        else:
            grid_encodings = positions.new_zeros(
                *positions.shape[:-1],
                self.encoding.n_output_dims,
            )
        return grid_encodings

    @jaxtyped()
    def _forward_camera_encoder(
        self,
        camera_indices: jt.Int[torch.Tensor, " *B 1 "],
    ) -> jt.Float[torch.Tensor, " *B {self.embedding_appearance.get_out_dim()} "]:
        if self.config.use_appearance_embedding:
            camera_masks = camera_indices < self.embedding_appearance.in_dim
            if self.training and torch.any(~camera_masks):
                raise ValueError(f"`camera_indices` are out of range: {camera_indices}")
            camera_encodings = self.embedding_appearance(
                torch.clamp(
                    input=camera_indices.squeeze(-1),
                    max=self.embedding_appearance.in_dim - 1,
                ),
            )
            camera_encodings = torch.where(
                condition=camera_masks,
                input=camera_encodings,
                other=(
                    self.embedding_appearance.mean(dim=0)
                    if self.use_average_appearance_embedding
                    else torch.zeros_like(camera_encodings)
                ),
            )
        else:
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
        grid_encodings = self._forward_grid_encoder(positions)
        inputs = torch.cat(
            [
                positions,
                position_encodings,
                grid_encodings,
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
        sdf_gradients: jt.Float[torch.Tensor, " *B 3 "],
        geo_features: jt.Float[torch.Tensor, " *B {self.config.geo_feat_dim} "],
        camera_indices: jt.Int[torch.Tensor, " *B 1 "],
    ) -> jt.Float[torch.Tensor, " *B 3 "]:
        direction_encodings = self._forward_direction_encoder(directions)
        camera_encodings = self._forward_camera_encoder(camera_indices)
        inputs = torch.cat(
            [
                positions,
                direction_encodings,
                sdf_gradients,
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
        sdf_values, _ = torch.split(geo_outputs, (1, self.config.geo_feat_dim), dim=-1)
        return sdf_values

    @jaxtyped()
    @override
    def get_sdf(self, ray_samples: RaySamples) -> jt.Float[torch.Tensor, " *B 1 "]:
        ray_positions = ray_samples.frustums.get_start_positions()
        sdf_values = self.sdf(ray_positions)
        return sdf_values

    @jaxtyped()
    def finite_diff_grad(
        self,
        positions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> tuple[
        jt.Float[torch.Tensor, " *B 3 "],
        jt.Float[torch.Tensor, " 3 *B 1 "],
        jt.Float[torch.Tensor, " 3 *B 1 "],
    ]:
        finite_diff_epsilons = torch.full_like(positions, self.finite_diff_epsilon)
        finite_diff_epsilons = torch.diag_embed(finite_diff_epsilons, dim1=0, dim2=-1)
        pos_sdf_values = self.sdf(positions + finite_diff_epsilons)
        neg_sdf_values = self.sdf(positions - finite_diff_epsilons)
        sdf_gradients = (pos_sdf_values - neg_sdf_values) / (self.finite_diff_epsilon * 2.0)
        sdf_gradients = sdf_gradients.squeeze(-1).movedim(0, -1)
        return sdf_gradients, pos_sdf_values, neg_sdf_values

    def update_grid_encoding_mask(self, num_grid_levels: int) -> None:
        num_grid_features = num_grid_levels * self.config.features_per_level
        self.grid_encoding_mask[..., :num_grid_features] = 1.0
        self.grid_encoding_mask[..., num_grid_features:] = 0.0

    def update_finite_diff_epsilon(self, grid_resolution: float) -> None:
        scale_factor = torch.max(-torch.sub(*self.aabb))
        finite_diff_epsilon = scale_factor / grid_resolution
        self.finite_diff_epsilon = finite_diff_epsilon

    @override
    def get_outputs(
        self,
        ray_samples: RaySamples,
        return_alphas: bool = False,
    ) -> dict[str, torch.Tensor]:
        ray_positions = ray_samples.frustums.get_start_positions()
        ray_directions = ray_samples.frustums.directions

        # NOTE: `RaySamples.camera_indices.dtype` is set to `torch.float32` by `nerfstudio.exporter.texture_utils.export_textured_mesh`.
        # https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/exporter/texture_utils.py#L388
        camera_indices = ray_samples.camera_indices.long()

        with torch.set_grad_enabled(not self.config.enable_finite_diff_grad or self.training):
            ray_positions.requires_grad_(not self.config.enable_finite_diff_grad)
            geo_outputs = self.forward_geo_network(ray_positions)
            sdf_values, geo_features = torch.split(
                tensor=geo_outputs,
                split_size_or_sections=(1, self.config.geo_feat_dim),
                dim=-1,
            )

        if self.config.enable_finite_diff_grad:
            sdf_gradients, pos_sdf_values, neg_sdf_values = self.finite_diff_grad(ray_positions)
        else:
            [sdf_gradients] = torch.autograd.grad(
                outputs=sdf_values,
                inputs=ray_positions,
                grad_outputs=torch.ones_like(sdf_values),
                create_graph=self.training,
                retain_graph=self.training,
            )

        sdf_normals = torch.nn.functional.normalize(sdf_gradients, p=2, dim=-1)

        colors = self.forward_color_network(
            positions=ray_positions,
            directions=ray_directions,
            sdf_gradients=sdf_gradients,
            geo_features=geo_features,
            camera_indices=camera_indices,
        )

        outputs = {
            FieldHeadNames.RGB: colors,
            FieldHeadNames.SDF: sdf_values,
            FieldHeadNames.NORMALS: sdf_normals,
            FieldHeadNames.GRADIENT: sdf_gradients,
        }

        outputs.update(geo_features=geo_features)

        if self.config.enable_finite_diff_grad:
            outputs.update(
                pos_sdf_values=pos_sdf_values,
                neg_sdf_values=neg_sdf_values,
            )

        if return_alphas:
            alphas = self.get_alpha(ray_samples, sdf_values, sdf_gradients)
            outputs |= {FieldHeadNames.ALPHA: alphas}

        return outputs

    def set_progress_ratio(self, progress_ratio: float) -> None:
        self.progress_ratio = progress_ratio

    def get_progress_ratio(self) -> float:
        return self.progress_ratio

    def get_linear_anneal_ratio(self, start_ratio: float = 0.0, end_ratio: float = 1.0) -> float:
        progress_ratio = self.get_progress_ratio()
        linear_anneal_ratio = (progress_ratio - start_ratio) / max(end_ratio - start_ratio, 1.0e-6)
        linear_anneal_ratio = min(max(linear_anneal_ratio, 0.0), 1.0)
        return linear_anneal_ratio

    def get_cosine_anneal_ratio(self, start_ratio: float = 0.0, end_ratio: float = 1.0) -> float:
        linear_anneal_ratio = self.get_linear_anneal_ratio(start_ratio, end_ratio)
        cosine_anneal_ratio = math.cos(linear_anneal_ratio * math.pi + math.pi) * 0.5 + 0.5
        return cosine_anneal_ratio
