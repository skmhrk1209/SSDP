import dataclasses
import enum
import functools
from collections.abc import Callable
from typing import Any, Concatenate, Literal, override

import jaxtyping as jt
import loguru
import torch
import torch.utils._pytree as pytree
from torch import nn

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.model_components.losses import interlevel_loss as get_interlevel_loss
from nerfstudio.model_components.ray_samplers import UniformSampler
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.neus_facto import NeuSFactoModel, NeuSFactoModelConfig
from nerfstudio.utils.colormaps import apply_depth_colormap
from ssdp.fields import NeuSField, NeuSFieldConfig
from ssdp.renderers import SphereTracer
from ssdp.utils.function import signature_of
from ssdp.utils.jaxtyping import jaxtyped


def _chunked[**P, R](
    function: Callable[Concatenate[RayBundle, P], R],
    chunk_size: int,
) -> Callable[Concatenate[RayBundle, P], R]:
    @functools.wraps(function)
    def wrapper(
        camera_ray_bundle: RayBundle,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        outputs_list = []
        for start_index in range(0, len(camera_ray_bundle), chunk_size):
            end_index = start_index + chunk_size
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_index, end_index)
            outputs = function(ray_bundle, *args, **kwargs)
            outputs_list.append(outputs)
        outputs = pytree.tree_map(
            lambda *chunks: torch.cat(chunks).reshape(*camera_ray_bundle.shape, -1),
            *outputs_list,
        )
        return outputs

    return wrapper


class CustomAABBBoxCollider(AABBBoxCollider):
    def __init__(self, *args: Any, far_plane: float | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.far_plane = far_plane

    @jaxtyped()
    @override
    def _intersect_with_aabb(
        self,
        rays_o: jt.Float[torch.Tensor, " B 3 "],
        rays_d: jt.Float[torch.Tensor, " B 3 "],
        aabb: jt.Float[torch.Tensor, " 2 3 "],
    ) -> tuple[
        jt.Float[torch.Tensor, " B "],
        jt.Float[torch.Tensor, " B "],
    ]:
        nears, fars = super()._intersect_with_aabb(rays_o, rays_d, aabb)
        if self.far_plane is not None:
            nears = torch.clamp(nears, max=self.far_plane)
            fars = torch.clamp(fars, max=self.far_plane)
        return nears, fars


class SamplerType(enum.StrEnum):
    NEUS = enum.auto()
    PROPOSAL = enum.auto()
    UNIFORM = enum.auto()


class LikelihoodDistributionType(enum.StrEnum):
    NORMAL = enum.auto()
    LAPLACE = enum.auto()

    def get(self) -> type[torch.distributions.Normal | torch.distributions.Laplace]:
        return getattr(torch.distributions, self.capitalize())


class RegularizationSampleDomain(enum.StrEnum):
    RAY = enum.auto()
    AABB = enum.auto()


@dataclasses.dataclass
class SphereTracerConfig:
    num_iterations: int = 1000
    distance_threshold: float = 1.0e-3


@dataclasses.dataclass
class NeuSModelConfig(NeuSFactoModelConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: NeuSModel,
    )
    sdf_field: SDFFieldConfig = dataclasses.field(
        default_factory=NeuSFieldConfig,
    )
    sphere_tracer: SphereTracerConfig = dataclasses.field(
        default_factory=SphereTracerConfig,
    )
    sampler_type: SamplerType = SamplerType.NEUS
    num_uniform_samples: int = 512
    color_nll_loss_mult: float = 1.0
    color_nlml_loss_mult: float = 0.0
    curvature_loss_mult: float = 0.0
    foreground_mask_loss_mult: float = 0.0
    color_distribution_type: LikelihoodDistributionType = LikelihoodDistributionType.LAPLACE
    grad_norm_distribution_type: LikelihoodDistributionType = LikelihoodDistributionType.NORMAL
    laplacian_distribution_type: LikelihoodDistributionType = LikelihoodDistributionType.NORMAL
    regularization_sample_domain: RegularizationSampleDomain = RegularizationSampleDomain.RAY
    num_regularization_samples: int = 100000
    regularize_outside_aabb: bool = False
    render_surface_images: bool = False
    background_color: Literal["black"] = "black"


class NeuSModel(NeuSFactoModel):
    config: NeuSModelConfig
    field: NeuSField

    @override
    def populate_modules(self) -> None:
        super().populate_modules()

        # NOTE: Just to use AABB on the device.
        self.scene_box = SceneBox(aabb=self.field.aabb)

        if self.config.sampler_type == SamplerType.UNIFORM:
            self.uniform_sampler = UniformSampler(
                num_samples=self.config.num_uniform_samples,
                single_jitter=self.config.use_single_jitter,
            )

        if self.config.far_plane > self.config.far_plane_bg:
            loguru.logger.warning(
                f"`SurfaceModelConfig.far_plane` <{self.config.far_plane}> is greater than "
                f"`SurfaceModelConfig.far_plane_bg` <{self.config.far_plane_bg}>, "
                f"yielding negative sampling interval and thus NaNs in the background model. "
                f"Therefore `SurfaceModelConfig.far_plane_bg` is used for `far_plane` in `CustomAABBBoxCollider`.",
            )

        # NOTE: There is a bug in the background rendering in `SurfaceModel`,
        # where `ray_bundle.fars` can be smaller than `ray_bundle.nears`, yielding negative sampling interval and thus NaNs in the background model.
        # Therefore, we apply far plane clipping in `CustomAABBBoxCollider` to address this issue.
        # https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/models/base_surface_model.py#L225
        self.collider = CustomAABBBoxCollider(
            scene_box=self.scene_box,
            near_plane=self.config.near_plane,
            far_plane=min(self.config.far_plane, self.config.far_plane_bg),
        )

        self.sphere_tracer = SphereTracer(
            scene_box=self.scene_box,
            num_iterations=self.config.sphere_tracer.num_iterations,
            distance_threshold=self.config.sphere_tracer.distance_threshold,
        )

    @override
    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        param_groups = super().get_param_groups()
        param_groups.update(self.field.get_param_groups())
        if self.config.sampler_type != SamplerType.PROPOSAL:
            param_groups.pop("proposal_networks")
        return param_groups

    @override
    def get_training_callbacks(
        self,
        training_callback_attributes: TrainingCallbackAttributes,
    ) -> list[TrainingCallback]:
        training_callbacks = super().get_training_callbacks(training_callback_attributes)

        def set_progress_ratio(step: int) -> None:
            assert training_callback_attributes.trainer is not None
            trainer_config = training_callback_attributes.trainer.config
            progress_ratio = min(1.0, step / trainer_config.max_num_iterations)
            self.field.progress_ratio = progress_ratio

        training_callbacks.extend(
            [
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_progress_ratio,
                ),
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.field.on_train_iteration_start,
                ),
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.field.on_train_iteration_end,
                ),
            ],
        )

        return training_callbacks

    @override
    def sample_and_forward_field(self, ray_bundle: RayBundle) -> dict[str, Any]:
        match self.config.sampler_type:
            case SamplerType.NEUS:
                outputs = super(NeuSFactoModel, self).sample_and_forward_field(ray_bundle)
            case SamplerType.PROPOSAL:
                outputs = super().sample_and_forward_field(ray_bundle)
            case SamplerType.UNIFORM:
                ray_samples: RaySamples = self.uniform_sampler(ray_bundle)
                field_outputs = self.field(ray_samples, return_alphas=True)
                opacities = field_outputs[FieldHeadNames.ALPHA]
                weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
                    alphas=opacities,
                    weights_only=False,
                )
                bg_transmittance = transmittance[..., -1, :]
                outputs = dict(
                    ray_samples=ray_samples,
                    field_outputs=field_outputs,
                    weights=weights,
                    bg_transmittance=bg_transmittance,
                )
        return outputs

    @override
    def get_outputs(self, ray_bundle: RayBundle) -> dict[str, Any]:
        # NOTE: `ray_bundle` is modified in-place in `super().get_outputs()`
        # to create the one for background, so we create a copy here.
        outputs: dict[str, Any] = super().get_outputs(dataclasses.replace(ray_bundle))
        outputs.update(ray_bundle=ray_bundle)
        # NOTE: Sometimes blended colors can be slightly out of bounds due to numerical issues,
        # yielding black images in visualization.
        if not self.training:
            rgb = torch.clamp(outputs["rgb"], min=0.0, max=1.0)
            outputs.update(rgb=rgb)
        return outputs

    @torch.no_grad()
    @override
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> dict[str, Any]:
        # NOTE: Since all outputs except tensors are deleted in `super().get_outputs_for_camera_ray_bundle()`,
        # we store `ray_bundle` again for subsequent sphere tracing.
        outputs: dict[str, Any] = super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        outputs.update(camera_ray_bundle=camera_ray_bundle)
        return outputs

    @signature_of(get_outputs)
    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return super().forward(*args, **kwargs)

    @signature_of(forward)
    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)

    @jaxtyped()
    def _color_nll_loss(
        self,
        mean_colors: jt.Float[torch.Tensor, " *B 3 "],
        target_colors: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " "]:
        ColorDistribution = self.config.color_distribution_type.get()
        color_distributions = ColorDistribution(loc=mean_colors, scale=1.0)
        color_nll_losses = -color_distributions.log_prob(target_colors)
        color_nll_losses = torch.sum(color_nll_losses, dim=-1)
        color_nll_loss = torch.mean(color_nll_losses)
        return color_nll_loss

    @jaxtyped()
    def _color_nlml_loss(
        self,
        foreground_colors: jt.Float[torch.Tensor, " *B S 3 "],
        foreground_weights: jt.Float[torch.Tensor, " *B S 1 "],
        background_colors: jt.Float[torch.Tensor, " *B 3 "],
        background_weights: jt.Float[torch.Tensor, " *B 1 "],
        target_colors: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " "]:
        mean_colors = torch.cat(
            [
                foreground_colors,
                background_colors.unsqueeze(-2),
            ],
            dim=-2,
        )
        mixture_weights = torch.cat(
            [
                foreground_weights,
                background_weights.unsqueeze(-2),
            ],
            dim=-2,
        )
        ColorDistribution = self.config.color_distribution_type.get()
        color_distributions = ColorDistribution(loc=mean_colors, scale=1.0)
        color_nll_losses = -color_distributions.log_prob(target_colors.unsqueeze(-2))
        color_nll_losses = torch.sum(color_nll_losses, dim=-1, keepdim=True)
        log_mixture_weights = torch.log(torch.clamp(mixture_weights, min=1.0e-6))
        color_nlml_losses = -torch.logsumexp(log_mixture_weights - color_nll_losses, dim=-2)
        color_nlml_loss = torch.mean(color_nlml_losses)
        return color_nlml_loss

    @jaxtyped()
    def _foreground_mask_loss(
        self,
        foreground_weights: jt.Float[torch.Tensor, " *B S 1 "],
        foreground_masks: jt.Float[torch.Tensor, " *B 1 "],
        epsilon: float = 1.0e-3,
    ) -> jt.Float[torch.Tensor, " "]:
        foreground_masses = torch.sum(foreground_weights, dim=-2)
        foreground_masses = torch.clamp(foreground_masses, min=epsilon, max=1.0 - epsilon)
        foreground_mask_distributions = torch.distributions.Bernoulli(foreground_masses)
        foreground_mask_losses = -foreground_mask_distributions.log_prob(foreground_masks)
        foreground_mask_loss = torch.mean(foreground_mask_losses)
        return foreground_mask_loss

    @jaxtyped()
    def _eikonal_loss(
        self,
        sdf_gradients: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " "]:
        sdf_grad_norms: torch.Tensor = torch.linalg.vector_norm(sdf_gradients, ord=2, dim=-1)
        GradNormDistribution = self.config.grad_norm_distribution_type.get()
        grad_norm_distribution = GradNormDistribution(loc=sdf_grad_norms, scale=1.0)
        eikonal_losses = -grad_norm_distribution.log_prob(sdf_grad_norms.new_ones(()))
        eikonal_loss = torch.mean(eikonal_losses)
        return eikonal_loss

    @jaxtyped()
    def _curvature_loss(
        self,
        sdf_laplacians: jt.Float[torch.Tensor, " *B 1 "],
    ) -> jt.Float[torch.Tensor, " "]:
        LaplacianDistribution = self.config.laplacian_distribution_type.get()
        laplacian_distribution = LaplacianDistribution(loc=sdf_laplacians, scale=1.0)
        curvature_losses = -laplacian_distribution.log_prob(sdf_laplacians.new_zeros(()))
        curvature_loss = torch.mean(curvature_losses)
        return curvature_loss

    @override
    def get_metrics_dict(
        self,
        outputs: dict[str, Any],
        inputs: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        metrics = super().get_metrics_dict(outputs, inputs)

        mean_colors: torch.Tensor = outputs["rgb"]
        target_colors: torch.Tensor = inputs["image"]
        target_colors = self.renderer_rgb.blend_background(target_colors)
        target_colors = target_colors.to(mean_colors)

        color_nll_loss = self._color_nll_loss(
            mean_colors=mean_colors,
            target_colors=target_colors,
        )
        metrics.update(color_nll_loss=color_nll_loss)

        if not self.training:
            return metrics

        field_outputs: dict[FieldHeadNames | str, torch.Tensor] = outputs["field_outputs"]

        if self.config.color_nlml_loss_mult:
            foreground_colors = field_outputs[FieldHeadNames.RGB]
            foreground_weights: torch.Tensor = outputs["weights"]
            background_weights: torch.Tensor = outputs["bg_transmittance"]
            background_colors: torch.Tensor | None = outputs.get("bg_rgb")
            if background_colors is None:
                background_colors = self.renderer_rgb.get_background_color(
                    background_color=self.renderer_rgb.background_color,
                    shape=(*target_colors.shape[:-1], 3),
                    device=target_colors.device,
                )
            color_nlml_loss = self._color_nlml_loss(
                foreground_colors=foreground_colors,
                foreground_weights=foreground_weights,
                background_colors=background_colors,
                background_weights=background_weights,
                target_colors=target_colors,
            )
            metrics.update(color_nlml_loss=color_nlml_loss)

        if self.config.foreground_mask_loss_mult:
            foreground_weights: torch.Tensor = outputs["weights"]
            foreground_masks: torch.Tensor = inputs["mask"]
            foreground_masks = foreground_masks.to(foreground_weights)
            foreground_mask_loss = self._foreground_mask_loss(
                foreground_weights=foreground_weights,
                foreground_masks=foreground_masks,
            )
            metrics.update(foreground_mask_loss=foreground_mask_loss)

        if self.config.eikonal_loss_mult or self.config.curvature_loss_mult:
            match self.config.regularization_sample_domain:
                case RegularizationSampleDomain.RAY:
                    ray_samples: RaySamples = outputs["ray_samples"]
                    positions = ray_samples.frustums.get_start_positions()
                    inside_masks = self.scene_box.within(positions)

                    if self.config.eikonal_loss_mult:
                        sdf_gradients = field_outputs[FieldHeadNames.GRADIENT]
                        if not self.config.regularize_outside_aabb:
                            sdf_gradients = sdf_gradients[inside_masks, ...]
                        eikonal_loss = self._eikonal_loss(sdf_gradients)
                        metrics.update(eikonal_loss=eikonal_loss)

                    if self.config.curvature_loss_mult:
                        if self.field.config.numerical_gradient:
                            sdf_laplacians = field_outputs["laplacian"]
                        else:
                            sdf_values = field_outputs[FieldHeadNames.SDF]
                            _, sdf_laplacians = self.field.numerical_gradient(
                                positions=positions,
                                sdf_values=sdf_values,
                                return_laplacian=True,
                            )
                        if not self.config.regularize_outside_aabb:
                            sdf_laplacians = sdf_laplacians[inside_masks, ...]
                        curvature_loss = self._curvature_loss(sdf_laplacians)
                        metrics.update(curvature_loss=curvature_loss)

                case RegularizationSampleDomain.AABB:
                    positions = torch.rand(
                        (self.config.num_regularization_samples, 3),
                        device=self.device,
                    )
                    positions = torch.lerp(
                        input=self.field.aabb[0, ...],
                        end=self.field.aabb[1, ...],
                        weight=positions,
                    )
                    sdf_values = self.field.sdf(positions)

                    if self.field.config.numerical_gradient and self.config.curvature_loss_mult:
                        sdf_gradients, sdf_laplacians = self.field.numerical_gradient(
                            positions=positions,
                            sdf_values=sdf_values,
                            return_laplacian=True,
                        )
                    else:
                        sdf_gradients = self.field.gradient(
                            positions=positions,
                            sdf_values=sdf_values,
                        )
                        sdf_laplacians = None

                    if self.config.eikonal_loss_mult:
                        eikonal_loss = self._eikonal_loss(sdf_gradients)
                        metrics.update(eikonal_loss=eikonal_loss)

                    if self.config.curvature_loss_mult:
                        if sdf_laplacians is None:
                            _, sdf_laplacians = self.field.numerical_gradient(
                                positions=positions,
                                sdf_values=sdf_values,
                                return_laplacian=True,
                            )
                        curvature_loss = self._curvature_loss(sdf_laplacians)
                        metrics.update(curvature_loss=curvature_loss)

        if self.config.sampler_type == SamplerType.PROPOSAL and self.config.interlevel_loss_mult:
            weights_list: list[torch.Tensor] = outputs["weights_list"]
            ray_samples_list: list[RaySamples] = outputs["ray_samples_list"]
            interlevel_loss = get_interlevel_loss(weights_list, ray_samples_list)
            metrics.update(interlevel_loss=interlevel_loss)

        return metrics

    @override
    def get_loss_dict(
        self,
        outputs: dict[str, Any],
        inputs: dict[str, Any],
        metrics: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        losses = {
            key: value * getattr(self.config, f"{key}_mult")
            for key, value in metrics.items()
            if key.endswith("loss")
        }
        return losses

    @jaxtyped()
    def _sphere_trace(
        self,
        ray_bundle: RayBundle,
        aabb_initialization: bool = True,
    ) -> tuple[
        jt.Float[torch.Tensor, " *B 3 "],
        jt.Float[torch.Tensor, " *B 3 "],
        jt.Float[torch.Tensor, " *B 3 "],
        jt.Bool[torch.Tensor, " *B 1 "],
    ]:
        ray_origins = ray_bundle.origins
        ray_directions = ray_bundle.directions

        if ray_bundle.nears is not None:
            ray_origins = ray_origins + ray_directions * ray_bundle.nears

        surface_positions, foreground_masks = self.sphere_tracer(
            sdf=self.field.sdf,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            aabb_initialization=aabb_initialization,
        )

        surface_normals = self.field.gradient(surface_positions)
        surface_normals = nn.functional.normalize(surface_normals, p=2, dim=-1)

        geo_outputs = self.field.forward_geo_network(surface_positions)
        _, geo_features = torch.split(geo_outputs, [1, self.field.config.geo_feat_dim], dim=-1)

        assert ray_bundle.camera_indices is not None
        surface_colors = self.field.forward_color_network(
            positions=surface_positions,
            directions=ray_directions,
            sdf_normals=surface_normals,
            geo_features=geo_features,
            camera_indices=ray_bundle.camera_indices,
        )

        return surface_positions, surface_normals, surface_colors, foreground_masks

    @jaxtyped()
    def _render_surface_color(
        self,
        surface_colors: jt.Float[torch.Tensor, " *B 3 "],
        foreground_masks: jt.Bool[torch.Tensor, " *B 1 "],
        background_colors: jt.Float[torch.Tensor, " *B 3 "] | None = None,
    ) -> jt.Float[torch.Tensor, " *B 3 "]:
        if background_colors is None:
            background_colors = self.renderer_rgb.get_background_color(
                background_color=self.renderer_rgb.background_color,
                shape=surface_colors.shape,
                device=surface_colors.device,
            )
        surface_colors = torch.where(
            condition=foreground_masks,
            input=surface_colors,
            other=background_colors,
        )
        return surface_colors

    @jaxtyped()
    def _render_surface_depth(
        self,
        surface_depths: jt.Float[torch.Tensor, " *B 1 "],
        foreground_masks: jt.Bool[torch.Tensor, " *B 1 "],
        background_depths: jt.Float[torch.Tensor, " *B 1 "] | None = None,
    ) -> jt.Float[torch.Tensor, " *B 3 "]:
        if background_depths is None:
            surface_depths = apply_depth_colormap(surface_depths)
            surface_depths = torch.where(
                condition=foreground_masks,
                input=surface_depths,
                other=torch.zeros_like(surface_depths),
            )
        else:
            surface_depths = torch.where(
                condition=foreground_masks,
                input=surface_depths,
                other=background_depths,
            )
            surface_depths = apply_depth_colormap(surface_depths)
        return surface_depths

    @jaxtyped()
    def _render_surface_normal(
        self,
        surface_normals: jt.Float[torch.Tensor, " *B 3 "],
        foreground_masks: jt.Bool[torch.Tensor, " *B 1 "],
    ) -> jt.Float[torch.Tensor, " *B 3 "]:
        surface_normals = (surface_normals + 1.0) / 2.0
        surface_normals = torch.where(
            condition=foreground_masks,
            input=surface_normals,
            other=torch.zeros_like(surface_normals),
        )
        return surface_normals

    def _render_surface_images(self, outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        camera_ray_bundle: RayBundle = outputs["camera_ray_bundle"]
        background_colors: torch.Tensor | None = outputs.get("bg_rgb")
        background_distances: torch.Tensor | None = outputs.get("bg_depth")

        sphere_tracer = _chunked(
            function=self._sphere_trace,
            chunk_size=self.config.eval_num_rays_per_chunk,
        )
        (
            surface_positions,
            surface_normals,
            surface_colors,
            foreground_masks,
        ) = sphere_tracer(camera_ray_bundle)

        ray_directions = camera_ray_bundle.directions
        surface_directions = surface_positions - camera_ray_bundle.origins
        surface_distances = torch.sum(surface_directions * ray_directions, dim=-1, keepdim=True)

        direction_norms = camera_ray_bundle.metadata["directions_norm"]
        surface_depths = surface_distances / direction_norms

        background_depths = None
        if background_distances is not None:
            background_depths = background_distances / direction_norms

        surface_colors = self._render_surface_color(
            surface_colors=surface_colors,
            foreground_masks=foreground_masks,
            background_colors=background_colors,
        )
        surface_depths = self._render_surface_depth(
            surface_depths=surface_depths,
            foreground_masks=foreground_masks,
            background_depths=background_depths,
        )
        surface_normals = self._render_surface_normal(
            surface_normals=surface_normals,
            foreground_masks=foreground_masks,
        )
        foreground_masks = foreground_masks.float()

        images = dict(
            surface_colors=surface_colors,
            surface_depths=surface_depths,
            surface_normals=surface_normals,
            foreground_masks=foreground_masks,
        )

        return images

    @torch.no_grad()
    @override
    def get_image_metrics_and_images(
        self,
        outputs: dict[str, Any],
        inputs: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        match self.config.sampler_type:
            case SamplerType.PROPOSAL:
                metrics, images = super().get_image_metrics_and_images(outputs, inputs)
            case SamplerType.NEUS | SamplerType.UNIFORM:
                metrics, images = super(NeuSFactoModel, self).get_image_metrics_and_images(
                    outputs, inputs
                )
        if self.config.render_surface_images:
            surface_images = self._render_surface_images(outputs)
            images.update(surface_images)
        return metrics, images
