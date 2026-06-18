import dataclasses
from typing import Any, override

import jaxtyping as jt
import loguru
import torch
import torch.nn as nn

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig
from nerfstudio.utils.colormaps import apply_depth_colormap
from ssdp.fields import SDF
from ssdp.renderers import NormalEstimator, SphereTracer
from ssdp.utils.jaxtyping import jaxtyped


@dataclasses.dataclass
class SphereTracerConfig:
    num_iterations: int = 1000
    distance_threshold: float = 1.0e-3


@dataclasses.dataclass
class SurfaceModelMixinConfig(SurfaceModelConfig):
    anneal_end_step: int = 50000
    nlml_beta_anneal_min_value: float = 0.1
    nlml_beta_anneal_max_value: float = 1.0
    nlml_beta_anneal_end_ratio: float = 0.0
    nlml_color_loss_mult: float = 0.0
    volume_color_loss_mult: float = 1.0
    surface_color_loss_mult: float = 0.0
    train_sphere_tracer: SphereTracerConfig = dataclasses.field(
        default_factory=lambda: SphereTracerConfig(
            num_iterations=100,
        ),
    )
    eval_sphere_tracer: SphereTracerConfig = dataclasses.field(
        default_factory=lambda: SphereTracerConfig(
            num_iterations=1000,
        ),
    )


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


class SurfaceModelMixin(SurfaceModel):
    config: SurfaceModelMixinConfig
    field: SDF

    @override
    def populate_modules(self) -> None:
        super().populate_modules()

        # NOTE: Since `NeuSModel.anneal_end` is hard-coded, we override here.
        self.anneal_end = self.config.anneal_end_step

        # NOTE: Just to use AABB on the device.
        self.scene_box = SceneBox(aabb=self.field.aabb)

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

        self.train_sphere_tracer = SphereTracer(
            scene_box=self.scene_box,
            num_iterations=self.config.train_sphere_tracer.num_iterations,
            distance_threshold=self.config.train_sphere_tracer.distance_threshold,
        )
        self.eval_sphere_tracer = SphereTracer(
            scene_box=self.scene_box,
            num_iterations=self.config.eval_sphere_tracer.num_iterations,
            distance_threshold=self.config.eval_sphere_tracer.distance_threshold,
        )
        self.normal_estimator = NormalEstimator()

    @override
    def get_training_callbacks(
        self,
        training_callback_attributes: TrainingCallbackAttributes,
    ) -> list[TrainingCallback]:
        training_callbacks = super().get_training_callbacks(training_callback_attributes)

        def _set_progress_ratio(step: int) -> None:
            config = training_callback_attributes.trainer.config
            progress_ratio = min(1.0, step / config.max_num_iterations)
            self.field.set_progress_ratio(progress_ratio)

        training_callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=_set_progress_ratio,
            )
        )

        return training_callbacks

    @jaxtyped()
    def _sphere_trace(
        self,
        ray_bundle: RayBundle,
        foreground_masks: jt.Bool[torch.Tensor, " *B 1 "] | None = None,
        aabb_initialization: bool = False,
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

        sphere_tracer = self.train_sphere_tracer if self.training else self.eval_sphere_tracer
        surface_positions, foreground_masks = sphere_tracer(
            sdf=self.field.sdf,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            foreground_masks=foreground_masks,
            aabb_initialization=aabb_initialization,
        )
        surface_normals = self.normal_estimator(
            sdf=self.field.sdf,
            surface_positions=surface_positions,
        )

        geo_outputs = self.field.forward_geo_network(surface_positions)
        _, geo_features = torch.split(geo_outputs, (1, self.field.config.geo_feat_dim), dim=-1)
        surface_colors = self.field.forward_color_network(
            positions=surface_positions,
            directions=ray_directions,
            sdf_gradients=surface_normals,
            geo_features=geo_features,
            camera_indices=ray_bundle.camera_indices,
        )

        return surface_positions, surface_normals, surface_colors, foreground_masks

    @override
    def get_outputs(self, ray_bundle: RayBundle) -> dict[str, Any]:
        # NOTE: `ray_bundle` is modified in-place in `super().get_outputs()`
        # to create the one for background, so we create a copy here.
        outputs = super().get_outputs(dataclasses.replace(ray_bundle))
        outputs.update(ray_bundle=ray_bundle)
        # NOTE: Sometimes blended colors can be slightly out of bounds due to numerical issues,
        # yielding black images in visualization.
        if not self.training:
            rgb = torch.clamp(outputs["rgb"], min=0.0, max=1.0)
            outputs.update(rgb=rgb)
        return outputs

    @torch.no_grad()
    @override
    def get_outputs_for_camera_ray_bundle(self, ray_bundle: RayBundle) -> dict[str, Any]:
        # NOTE: Since all outputs except tensors are deleted in `super().get_outputs_for_camera_ray_bundle()`,
        # we store `ray_bundle` again for subsequent sphere tracing.
        outputs = super().get_outputs_for_camera_ray_bundle(ray_bundle)
        outputs.update(ray_bundle=ray_bundle)
        return outputs

    @jaxtyped()
    def _nlml_color_loss(
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

        nlml_beta_anneal_ratio = self.field.get_cosine_anneal_ratio(
            end_ratio=self.config.nlml_beta_anneal_end_ratio,
        )
        nlml_beta_anneal_min_value = mean_colors.new_tensor(self.config.nlml_beta_anneal_min_value)
        nlml_beta_anneal_max_value = mean_colors.new_tensor(self.config.nlml_beta_anneal_max_value)
        nlml_beta = torch.lerp(
            input=nlml_beta_anneal_min_value,
            end=nlml_beta_anneal_max_value,
            weight=nlml_beta_anneal_ratio,
        )

        laplace_distributions = torch.distributions.Laplace(mean_colors, nlml_beta)
        log_mixture_weights = torch.log(torch.clamp(mixture_weights, min=1.0e-6))
        nll_color_losses = -laplace_distributions.log_prob(target_colors.unsqueeze(-2))
        nll_color_losses = torch.sum(nll_color_losses, dim=-1, keepdim=True)
        nlml_color_losses = -torch.logsumexp(log_mixture_weights - nll_color_losses, dim=-2)
        nlml_color_loss = torch.mean(nlml_color_losses) * nlml_beta

        return nlml_color_loss

    @jaxtyped()
    def _surface_color_loss(
        self,
        ray_bundle: RayBundle,
        ray_samples: RaySamples,
        sdf_values: jt.Float[torch.Tensor, " *B S 1 "],
        sdf_gradients: jt.Float[torch.Tensor, " *B S 3 "],
        background_colors: jt.Float[torch.Tensor, " *B 3 "],
        target_colors: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " "]:
        prev_sdf_values = sdf_values[..., :-1, :]
        next_sdf_values = sdf_values[..., 1:, :]

        intersection_masks = (prev_sdf_values * next_sdf_values) <= 0.0
        intersection_indices = torch.argmax(intersection_masks.long(), dim=-2, keepdim=True)

        ray_intervals = torch.clamp(ray_samples.deltas[..., :-1, :], min=1.0e-6)
        sdf_gradients = (next_sdf_values - prev_sdf_values) / ray_intervals

        surface_distances = torch.gather(
            input=ray_samples.frustums.starts,
            index=intersection_indices,
            dim=-2,
        )
        surface_distances = surface_distances.squeeze(-2)

        sdf_values = torch.gather(
            input=sdf_values,
            index=intersection_indices,
            dim=-2,
        )
        sdf_values = sdf_values.squeeze(-2)

        sdf_gradients = torch.gather(
            input=sdf_gradients,
            index=intersection_indices,
            dim=-2,
        )
        sdf_gradients = sdf_gradients.squeeze(-2)

        sdf_gradients = torch.where(
            condition=sdf_gradients >= 0.0,
            input=torch.clamp(sdf_gradients, min=+1.0e-6),
            other=torch.clamp(sdf_gradients, max=-1.0e-6),
        )
        surface_distances = surface_distances - sdf_values / sdf_gradients

        ray_bundle = dataclasses.replace(ray_bundle, nears=surface_distances)
        foreground_masks = torch.any(intersection_masks, dim=-2)

        _, _, surface_colors, foreground_masks = self._sphere_trace(
            ray_bundle=ray_bundle,
            foreground_masks=foreground_masks,
            aabb_initialization=False,
        )
        surface_colors = torch.where(
            condition=foreground_masks,
            input=surface_colors,
            other=background_colors,
        )
        surface_color_losses = nn.functional.l1_loss(
            input=surface_colors,
            target=target_colors,
            reduction="none",
        )
        surface_color_loss = torch.div(
            input=torch.sum(surface_color_losses * foreground_masks),
            other=torch.clamp(torch.sum(foreground_masks), min=1.0e-6),
        )

        return surface_color_loss

    @jaxtyped()
    def _eikonal_loss(
        self,
        ray_samples: RaySamples,
        sdf_gradients: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " "]:
        positions = ray_samples.frustums.get_start_positions()
        inside_masks = self.scene_box.within(positions)
        sdf_gradients = sdf_gradients[inside_masks, ...]
        gradient_norms = torch.linalg.vector_norm(sdf_gradients, ord=2, dim=-1)
        eikonal_loss = nn.functional.mse_loss(gradient_norms, torch.ones_like(gradient_norms))
        return eikonal_loss

    @override
    def get_loss_dict(
        self,
        outputs: dict[str, Any],
        inputs: dict[str, Any],
        metrics: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = super().get_loss_dict(outputs, inputs, metrics)

        volume_color_loss: torch.Tensor = losses.pop("rgb_loss")
        volume_color_loss = volume_color_loss * self.config.volume_color_loss_mult
        losses.update(volume_color_loss=volume_color_loss)

        if self.training:
            if self.config.nlml_color_loss_mult or self.config.surface_color_loss_mult:
                target_colors: torch.Tensor = inputs["image"]
                target_colors = self.renderer_rgb.blend_background(target_colors)
                target_colors = target_colors.to(self.device)

                background_colors: torch.Tensor | None = outputs.get("bg_rgb")
                if background_colors is None:
                    background_colors = self.renderer_rgb.get_background_color(
                        background_color=self.renderer_rgb.background_color,
                        shape=(*target_colors.shape[:-1], 3),
                        device=target_colors.device,
                    )

            if self.config.nlml_color_loss_mult:
                foreground_colors: torch.Tensor = outputs["field_outputs"][FieldHeadNames.RGB]
                foreground_weights: torch.Tensor = outputs["weights"]
                background_weights: torch.Tensor = outputs["bg_transmittance"]
                nlml_color_loss = self._nlml_color_loss(
                    foreground_colors=foreground_colors,
                    foreground_weights=foreground_weights,
                    background_colors=background_colors,
                    background_weights=background_weights,
                    target_colors=target_colors,
                )
                nlml_color_loss = nlml_color_loss * self.config.nlml_color_loss_mult
                losses.update(nlml_color_loss=nlml_color_loss)

            if self.config.surface_color_loss_mult:
                ray_bundle: RayBundle = outputs["ray_bundle"]
                ray_samples: RaySamples = outputs["ray_samples"]
                sdf_values: torch.Tensor = outputs["field_outputs"][FieldHeadNames.SDF]
                sdf_gradients: torch.Tensor = outputs["field_outputs"][FieldHeadNames.GRADIENT]
                surface_color_loss = self._surface_color_loss(
                    ray_bundle=ray_bundle,
                    ray_samples=ray_samples,
                    sdf_values=sdf_values,
                    sdf_gradients=sdf_gradients,
                    background_colors=background_colors,
                    target_colors=target_colors,
                )
                surface_color_loss = surface_color_loss * self.config.surface_color_loss_mult
                losses.update(surface_color_loss=surface_color_loss)

            if self.config.eikonal_loss_mult:
                losses.pop("eikonal_loss")
                ray_samples: RaySamples = outputs["ray_samples"]
                sdf_gradients: torch.Tensor = outputs["field_outputs"][FieldHeadNames.GRADIENT]
                eikonal_loss = self._eikonal_loss(
                    ray_samples=ray_samples,
                    sdf_gradients=sdf_gradients,
                )
                eikonal_loss = eikonal_loss * self.config.eikonal_loss_mult
                losses.update(eikonal_loss=eikonal_loss)

        return losses

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
        ray_bundle: RayBundle = outputs["ray_bundle"]
        background_colors: torch.Tensor | None = outputs.get("bg_rgb")
        background_distances: torch.Tensor | None = outputs.get("bg_depth")

        (
            surface_positions,
            surface_normals,
            surface_colors,
            foreground_masks,
        ) = self._sphere_trace(
            ray_bundle=ray_bundle,
            aabb_initialization=True,
        )

        ray_directions = ray_bundle.directions
        surface_directions = surface_positions - ray_bundle.origins
        surface_distances = torch.sum(surface_directions * ray_directions, dim=-1, keepdim=True)

        direction_norms: torch.Tensor = ray_bundle.metadata["directions_norm"]
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
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        metrics, images = super().get_image_metrics_and_images(outputs, inputs)
        images.update(self._render_surface_images(outputs))
        return metrics, images
