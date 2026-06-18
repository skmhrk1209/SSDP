import dataclasses
from typing import Any, override

import jaxtyping as jt
import torch

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.base_surface_model import SurfaceModel
from ssdp.fields import SDF
from ssdp.utils.jaxtyping import jaxtyped


@dataclasses.dataclass
class NeuralangeloMixinConfig:
    enable_finite_diff_grad_schedule: bool = True
    enable_curvature_loss_schedule: bool = True
    enable_grid_encoding_schedule: bool = True
    grid_level_warmup_steps: int = 200
    steps_per_grid_level: int = 200
    init_num_grid_levels: int = 4
    curvature_loss_mult: float = 5.0e-4


class NeuralangeloMixin(SurfaceModel):
    config: NeuralangeloMixinConfig
    field: SDF

    @override
    def populate_modules(self) -> None:
        super().populate_modules()
        self.curvature_loss_mult = 1.0

    @override
    def get_training_callbacks(
        self,
        training_callback_attributes: TrainingCallbackAttributes,
    ) -> list[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        def _get_num_grid_levels(
            step: int,
            warmup_steps: int = self.config.grid_level_warmup_steps,
            steps_per_level: int = self.config.steps_per_grid_level,
            max_num_levels: int = self.field.config.num_levels,
        ) -> float:
            num_levels = (step - warmup_steps) / steps_per_level
            num_levels = min(max(num_levels, 1.0), max_num_levels)
            return num_levels

        def _get_grid_resolution(
            num_levels: float,
            base_resolution: int = self.field.config.base_res,
            scale_per_level: float = self.field.encoding.encoding_config["per_level_scale"],
        ) -> float:
            resolution = base_resolution * scale_per_level ** (num_levels - 1.0)
            return resolution

        if self.config.enable_finite_diff_grad_schedule:

            def _update_finite_diff_epsilon(step: int) -> None:
                grid_levels = _get_num_grid_levels(step)
                grid_resolution = _get_grid_resolution(grid_levels)
                self.field.update_finite_diff_epsilon(grid_resolution)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=_update_finite_diff_epsilon,
                )
            )

        if self.config.enable_curvature_loss_schedule:

            def _update_curvature_loss_mult(step: int) -> None:
                if step < self.config.grid_level_warmup_steps:
                    curvature_loss_mult = step / self.config.grid_level_warmup_steps
                else:
                    num_grid_levels = _get_num_grid_levels(step)
                    grid_resolution = _get_grid_resolution(num_grid_levels)
                    curvature_loss_mult = self.field.config.base_res / grid_resolution
                self.curvature_loss_mult = curvature_loss_mult

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=_update_curvature_loss_mult,
                )
            )

        if self.config.enable_grid_encoding_schedule:

            def _update_grid_encoding_mask(step: int) -> None:
                num_grid_levels = int(_get_num_grid_levels(step))
                num_grid_levels = max(num_grid_levels, self.config.init_num_grid_levels)
                self.field.update_grid_encoding_mask(num_grid_levels)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=_update_grid_encoding_mask,
                )
            )

        return callbacks

    @jaxtyped()
    def _curvature_loss(
        self,
        ray_samples: RaySamples,
        sdf_values: jt.Float[torch.Tensor, " *B 1 "],
        pos_sdf_values: jt.Float[torch.Tensor, " 3 *B 1 "],
        neg_sdf_values: jt.Float[torch.Tensor, " 3 *B 1 "],
    ) -> jt.Float[torch.Tensor, " "]:
        positions = ray_samples.frustums.get_start_positions()
        inside_masks = self.scene_box.within(positions)
        sdf_values = sdf_values[inside_masks, ...]
        pos_sdf_values = pos_sdf_values[:, inside_masks, ...]
        neg_sdf_values = neg_sdf_values[:, inside_masks, ...]
        hessians = torch.div(
            input=pos_sdf_values + neg_sdf_values - 2.0 * sdf_values,
            other=self.field.finite_diff_epsilon**2.0,
        )
        curvature_losses = torch.abs(torch.sum(hessians, dim=0))
        curvature_loss = torch.mean(curvature_losses)
        return curvature_loss

    @override
    def get_loss_dict(
        self,
        outputs: dict[str, Any],
        inputs: dict[str, Any],
        metrics: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = super().get_loss_dict(outputs, inputs, metrics)
        if self.training and self.config.curvature_loss_mult:
            ray_samples: RaySamples = outputs["ray_samples"]
            sdf_values: torch.Tensor = outputs["field_outputs"][FieldHeadNames.SDF]
            pos_sdf_values: torch.Tensor = outputs["field_outputs"]["pos_sdf_values"]
            neg_sdf_values: torch.Tensor = outputs["field_outputs"]["neg_sdf_values"]
            curvature_loss = self._curvature_loss(
                ray_samples=ray_samples,
                sdf_values=sdf_values,
                pos_sdf_values=pos_sdf_values,
                neg_sdf_values=neg_sdf_values,
            )
            curvature_loss = (
                curvature_loss * self.curvature_loss_mult * self.config.curvature_loss_mult
            )
            losses.update(curvature_loss=curvature_loss)
        return losses
