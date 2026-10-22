import dataclasses
import enum
from collections.abc import Callable
from typing import Any, override

import jaxtyping as jt
import torch
import torch.nn as nn
import torch.utils.checkpoint

from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig
from ssdp.fields.ldm_mixin import LDMMixin
from ssdp.priors.ldm import edm_noise_levels
from ssdp.utils.jaxtyping import jaxtyped

type SDFType = Callable[[jt.Float[torch.Tensor, " *B 3 "]], jt.Float[torch.Tensor, " *B 1 "]]


class OptimizationPhase(enum.StrEnum):
    LATENT = enum.auto()
    FIELD = enum.auto()


class LatentInitSource(enum.StrEnum):
    FIELD = enum.auto()
    PFODE = enum.auto()


def _chunked(sdf: SDFType, chunk_size: int) -> SDFType:
    @jaxtyped()
    def wrapper(positions: jt.Float[torch.Tensor, " *B 3 "]) -> jt.Float[torch.Tensor, " *B 1 "]:
        chunks = positions.flatten(0, -2).split(chunk_size, dim=0)
        sdf_values = torch.cat(list(map(sdf, chunks)), dim=0)
        sdf_values = sdf_values.reshape(*positions.shape[:-1], 1)
        return sdf_values

    return wrapper


@dataclasses.dataclass
class MAPMixinConfig(SurfaceModelConfig):
    num_steps_anneal: int = 100
    num_steps_latent: int = 100
    num_steps_pfode: int = 10
    max_noise_level_anneal: float = 10.0
    min_noise_level_anneal: float = 0.1
    min_noise_level_pfode: float = 0.001
    field_prior_scale_mult: float = 1.0e-4
    field_prior_loss_mult: float = 1.0
    latent_prior_loss_mult: float = 1.0
    sdf_num_samples: int = 100000
    sdf_resolution: int = 64
    sdf_chunk_size: int = 1 << 18
    latent_init_source: LatentInitSource = LatentInitSource.FIELD


class MAPMixin(SurfaceModel):
    config: MAPMixinConfig
    field: LDMMixin
    noise_levels: torch.Tensor

    @override
    def populate_modules(self) -> None:
        super().populate_modules()
        noise_levels = edm_noise_levels(
            max_noise_level=torch.as_tensor(self.config.max_noise_level_anneal),
            min_noise_level=torch.as_tensor(self.config.min_noise_level_anneal),
            num_ode_steps=self.config.num_steps_anneal,
        )
        self.register_buffer(
            name="noise_levels",
            tensor=noise_levels,
            persistent=False,
        )

    @override
    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        param_groups = super().get_param_groups()
        param_groups.pop("fields")
        param_groups.update(
            field=list(self.field.field_parameters()),
            latent=list(self.field.latent_parameters()),
            nuisance=list(self.field.nuisance_parameters()),
        )
        return param_groups

    @jaxtyped()
    def _discretize_sdf(self, sdf: SDFType) -> jt.Float[torch.Tensor, " D H W 1 "]:
        positions = [
            torch.linspace(*range, self.config.sdf_resolution + 1, device=self.device)[..., :-1]
            - torch.sub(*range) / self.config.sdf_resolution / 2.0
            for range in self.field.aabb.T
        ]
        positions = torch.meshgrid(*reversed(positions), indexing="ij")
        positions = torch.stack(list(reversed(positions)), dim=-1)
        sdf_grid = _chunked(sdf, self.config.sdf_chunk_size)(positions)
        return sdf_grid

    @jaxtyped()
    def _sample_sdf(
        self,
        sdf: SDFType,
    ) -> tuple[jt.Float[torch.Tensor, " *B 1 "], jt.Float[torch.Tensor, " *B 3 "]]:
        positions = torch.rand(self.config.sdf_num_samples, 3, device=self.device)
        positions = torch.lerp(*self.field.aabb, positions)
        sdf_values = _chunked(sdf, self.config.sdf_chunk_size)(positions)
        return sdf_values, positions

    @jaxtyped()
    def _interpolate_sdf(
        self,
        sdf_grid: jt.Float[torch.Tensor, " D H W 1 "],
        positions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        aabb_min, aabb_max = self.field.aabb
        voxel_size = (aabb_max - aabb_min) / self.config.sdf_resolution
        positions = (positions - aabb_min - voxel_size / 2.0) / (aabb_max - aabb_min - voxel_size)
        positions = positions * 2.0 - 1.0
        sdf_values = nn.functional.grid_sample(
            input=sdf_grid.movedim(-1, 0).unsqueeze(0),
            grid=positions.reshape(1, 1, 1, -1, 3),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sdf_values = sdf_values.reshape(*positions.shape[:-1], 1)
        return sdf_values

    @torch.no_grad()
    def _on_latent_phase_start(self, noise_level: jt.Float[torch.Tensor, " "]) -> None:
        self.noise_level = noise_level
        self.denoised_latents = self.field.ldm.tweedie(
            noisy_latents=self.field.latents,
            noise_level=noise_level,
        )
        self.sdf_grid = self._discretize_sdf(self.field.sdf)
        self.optimization_phase = OptimizationPhase.LATENT

    @torch.no_grad()
    def _on_field_phase_start(self, noise_level: jt.Float[torch.Tensor, " "]) -> None:
        self.noise_level = noise_level
        sdf = self.field.decode(self.field.latents)
        self.sdf_grid = self._discretize_sdf(sdf)
        self.optimization_phase = OptimizationPhase.FIELD

    @override
    def get_training_callbacks(
        self,
        training_callback_attributes: TrainingCallbackAttributes,
    ) -> list[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        num_steps_train = training_callback_attributes.trainer.config.max_num_iterations
        num_steps_stage, remainder = divmod(num_steps_train, self.config.num_steps_anneal)
        assert not remainder
        assert self.config.num_steps_latent < num_steps_stage

        def on_train_start(step: int) -> None:
            if self.field.ldm.config.condition_image_file is None:
                self.field.ldm.uncondition()
            else:
                self.field.ldm.condition()
            with torch.no_grad():
                if self.config.latent_init_source is LatentInitSource.FIELD:
                    latents = self.field.encode()
                    self.field.latents.copy_(latents)
                elif self.config.latent_init_source is LatentInitSource.PFODE:
                    noise_level = self.noise_levels[0, ...]
                    noisy_latents = self.field.ldm.forward_sde(
                        latents=self.field.latents,
                        noise_level=noise_level,
                    )
                    denoised_latents = self.field.ldm.reverse_pfode(
                        noisy_latents=noisy_latents,
                        max_noise_level=noise_level,
                        min_noise_level=noise_level.new_tensor(self.config.min_noise_level_pfode),
                        num_ode_steps=self.config.num_steps_pfode,
                    )
                    self.field.latents.copy_(denoised_latents)

        def on_latent_phase_start(step: int) -> None:
            annealing_step = step // num_steps_stage
            noise_level = self.noise_levels[annealing_step, ...]
            self._on_latent_phase_start(noise_level)

        def on_field_phase_start(step: int) -> None:
            annealing_step = step // num_steps_stage
            noise_level = self.noise_levels[annealing_step, ...]
            self._on_field_phase_start(noise_level)

        callbacks.extend(
            [
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=num_steps_train,
                    func=on_train_start,
                ),
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=num_steps_stage,
                    func=on_latent_phase_start,
                ),
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    iters=tuple(
                        range(self.config.num_steps_latent, num_steps_train, num_steps_stage)
                    ),
                    func=on_field_phase_start,
                ),
            ]
        )
        return callbacks

    @jaxtyped()
    def _latent_prior_loss(self) -> jt.Float[torch.Tensor, " "]:
        latent_prior_distributions = torch.distributions.Normal(
            loc=self.denoised_latents,
            scale=self.noise_level,
        )
        latent_prior_losses = -latent_prior_distributions.log_prob(self.field.latents)
        latent_prior_loss = torch.mean(latent_prior_losses)
        return latent_prior_loss

    @jaxtyped()
    def _field_prior_loss(self) -> jt.Float[torch.Tensor, " "]:
        if self.optimization_phase is OptimizationPhase.LATENT:
            sdf = self.field.decode(self.field.latents)
            latent_sdf_values, positions = self._sample_sdf(sdf)
            field_sdf_values = self._interpolate_sdf(self.sdf_grid, positions)
        elif self.optimization_phase is OptimizationPhase.FIELD:
            field_sdf_values, positions = self._sample_sdf(self.field.sdf)
            latent_sdf_values = self._interpolate_sdf(self.sdf_grid, positions)
        field_prior_distributions = torch.distributions.Normal(
            loc=latent_sdf_values,
            scale=self.noise_level * self.config.field_prior_scale_mult,
        )
        field_prior_losses = -field_prior_distributions.log_prob(field_sdf_values)
        field_prior_loss = torch.mean(field_prior_losses)
        return field_prior_loss

    @override
    def get_loss_dict(
        self,
        outputs: dict[str, Any],
        inputs: dict[str, Any],
        metrics: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        field_prior_loss = self._field_prior_loss()
        field_prior_loss = field_prior_loss * self.config.field_prior_loss_mult
        losses = dict(field_prior_loss=field_prior_loss)
        if self.optimization_phase is OptimizationPhase.LATENT:
            latent_prior_loss = self._latent_prior_loss()
            latent_prior_loss = latent_prior_loss * self.config.latent_prior_loss_mult
            losses.update(latent_prior_loss=latent_prior_loss)
        elif self.optimization_phase is OptimizationPhase.FIELD:
            losses.update(super().get_loss_dict(outputs, inputs, metrics))
        return losses
