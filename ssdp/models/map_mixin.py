import dataclasses
import enum
import functools
from collections.abc import Callable
from typing import Any, override

import jaxtyping as jt
import torch

from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from ssdp.utils.jaxtyping import jaxtyped

from .daps_mixin import DAPSMixin, DAPSMixinConfig


@jaxtyped()
def _straight_through(
    function: Callable[[jt.Float[torch.Tensor, " *B "]], jt.Float[torch.Tensor, " *B "]],
    inputs: jt.Float[torch.Tensor, " *B "],
) -> jt.Float[torch.Tensor, " *B "]:
    with torch.no_grad():
        outputs = function(inputs)
    return outputs + inputs - inputs.detach()


class OptimizationPhase(enum.StrEnum):
    LATENT = enum.auto()
    FIELD = enum.auto()


class LatentInitSource(enum.StrEnum):
    FIELD = enum.auto()
    PFODE = enum.auto()


@dataclasses.dataclass
class MAPMixinConfig(DAPSMixinConfig):
    num_steps_latent: int = 100
    latent_prior_loss_mult: float = 1.0
    latent_init_source: LatentInitSource = LatentInitSource.PFODE


class MAPMixin(DAPSMixin):
    config: MAPMixinConfig

    @torch.no_grad()
    def _on_latent_phase_start(self, noise_level: jt.Float[torch.Tensor, " "]) -> None:
        self.noise_level = noise_level
        self.denoised_latents = self.field.ldm.tweedie(
            noisy_latents=self.field.latents,
            noise_level=noise_level,
        )
        self.sdf_grid = self._discretize_sdf(self.field.sdf)
        if self.config.redistance_sdf:
            self.sdf_grid = self._redistance_sdf(self.sdf_grid)
        self.optimization_phase = OptimizationPhase.LATENT

    @torch.no_grad()
    def _on_field_phase_start(self, noise_level: jt.Float[torch.Tensor, " "]) -> None:
        self.noise_level = noise_level
        sdf = self.field.decode(self.field.latents)
        self.sdf_grid = self._discretize_sdf(sdf)
        if self.config.redistance_sdf:
            self.sdf_grid = self._redistance_sdf(self.sdf_grid)
        self.optimization_phase = OptimizationPhase.FIELD

    @override
    def get_training_callbacks(
        self,
        training_callback_attributes: TrainingCallbackAttributes,
    ) -> list[TrainingCallback]:
        callbacks = super(DAPSMixin, self).get_training_callbacks(training_callback_attributes)

        num_steps_train = training_callback_attributes.trainer.config.max_num_iterations
        num_steps_stage, remainder = divmod(num_steps_train, self.config.num_steps_anneal)
        assert not remainder
        assert self.config.num_steps_latent < num_steps_stage

        @torch.no_grad()
        def on_train_start(step: int) -> None:
            if self.field.ldm.config.condition_image_file is None:
                self.field.ldm.uncondition()
            else:
                self.field.ldm.condition()
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
            if self.config.redistance_sdf:
                sdf_grid = self._discretize_sdf(sdf)
                sdf_grid = _straight_through(self._redistance_sdf)(sdf_grid)
                sdf = functools.partial(self._interpolate_sdf, sdf_grid)
                sdf = self.field.transform(sdf)
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
        if self.optimization_phase is OptimizationPhase.LATENT:
            field_prior_loss = self._field_prior_loss()
            field_prior_loss = field_prior_loss * self.config.field_prior_loss_mult
            latent_prior_loss = self._latent_prior_loss()
            latent_prior_loss = latent_prior_loss * self.config.latent_prior_loss_mult
            losses = dict(
                field_prior_loss=field_prior_loss,
                latent_prior_loss=latent_prior_loss,
            )
        elif self.optimization_phase is OptimizationPhase.FIELD:
            losses = super().get_loss_dict(outputs, inputs, metrics)
        return losses
