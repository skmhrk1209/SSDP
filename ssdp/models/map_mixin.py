import dataclasses
import enum
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
from ssdp.utils.aliases import SDFType
from ssdp.utils.jaxtyping import jaxtyped


class OptimizationPhase(enum.StrEnum):
    LATENT = enum.auto()
    FIELD = enum.auto()


class LatentInitSource(enum.StrEnum):
    FIELD = enum.auto()
    PFODE = enum.auto()


@dataclasses.dataclass
class MAPMixinConfig(SurfaceModelConfig):
    num_steps_anneal: int = 100
    num_steps_latent: int = 10
    num_steps_pfode: int = 10
    max_noise_level_anneal: float = 10.0
    min_noise_level_anneal: float = 0.1
    min_noise_level_pfode: float = 0.001
    effective_sample_size_field: float = 1.0
    effective_sample_size_latent: float = 1.0
    coupling_scale_mult: float = 1.0
    sdf_resolution: int = 64
    sdf_chunk_size: int = 1 << 18
    latent_init_source: LatentInitSource = LatentInitSource.FIELD


class MAPMixin(SurfaceModel):
    config: MAPMixinConfig
    field: LDMMixin
    noise_levels: torch.Tensor
    coupling_distributions: torch.distributions.Distribution

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
    def _discretize_sdf(self, sdf: SDFType) -> jt.Float[torch.Tensor, " D H W "]:
        positions = (
            torch.linspace(*range, self.config.sdf_resolution + 1, device=self.device)[..., :-1]
            for range in self.field.aabb.T
        )
        positions = torch.stack(torch.meshgrid(*positions, indexing="ij"), dim=-1)
        chunks = positions.flatten(0, -2).split(self.config.sdf_chunk_size)
        if torch.is_grad_enabled():
            # NOTE: the decoder's graph is recomputed chunk by chunk in the backward pass (memory)
            sdf_values = torch.cat(
                [
                    torch.utils.checkpoint.checkpoint(sdf, chunk, use_reentrant=False)
                    for chunk in chunks
                ],
                dim=0,
            )
        else:
            sdf_values = torch.cat(list(map(sdf, chunks)), dim=0)
        sdf_values = sdf_values.reshape(positions.shape[:-1])
        return sdf_values

    @jaxtyped()
    def _get_coupling_distribution(
        self,
        latents: jt.Float[torch.Tensor, " L C "],
        noise_level: jt.Float[torch.Tensor, " "],
    ) -> torch.distributions.Distribution:
        sdf = self.field.decode(latents)
        sdf_grid = self._discretize_sdf(sdf)
        coupling_distributions = torch.distributions.Normal(
            loc=sdf_grid,
            scale=noise_level * self.config.coupling_scale_mult,
        )
        return coupling_distributions

    @torch.no_grad()
    def _on_latent_phase_start(self, noise_level: jt.Float[torch.Tensor, " "]) -> None:
        self.noise_level = noise_level
        self.sdf_grid = self._discretize_sdf(self.field.sdf)
        denoised_latents = self.field.ldm.tweedie(
            noisy_latents=self.field.latents,
            noise_level=noise_level,
        )
        self.latent_prior_distributions = torch.distributions.Normal(
            loc=denoised_latents,
            scale=noise_level,
        )
        self.optimization_phase = OptimizationPhase.LATENT

    @torch.no_grad()
    def _on_field_phase_start(self, noise_level: jt.Float[torch.Tensor, " "]) -> None:
        self.coupling_distributions = self._get_coupling_distribution(
            latents=self.field.latents,
            noise_level=noise_level,
        )
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
        latent_prior_losses = -self.latent_prior_distributions.log_prob(self.field.latents)
        latent_prior_loss = torch.mean(latent_prior_losses)
        latent_prior_loss = latent_prior_loss * self.config.effective_sample_size_latent
        return latent_prior_loss

    @jaxtyped()
    def _latent_coupling_loss(self) -> jt.Float[torch.Tensor, " "]:
        coupling_distributions = self._get_coupling_distribution(
            latents=self.field.latents,
            noise_level=self.noise_level,
        )
        coupling_losses = -coupling_distributions.log_prob(self.sdf_grid)
        coupling_loss = torch.mean(coupling_losses)
        coupling_loss = coupling_loss * self.config.effective_sample_size_field
        return coupling_loss

    @jaxtyped()
    def _field_coupling_loss(self) -> jt.Float[torch.Tensor, " "]:
        sdf_grid = self._discretize_sdf(self.field.sdf)
        field_coupling_losses = -self.coupling_distributions.log_prob(sdf_grid)
        field_coupling_loss = torch.mean(field_coupling_losses)
        field_coupling_loss = field_coupling_loss * self.config.effective_sample_size_field
        return field_coupling_loss

    @override
    def get_loss_dict(
        self,
        outputs: dict[str, Any],
        inputs: dict[str, Any],
        metrics: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.optimization_phase is OptimizationPhase.LATENT:
            latent_coupling_loss = self._latent_coupling_loss()
            latent_prior_loss = self._latent_prior_loss()
            losses = dict(
                latent_coupling_loss=latent_coupling_loss,
                latent_prior_loss=latent_prior_loss,
            )
        elif self.optimization_phase is OptimizationPhase.FIELD:
            losses = super().get_loss_dict(outputs, inputs, metrics)
            field_coupling_loss = self._field_coupling_loss()
            losses.update(field_coupling_loss=field_coupling_loss)
        return losses
