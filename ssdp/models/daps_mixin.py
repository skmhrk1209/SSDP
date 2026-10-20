import dataclasses
from typing import Any, override

import jaxtyping as jt
import skfmm
import torch
import torch.nn as nn

from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig
from ssdp.fields.ldm_mixin import LDMMixin, MCMCSpace
from ssdp.priors.ldm import edm_noise_levels
from ssdp.utils.aliases import SDFType
from ssdp.utils.jaxtyping import jaxtyped


@dataclasses.dataclass
class DAPSMixinConfig(SurfaceModelConfig):
    seed: int = 0
    num_steps_anneal: int = 50
    num_steps_pfode: int = 2
    max_noise_level_anneal: float = 10.0
    min_noise_level_anneal: float = 0.1
    min_noise_level_pfode: float = 0.001
    denoising_scale_mult: float = 1.0
    sdf_resolution: int = 128
    sdf_chunk_size: int = 1 << 18


class DAPSMixin(SurfaceModel):
    config: DAPSMixinConfig
    field: LDMMixin
    noisy_latents: torch.Tensor
    denoising_posterior: torch.distributions.Distribution
    max_noise_level_anneal: torch.Tensor
    min_noise_level_anneal: torch.Tensor
    min_noise_level_pfode: torch.Tensor

    @override
    def populate_modules(self) -> None:
        super().populate_modules()
        self.register_buffer(
            name="max_noise_level_anneal",
            tensor=torch.as_tensor(self.config.max_noise_level_anneal),
            persistent=False,
        )
        self.register_buffer(
            name="min_noise_level_anneal",
            tensor=torch.as_tensor(self.config.min_noise_level_anneal),
            persistent=False,
        )
        self.register_buffer(
            name="min_noise_level_pfode",
            tensor=torch.as_tensor(self.config.min_noise_level_pfode),
            persistent=False,
        )

    @override
    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        param_groups = super().get_param_groups()
        param_groups.update(
            mcmc=list(self.field.mcmc_parameters()),
            fields=list(self.field.nuisance_parameters()),
        )
        return param_groups

    @jaxtyped()
    @torch.no_grad()
    def _redistance_sdf(
        self,
        sdf_grid: jt.Float[torch.Tensor, " D H W "],
    ) -> jt.Float[torch.Tensor, " D H W "]:
        sdf_signs = torch.sign(sdf_grid)
        if torch.any(sdf_signs >= 0.0) and torch.any(sdf_signs <= 0.0):
            voxel_size = -torch.sub(*self.field.aabb) / self.config.sdf_resolution
            sdf_grid = sdf_grid.new_tensor(
                skfmm.distance(
                    phi=sdf_grid.double().cpu().numpy(),
                    dx=voxel_size.double().cpu().numpy(),
                    order=2,
                )
            )
        return sdf_grid

    @jaxtyped()
    def _discretize_sdf(self, sdf: SDFType) -> jt.Float[torch.Tensor, " D H W "]:
        positions = (
            torch.linspace(
                *range,
                self.config.sdf_resolution + 1,
                device=self.device,
            )[..., :-1]
            for range in self.field.aabb.T
        )
        positions = torch.stack(torch.meshgrid(*positions, indexing="ij"), dim=-1)
        chunks = positions.flatten(0, -2).split(self.config.sdf_chunk_size)
        sdf_values = torch.cat(list(map(sdf, chunks)), dim=0)
        sdf_values = sdf_values.reshape(positions.shape[:-1])
        return sdf_values

    @jaxtyped()
    @torch.no_grad()
    def _get_denoising_posterior(
        self,
        denoised_latents: jt.Float[torch.Tensor, " L C "],
        denoising_scale: jt.Float[torch.Tensor, " "],
    ) -> torch.distributions.Distribution:
        if self.field.config.mcmc_space is MCMCSpace.LATENT:
            denoising_posterior = torch.distributions.Normal(
                loc=denoised_latents,
                scale=denoising_scale,
            )
        elif self.field.config.mcmc_space is MCMCSpace.FIELD:
            sdf = self.field.decode(denoised_latents)
            sdf_grid = self._discretize_sdf(sdf)
            sdf_grid = self._redistance_sdf(sdf_grid)
            denoising_posterior = torch.distributions.Normal(
                loc=sdf_grid,
                scale=denoising_scale,
            )
        return denoising_posterior

    @torch.no_grad()
    def _on_mcmc_start(self, noise_level: jt.Float[torch.Tensor, " "]) -> None:
        noisy_latents = self.field.ldm.forward_sde(
            latents=self.field.latents,
            noise_level=noise_level,
        )
        denoised_latents = self.field.ldm.reverse_pfode(
            noisy_latents=noisy_latents,
            max_noise_level=noise_level,
            min_noise_level=self.min_noise_level_pfode,
            num_ode_steps=self.config.num_steps_pfode,
        )
        self.field.latents.copy_(denoised_latents)
        self.denoising_posterior = self._get_denoising_posterior(
            denoised_latents=denoised_latents,
            denoising_scale=noise_level * self.config.denoising_scale_mult,
        )

    @torch.no_grad()
    def _on_mcmc_end(self) -> None:
        if self.field.config.mcmc_space is MCMCSpace.FIELD:
            latents = self.field.encode()
            self.field.latents.copy_(latents)

    @override
    def get_training_callbacks(
        self,
        training_callback_attributes: TrainingCallbackAttributes,
    ) -> list[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        num_steps_train = training_callback_attributes.trainer.config.max_num_iterations
        num_steps_mcmc, remainder = divmod(num_steps_train, self.config.num_steps_anneal)
        assert not remainder

        def on_train_start(step: int) -> None:
            if self.field.ldm.config.condition_image_file is None:
                self.field.ldm.uncondition()
            else:
                self.field.ldm.condition()

        def on_mcmc_start(step: int) -> None:
            if step:
                self._on_mcmc_end()
            annealing_step = step // num_steps_mcmc
            noise_levels = edm_noise_levels(
                max_noise_level=self.max_noise_level_anneal,
                min_noise_level=self.min_noise_level_anneal,
                num_ode_steps=self.config.num_steps_anneal,
                device=self.device,
            )
            noise_level = noise_levels[annealing_step]
            self._on_mcmc_start(noise_level)

        callbacks.extend(
            [
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=num_steps_train,
                    func=on_train_start,
                ),
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=num_steps_mcmc,
                    func=on_mcmc_start,
                ),
            ]
        )
        return callbacks

    @jaxtyped()
    def _denoising_posterior_loss(self) -> jt.Float[torch.Tensor, " "]:
        if self.field.config.mcmc_space is MCMCSpace.LATENT:
            denoising_posterior_losses = -self.denoising_posterior.log_prob(self.field.latents)
        elif self.field.config.mcmc_space is MCMCSpace.FIELD:
            sdf_grid = self._discretize_sdf(self.field.sdf)
            denoising_posterior_losses = -self.denoising_posterior.log_prob(sdf_grid)
        denoising_posterior_loss = torch.mean(denoising_posterior_losses)
        return denoising_posterior_loss

    @override
    def get_loss_dict(
        self,
        outputs: dict[str, Any],
        inputs: dict[str, Any],
        metrics: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = super().get_loss_dict(outputs, inputs, metrics)
        if self.training:
            denoising_posterior_loss = self._denoising_posterior_loss()
            losses.update(denoising_posterior_loss=denoising_posterior_loss)
        return losses
