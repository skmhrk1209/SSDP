import dataclasses
import enum
from collections.abc import Callable
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
from ssdp.utils.jaxtyping import jaxtyped

type SDFType = Callable[
    [jt.Float[torch.Tensor, " *B 3 "]],
    jt.Float[torch.Tensor, " *B 1 "],
]


class LDMDType(enum.StrEnum):
    FLOAT32 = enum.auto()
    FLOAT16 = enum.auto()
    BFLOAT16 = enum.auto()


@dataclasses.dataclass
class DAPSMixinConfig(SurfaceModelConfig):
    num_steps_anneal: int = 100
    num_steps_pfode: int = 10
    max_noise_level_anneal: float = 10.0
    min_noise_level_anneal: float = 0.1
    min_noise_level_pfode: float = 0.001
    field_prior_scale_mult: float = 1.0
    field_prior_loss_mult: float = 1.0
    sdf_num_samples: int = 100000
    sdf_resolution: int = 64
    sdf_chunk_size: int = 1 << 18
    redistance_sdf: bool = True
    denoiser_dtype: LDMDType = LDMDType.FLOAT16
    encoder_dtype: LDMDType = LDMDType.FLOAT32
    decoder_dtype: LDMDType = LDMDType.FLOAT32


class DAPSMixin(SurfaceModel):
    config: DAPSMixinConfig
    field: LDMMixin
    voxel_grid: torch.Tensor
    noise_levels: torch.Tensor

    @override
    def populate_modules(self) -> None:
        super().populate_modules()
        self.register_buffer(
            name="voxel_grid",
            tensor=self._get_voxel_grid(),
            persistent=False,
        )
        self.register_buffer(
            name="noise_levels",
            tensor=edm_noise_levels(
                max_noise_level=torch.as_tensor(self.config.max_noise_level_anneal),
                min_noise_level=torch.as_tensor(self.config.min_noise_level_anneal),
                num_ode_steps=self.config.num_steps_anneal,
            ),
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
    def _get_voxel_sizes(self) -> jt.Float[torch.Tensor, " 3 "]:
        voxel_sizes = -torch.sub(*self.field.ldm.aabb) / self.config.sdf_resolution
        return voxel_sizes

    @jaxtyped()
    def _get_voxel_grid(self) -> jt.Float[torch.Tensor, " D H W 3 "]:
        voxel_sizes = self._get_voxel_sizes()
        voxel_coords = torch.linspace(0.0, 1.0, self.config.sdf_resolution + 1)
        voxel_coords = torch.lerp(*self.field.ldm.aabb, voxel_coords.unsqueeze(-1))
        voxel_coords = voxel_coords[:-1, ...] + voxel_sizes / 2.0
        voxel_grid = torch.meshgrid(*reversed(voxel_coords.T), indexing="ij")
        voxel_grid = torch.stack(list(reversed(voxel_grid)), dim=-1)
        return voxel_grid

    def _chunked_sdf(self, sdf: SDFType) -> SDFType:
        @jaxtyped()
        def chunked_sdf(
            positions: jt.Float[torch.Tensor, " *B 3 "],
        ) -> jt.Float[torch.Tensor, " *B 1 "]:
            chunks = positions.flatten(0, -2).split(self.config.sdf_chunk_size, dim=0)
            sdf_values = torch.cat(list(map(sdf, chunks)), dim=0)
            sdf_values = sdf_values.reshape(*positions.shape[:-1], 1)
            return sdf_values

        return chunked_sdf

    @jaxtyped()
    def _discretize_sdf(self, sdf: SDFType) -> jt.Float[torch.Tensor, " D H W 1 "]:
        sdf = self.field.transform.inverse()(sdf)
        sdf_grid = self._chunked_sdf(sdf)(self.voxel_grid)
        return sdf_grid

    @jaxtyped()
    def _get_random_positions(
        self,
        num_positions: int,
    ) -> jt.Float[torch.Tensor, " {num_positions} 3 "]:
        positions = torch.rand(num_positions, 3, device=self.device)
        positions = torch.lerp(*self.field.ldm.aabb, positions)
        return positions

    @jaxtyped()
    def _sample_sdf(
        self,
        sdf: SDFType,
    ) -> tuple[jt.Float[torch.Tensor, " *B 1 "], jt.Float[torch.Tensor, " *B 3 "]]:
        positions = self._get_random_positions(self.config.sdf_num_samples)
        sdf = self.field.transform.inverse()(sdf)
        sdf_values = self._chunked_sdf(sdf)(positions)
        return sdf_values, positions

    @jaxtyped()
    def _normalize_position(
        self,
        positions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B 3 "]:
        voxel_sizes = self._get_voxel_sizes()
        aabb_min, aabb_max = self.field.ldm.aabb
        grid_min = aabb_min + voxel_sizes / 2.0
        grid_max = aabb_max - voxel_sizes / 2.0
        positions = (positions - grid_min) / (grid_max - grid_min)
        return positions

    @jaxtyped()
    def _denormalize_position(
        self,
        positions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B 3 "]:
        voxel_sizes = self._get_voxel_sizes()
        aabb_min, aabb_max = self.field.ldm.aabb
        grid_min = aabb_min + voxel_sizes / 2.0
        grid_max = aabb_max - voxel_sizes / 2.0
        positions = positions * (grid_max - grid_min) + grid_min
        return positions

    @jaxtyped()
    def _interpolate_sdf(
        self,
        sdf_grid: jt.Float[torch.Tensor, " D H W 1 "],
        positions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        positions = self._normalize_position(positions)
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

    @jaxtyped()
    def _has_zero_level(
        self,
        sdf_values: jt.Float[torch.Tensor, " *B "],
    ) -> jt.Bool[torch.Tensor, " "]:
        sdf_signs = torch.sign(sdf_values)
        return torch.any(sdf_signs >= 0.0) and torch.any(sdf_signs <= 0.0)

    @jaxtyped()
    @torch.no_grad()
    def _redistance_sdf(
        self,
        sdf_grid: jt.Float[torch.Tensor, " D H W 1 "],
    ) -> jt.Float[torch.Tensor, " D H W 1 "]:
        if not self._has_zero_level(sdf_grid):
            return sdf_grid
        voxel_sizes = self._get_voxel_sizes()
        voxel_sizes = voxel_sizes.flip(-1)
        sdf_grid = sdf_grid.squeeze(-1)
        sdf_grid = skfmm.distance(
            phi=sdf_grid.cpu().numpy(),
            dx=voxel_sizes.cpu().numpy(),
            order=2,
        )
        sdf_grid = voxel_sizes.new_tensor(sdf_grid)
        sdf_grid = sdf_grid.unsqueeze(-1)
        return sdf_grid

    @torch.no_grad()
    def _on_mcmc_start(self, noise_level: jt.Float[torch.Tensor, " "]) -> None:
        self.noise_level = noise_level
        noisy_latents = self.field.ldm.forward_sde(
            latents=self.field.latents,
            noise_level=noise_level,
        )
        with torch.autocast(self.device.type, getattr(torch, self.config.denoiser_dtype)):
            denoised_latents = self.field.ldm.reverse_pfode(
                noisy_latents=noisy_latents,
                max_noise_level=noise_level,
                min_noise_level=noise_level.new_tensor(self.config.min_noise_level_pfode),
                num_ode_steps=self.config.num_steps_pfode,
            )
            denoised_latents = denoised_latents.float()
        self.denoised_latents = denoised_latents
        self.field.latents.copy_(denoised_latents)
        if self.field.config.mcmc_space is MCMCSpace.FIELD:
            with torch.autocast(self.device.type, getattr(torch, self.config.decoder_dtype)):
                sdf = self.field.decode(denoised_latents)
                sdf_grid = self._discretize_sdf(sdf)
                sdf_grid = sdf_grid.float()
            if self.config.redistance_sdf:
                sdf_grid = self._redistance_sdf(sdf_grid)
            self.sdf_grid = sdf_grid

    @torch.no_grad()
    def _on_mcmc_end(self) -> None:
        if self.field.config.mcmc_space is MCMCSpace.FIELD:
            with torch.autocast(self.device.type, getattr(torch, self.config.encoder_dtype)):
                latents = self.field.encode()
                latents = latents.float()
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
            noise_level = self.noise_levels[annealing_step, ...]
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
    def _field_prior_loss(self) -> jt.Float[torch.Tensor, " "]:
        if self.field.config.mcmc_space is MCMCSpace.LATENT:
            field_prior_distributions = torch.distributions.Normal(
                loc=self.denoised_latents,
                scale=self.noise_level * self.config.field_prior_scale_mult,
            )
            field_prior_losses = -field_prior_distributions.log_prob(self.field.latents)
        elif self.field.config.mcmc_space is MCMCSpace.FIELD:
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
        losses = super().get_loss_dict(outputs, inputs, metrics)
        field_prior_loss = self._field_prior_loss()
        field_prior_loss = field_prior_loss * self.config.field_prior_loss_mult
        losses.update(field_prior_loss=field_prior_loss)
        return losses
