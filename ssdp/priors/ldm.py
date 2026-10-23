import abc
import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jaxtyping as jt
import torch
import torch.nn as nn
import torchdiffeq
import trimesh

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.scene_box import SceneBox
from ssdp.utils.jaxtyping import jaxtyped

type SDFType = Callable[
    [jt.Float[torch.Tensor, " *B 3 "]],
    jt.Float[torch.Tensor, " *B 1 "],
]


@jaxtyped()
def edm_noise_levels(
    max_noise_level: jt.Float[torch.Tensor, " "],
    min_noise_level: jt.Float[torch.Tensor, " "],
    num_ode_steps: int,
    exponent: float = 7.0,
    **kwargs: Any,
) -> jt.Float[torch.Tensor, " {num_ode_steps} "]:
    return (
        torch.linspace(
            start=max_noise_level ** (1.0 / exponent),
            end=min_noise_level ** (1.0 / exponent),
            steps=num_ode_steps,
            **kwargs,
        )
        ** exponent
    )


@dataclasses.dataclass
class LDMConfig(InstantiateConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: LDM,
    )
    condition_image_file: Path | None = None


class LDM(nn.Module, abc.ABC):
    config: LDMConfig
    aabb: torch.Tensor

    def __init__(self, config: LDMConfig) -> None:
        super().__init__()
        self.config = config

    @property
    def scene_box(self) -> SceneBox:
        return SceneBox(aabb=self.aabb)

    @property
    @abc.abstractmethod
    def latent_shape(self) -> tuple[int, ...]: ...

    @abc.abstractmethod
    def condition(self, image_file: Path | None) -> None: ...

    @abc.abstractmethod
    def uncondition(self) -> None: ...

    @abc.abstractmethod
    def encode(self, mesh: trimesh.Trimesh) -> jt.Float[torch.Tensor, " L C "]: ...

    @abc.abstractmethod
    def decode(
        self,
        latents: jt.Float[torch.Tensor, " L C "],
    ) -> SDFType: ...

    @abc.abstractmethod
    def tweedie(
        self,
        noisy_latents: jt.Float[torch.Tensor, " L C "],
        noise_level: jt.Float[torch.Tensor, " "],
    ) -> jt.Float[torch.Tensor, " L C "]: ...

    @jaxtyped()
    def score(
        self,
        noisy_latents: jt.Float[torch.Tensor, " L C "],
        noise_level: jt.Float[torch.Tensor, " "],
    ) -> jt.Float[torch.Tensor, " L C "]:
        denoised_latents = self.tweedie(noisy_latents, noise_level)
        scores = (denoised_latents - noisy_latents) / (noise_level**2.0)
        return scores

    @jaxtyped()
    def forward_sde(
        self,
        latents: jt.Float[torch.Tensor, " L C "],
        noise_level: jt.Float[torch.Tensor, " "],
        seed: int | None = None,
    ) -> jt.Float[torch.Tensor, " L C "]:
        # NOTE: without a seed, the global RNG (seeded by the trainer) is used, so that every call
        # draws fresh noise; a fresh `torch.Generator()` would start from the same default seed.
        generator = None
        if seed is not None:
            generator = torch.Generator(latents.device)
            generator.manual_seed(seed)
        noises = torch.randn(
            latents.shape,
            dtype=latents.dtype,
            device=latents.device,
            generator=generator,
        )
        noisy_latents = latents + noise_level * noises
        return noisy_latents

    @jaxtyped()
    @torch.no_grad()
    def reverse_pfode(
        self,
        noisy_latents: jt.Float[torch.Tensor, " L C "],
        max_noise_level: jt.Float[torch.Tensor, " "],
        min_noise_level: jt.Float[torch.Tensor, " "],
        num_ode_steps: int,
    ) -> jt.Float[torch.Tensor, " L C "]:
        @jaxtyped()
        def derivative(
            noise_levels: jt.Float[torch.Tensor, " "],
            noisy_latents: jt.Float[torch.Tensor, " L C "],
        ) -> jt.Float[torch.Tensor, " L C "]:
            return -noise_levels * self.score(noisy_latents, noise_levels)

        noise_levels = edm_noise_levels(
            max_noise_level=max_noise_level,
            min_noise_level=min_noise_level,
            num_ode_steps=num_ode_steps,
        )

        *_, denoised_latents = torchdiffeq.odeint(
            func=derivative,
            y0=noisy_latents,
            t=noise_levels,
            method="euler",
        )

        return denoised_latents
