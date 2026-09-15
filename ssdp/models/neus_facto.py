import dataclasses
from collections.abc import Callable
from typing import Any, override

import torch

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.model_components.ray_samplers import UniformSampler
from nerfstudio.models.neus_facto import NeuSFactoModel, NeuSFactoModelConfig
from ssdp.fields import SDFConfig

from .surface_model_mixin import SurfaceModelMixin, SurfaceModelMixinConfig


class _UniformSampler(UniformSampler):
    @override
    def generate_ray_samples(
        self,
        ray_bundle: RayBundle,
        density_fns: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> tuple[RaySamples, list[torch.Tensor], list[RaySamples]]:
        return super().generate_ray_samples(ray_bundle), [], []

    def set_anneal(self, anneal: float) -> None:
        pass

    def step_cb(self, step: int) -> None:
        pass


@dataclasses.dataclass
class _NeuSFactoModelConfig(NeuSFactoModelConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: _NeuSFactoModel,
    )
    uniform_sampling: bool = False


class _NeuSFactoModel(NeuSFactoModel):
    config: _NeuSFactoModelConfig

    @override
    def populate_modules(self) -> None:
        super().populate_modules()
        if self.config.uniform_sampling:
            self.proposal_sampler = _UniformSampler(
                num_samples=self.config.num_neus_samples_per_ray,
                single_jitter=self.config.use_single_jitter,
            )

    @override
    def get_loss_dict(
        self,
        outputs: dict[str, Any],
        inputs: dict[str, Any],
        metrics: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.config.uniform_sampling:
            return super(NeuSFactoModel, self).get_loss_dict(outputs, inputs, metrics)
        return super().get_loss_dict(outputs, inputs, metrics)

    @torch.no_grad()
    @override
    def get_image_metrics_and_images(
        self,
        outputs: dict[str, Any],
        inputs: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.config.uniform_sampling:
            return super(NeuSFactoModel, self).get_image_metrics_and_images(outputs, inputs)
        return super().get_image_metrics_and_images(outputs, inputs)


@dataclasses.dataclass
class NeuSFactoConfig(SurfaceModelMixinConfig, _NeuSFactoModelConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: NeuSFacto,
    )
    sdf_field: SDFFieldConfig = dataclasses.field(
        default_factory=SDFConfig,
    )


class NeuSFacto(SurfaceModelMixin, _NeuSFactoModel):
    pass
