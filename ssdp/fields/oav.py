import dataclasses
import enum
import math
from typing import Any, override

import jaxtyping as jt
import torch
import torch.nn as nn
import torchvision
from torch.distributions.utils import broadcast_all

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from ssdp.utils.jaxtyping import jaxtyped

from .sdf import SDF, SDFConfig


class SDFDistributionType(enum.StrEnum):
    NORMAL = enum.auto()
    LAPLACE = enum.auto()
    LOGISTIC = enum.auto()


class NormalDistributionType(enum.StrEnum):
    DELTA = enum.auto()
    UNIFORM = enum.auto()
    MIXTURE = enum.auto()


class Logistic(torch.distributions.TransformedDistribution):
    def __init__(self, loc: torch.Tensor | float, scale: torch.Tensor | float) -> None:
        loc, scale = broadcast_all(loc, scale)
        base_distribution = torch.distributions.Uniform(
            low=torch.zeros_like(loc),
            high=torch.ones_like(loc),
        )
        transforms = [
            torch.distributions.SigmoidTransform().inv,
            torch.distributions.AffineTransform(loc, scale),
        ]
        super().__init__(base_distribution, transforms)


@dataclasses.dataclass
class OaVConfig(SDFConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: OaV,
    )
    num_layers_anisotropy: int = 0
    sdf_distribution_type: SDFDistributionType = SDFDistributionType.NORMAL
    normal_distribution_type: NormalDistributionType = NormalDistributionType.MIXTURE


class OaV(SDF):
    config: OaVConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.anisotropy_network = torchvision.ops.MLP(
            in_channels=self.config.geo_feat_dim,
            hidden_channels=[*[self.config.hidden_dim] * self.config.num_layers_anisotropy, 1],
            activation_layer=nn.ReLU,
        )

    @jaxtyped()
    def _get_density(
        self,
        sdf_means: jt.Float[torch.Tensor, " *B 1 "],
        sdf_scales: jt.Float[torch.Tensor, " *B 1 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        if self.config.sdf_distribution_type is SDFDistributionType.NORMAL:
            sdf_scales = sdf_scales * math.pi / math.sqrt(3.0)
            sdf_distributions = torch.distributions.Normal(sdf_means, sdf_scales)
        if self.config.sdf_distribution_type is SDFDistributionType.LAPLACE:
            sdf_scales = sdf_scales * math.pi / math.sqrt(6.0)
            sdf_distributions = torch.distributions.Laplace(sdf_means, sdf_scales)
        if self.config.sdf_distribution_type is SDFDistributionType.LOGISTIC:
            sdf_distributions = Logistic(sdf_means, sdf_scales)
        pdf_values = torch.exp(sdf_distributions.log_prob(0.0))
        cdf_values = sdf_distributions.cdf(0.0)
        densities = pdf_values / torch.clamp(1.0 - cdf_values, min=1.0e-6)
        return densities

    @jaxtyped()
    def _get_projected_area(
        self,
        directions: jt.Float[torch.Tensor, " *B 3 "],
        sdf_normals: jt.Float[torch.Tensor, " *B 3 "],
        anisotropies: jt.Float[torch.Tensor, " *B 1 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        if (
            self.config.normal_distribution_type is NormalDistributionType.DELTA
            or self.config.normal_distribution_type is NormalDistributionType.MIXTURE
        ):
            projected_areas = torch.abs(torch.sum(directions * sdf_normals, dim=-1, keepdim=True))
            if self.config.normal_distribution_type is NormalDistributionType.MIXTURE:
                delta_projected_areas = projected_areas

        if (
            self.config.normal_distribution_type is NormalDistributionType.UNIFORM
            or self.config.normal_distribution_type is NormalDistributionType.MIXTURE
        ):
            projected_areas = directions.new_full((*directions.shape[:-1], 1), 0.5)
            if self.config.normal_distribution_type is NormalDistributionType.MIXTURE:
                uniform_projected_areas = projected_areas

        if self.config.normal_distribution_type is NormalDistributionType.MIXTURE:
            projected_areas = torch.lerp(
                input=uniform_projected_areas,
                end=delta_projected_areas,
                weight=anisotropies,
            )

        return projected_areas

    @jaxtyped()
    def _get_anisotropy(
        self,
        geo_features: jt.Float[torch.Tensor, " *B {self.config.geo_feat_dim} "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        anisotropies = self.anisotropy_network(geo_features)
        anisotropies = torch.sigmoid(anisotropies)
        return anisotropies

    @jaxtyped()
    def _get_attenuation_coefficient(
        self,
        directions: jt.Float[torch.Tensor, " *B 3 "],
        sdf_means: jt.Float[torch.Tensor, " *B 1 "],
        sdf_scales: jt.Float[torch.Tensor, " *B 1 "],
        sdf_gradients: jt.Float[torch.Tensor, " *B 3 "],
        anisotropies: jt.Float[torch.Tensor, " *B 1 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        sdf_normals = nn.functional.normalize(sdf_gradients, p=2, dim=-1)
        densities = self._get_density(sdf_means, sdf_scales)
        projected_areas = self._get_projected_area(directions, sdf_normals, anisotropies)
        attenuation_coefficients = densities * projected_areas
        return attenuation_coefficients

    @jaxtyped()
    @override
    def get_alpha(
        self,
        ray_samples: RaySamples,
        sdf_means: jt.Float[torch.Tensor, " *B 1 "],
        sdf_gradients: jt.Float[torch.Tensor, " *B 3 "],
        anisotropies: jt.Float[torch.Tensor, " *B 1 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        ray_directions = ray_samples.frustums.directions
        sdf_scales = 1.0 / self.deviation_network.get_variance()
        sdf_scales = sdf_scales.expand_as(sdf_means)
        densities = self._get_attenuation_coefficient(
            directions=ray_directions,
            sdf_means=sdf_means,
            sdf_scales=sdf_scales,
            sdf_gradients=sdf_gradients,
            anisotropies=anisotropies,
        )
        alphas = 1.0 - torch.exp(-densities * ray_samples.deltas)
        alphas = torch.clamp(alphas, min=0.0, max=1.0)
        return alphas

    @override
    def get_outputs(
        self,
        ray_samples: RaySamples,
        return_alphas: bool = False,
    ) -> dict[str, torch.Tensor]:
        outputs = super().get_outputs(ray_samples, return_alphas=False)
        if return_alphas:
            sdf_means = outputs[FieldHeadNames.SDF]
            sdf_gradients = outputs[FieldHeadNames.GRADIENT]
            geo_features = outputs["geo_features"]
            anisotropies = self._get_anisotropy(geo_features)
            alphas = self.get_alpha(
                ray_samples=ray_samples,
                sdf_means=sdf_means,
                sdf_gradients=sdf_gradients,
                anisotropies=anisotropies,
            )
            outputs |= {FieldHeadNames.ALPHA: alphas}
        return outputs
