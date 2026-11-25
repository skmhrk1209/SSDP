import dataclasses
import enum
import math
from typing import Any, override

import jaxtyping as jt
import torch
import torchvision
from torch import nn
from torch.distributions.utils import broadcast_all

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from ssdp.utils.jaxtyping import jaxtyped

from .neus_field import NeuSField, NeuSFieldConfig, RaySampleLocation


class _Logistic(torch.distributions.TransformedDistribution):
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


class SDFDistributionType(enum.StrEnum):
    NORMAL = enum.auto()
    LAPLACE = enum.auto()
    LOGISTIC = enum.auto()


class NormalDistributionType(enum.StrEnum):
    DELTA = enum.auto()
    UNIFORM = enum.auto()
    MIXTURE = enum.auto()


@dataclasses.dataclass
class OaVFieldConfig(NeuSFieldConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: OaVField,
    )
    num_anisotropy_layers: int = 0
    sdf_distribution_type: SDFDistributionType = SDFDistributionType.NORMAL
    normal_distribution_type: NormalDistributionType = NormalDistributionType.MIXTURE


class OaVField(NeuSField):
    config: OaVFieldConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.anisotropy_network = torchvision.ops.MLP(
            in_channels=self.config.geo_feat_dim,
            hidden_channels=[*[self.config.hidden_dim] * self.config.num_anisotropy_layers, 1],
            activation_layer=nn.ReLU,
        )

    @jaxtyped()
    def _get_density(
        self,
        sdf_means: jt.Float[torch.Tensor, " *B 1 "],
        sdf_scales: jt.Float[torch.Tensor, " *B 1 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        match self.config.sdf_distribution_type:
            case SDFDistributionType.NORMAL:
                sdf_scales = sdf_scales * math.pi / math.sqrt(3.0)
                sdf_distributions = torch.distributions.Normal(sdf_means, sdf_scales)
            case SDFDistributionType.LAPLACE:
                sdf_scales = sdf_scales * math.pi / math.sqrt(6.0)
                sdf_distributions = torch.distributions.Laplace(sdf_means, sdf_scales)
            case SDFDistributionType.LOGISTIC:
                sdf_distributions = _Logistic(sdf_means, sdf_scales)
        pdf_values = torch.exp(sdf_distributions.log_prob(sdf_means.new_zeros(())))
        cdf_values = sdf_distributions.cdf(sdf_means.new_zeros(()))
        # NOTE: The official implementation of OaV omits the gradient norm term,
        # assuming the SDF sarisfy the Eikonal equation.
        # https://github.com/cmu-ci-lab/volumetric_opaque_solids/blob/main/models/attenuation_coefficient.py#L106
        densities = pdf_values / torch.clamp(1.0 - cdf_values, min=1.0e-6)
        return densities

    @jaxtyped()
    def _get_projected_area_delta(
        self,
        directions: jt.Float[torch.Tensor, " *B 3 "],
        sdf_normals: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        projected_areas = torch.abs(torch.sum(directions * sdf_normals, dim=-1, keepdim=True))
        return projected_areas

    @jaxtyped()
    def _get_projected_area_uniform(
        self,
        directions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        projected_areas = directions.new_full((*directions.shape[:-1], 1), 0.5)
        return projected_areas

    @jaxtyped()
    def _get_projected_area_mixture(
        self,
        directions: jt.Float[torch.Tensor, " *B 3 "],
        sdf_normals: jt.Float[torch.Tensor, " *B 3 "],
        anisotropies: jt.Float[torch.Tensor, " *B 1 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        delta_projected_areas = self._get_projected_area_delta(
            directions=directions,
            sdf_normals=sdf_normals,
        )
        uniform_projected_areas = self._get_projected_area_uniform(
            directions=directions,
        )
        projected_areas = torch.lerp(
            input=uniform_projected_areas,
            end=delta_projected_areas,
            weight=anisotropies,
        )
        return projected_areas

    @jaxtyped()
    def _get_projected_area(
        self,
        directions: jt.Float[torch.Tensor, " *B 3 "],
        sdf_normals: jt.Float[torch.Tensor, " *B 3 "],
        anisotropies: jt.Float[torch.Tensor, " *B 1 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        match self.config.normal_distribution_type:
            case NormalDistributionType.DELTA:
                projected_areas = self._get_projected_area_delta(
                    directions=directions,
                    sdf_normals=sdf_normals,
                )
            case NormalDistributionType.UNIFORM:
                projected_areas = self._get_projected_area_uniform(
                    directions=directions,
                )
            case NormalDistributionType.MIXTURE:
                projected_areas = self._get_projected_area_mixture(
                    directions=directions,
                    sdf_normals=sdf_normals,
                    anisotropies=anisotropies,
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
        sdf_normals: jt.Float[torch.Tensor, " *B 3 "],
        anisotropies: jt.Float[torch.Tensor, " *B 1 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        densities = self._get_density(sdf_means, sdf_scales)
        projected_areas = self._get_projected_area(directions, sdf_normals, anisotropies)
        attenuation_coefficients = densities * projected_areas
        return attenuation_coefficients

    @jaxtyped()
    @override
    def get_opacity(
        self,
        intervals: jt.Float[torch.Tensor, " *B 1 "],
        directions: jt.Float[torch.Tensor, " *B 3 "],
        sdf_values: jt.Float[torch.Tensor, " *B 1 "],
        sdf_normals: jt.Float[torch.Tensor, " *B 3 "],
        anisotropies: jt.Float[torch.Tensor, " *B 1 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        sdf_scales = self.sdf_scale.expand_as(sdf_values)
        densities = self._get_attenuation_coefficient(
            directions=directions,
            sdf_means=sdf_values,
            sdf_scales=sdf_scales,
            sdf_normals=sdf_normals,
            anisotropies=anisotropies,
        )
        opacities = 1.0 - torch.exp(-densities * intervals)
        opacities = torch.clamp(opacities, min=0.0, max=1.0)
        return opacities

    @override
    def get_outputs(
        self,
        ray_samples: RaySamples,
        return_alphas: bool = True,
        return_colors: bool = True,
        return_gradients: bool | None = None,
        ray_sample_location: RaySampleLocation = RaySampleLocation.START,
    ) -> dict[FieldHeadNames | str, torch.Tensor]:
        outputs = super().get_outputs(
            ray_samples=ray_samples,
            return_alphas=False,
            return_colors=return_colors,
            return_gradients=return_gradients or return_alphas,
            ray_sample_location=ray_sample_location,
        )

        if return_alphas:
            assert ray_samples.deltas is not None
            intervals = torch.clamp(ray_samples.deltas, min=0.0)
            directions = ray_samples.frustums.directions
            sdf_values = outputs[FieldHeadNames.SDF]
            sdf_normals = outputs[FieldHeadNames.NORMALS]
            geo_features = outputs["geo_features"]
            anisotropies = self._get_anisotropy(geo_features)
            opacities = self.get_opacity(
                intervals=intervals,
                directions=directions,
                sdf_values=sdf_values,
                sdf_normals=sdf_normals,
                anisotropies=anisotropies,
            )
            outputs |= {FieldHeadNames.ALPHA: opacities}

        return outputs
