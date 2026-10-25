import dataclasses
import enum
import json
import operator
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Self, override

import jaxtyping as jt
import torch
import torch.nn as nn

from nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes
from ssdp.priors.ldm import LDM, LDMConfig
from ssdp.priors.triposg import TripoSGPriorConfig
from ssdp.utils.geometry import SimilarityTransform
from ssdp.utils.jaxtyping import jaxtyped

from .sdf import SDF, SDFConfig

type SDFType = Callable[
    [jt.Float[torch.Tensor, " *B 3 "]],
    jt.Float[torch.Tensor, " *B 1 "],
]


class MCMCSpace(enum.StrEnum):
    FIELD = enum.auto()
    LATENT = enum.auto()


class EncoderOutMode(enum.StrEnum):
    MEAN = enum.auto()
    SAMPLE = enum.auto()


@dataclasses.dataclass
class LDMMixinConfig(SDFConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: LDMMixin,
    )
    ldm: LDMConfig = dataclasses.field(
        default_factory=TripoSGPriorConfig,
    )
    mcmc_space: MCMCSpace = MCMCSpace.FIELD
    latent_file: Path | None = None
    transform_file: Path | None = None
    sdf_resolution: int = 512
    encoder_out_mode: EncoderOutMode = EncoderOutMode.MEAN


class LDMMixin(SDF):
    config: LDMMixinConfig
    latents: torch.Tensor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.ldm: LDM = self.config.ldm.setup()
        self.ldm.requires_grad_(False)
        self.ldm.eval()

        if self.config.transform_file is None:
            self.transform = SimilarityTransform.identity()
        else:
            with open(self.config.transform_file) as fp:
                transform = json.load(fp)
            scale, rotation, translation = map(
                torch.as_tensor,
                operator.itemgetter("scale", "rotation", "translation")(transform),
            )
            self.transform = SimilarityTransform(scale, rotation, translation)

        if self.config.latent_file is None:
            latents = torch.zeros(*self.ldm.latent_shape)
        else:
            latents = torch.load(self.config.latent_file)
        self.register_parameter("latents", nn.Parameter(latents))

    @override
    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.ldm.eval()
        return self

    def ldm_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.ldm.parameters()

    def field_parameters(self) -> Iterator[nn.Parameter]:
        yield from (
            parameter for name, parameter in self.named_parameters() if name.startswith("glin")
        )

    def latent_parameters(self) -> Iterator[nn.Parameter]:
        yield self.latents

    def nuisance_parameters(self) -> Iterator[nn.Parameter]:
        ldm_parameter_ids = set(map(id, self.ldm_parameters()))
        field_parameter_ids = set(map(id, self.field_parameters()))
        latent_parameter_ids = set(map(id, self.latent_parameters()))
        yield from (
            parameter
            for parameter in self.parameters()
            if id(parameter)
            not in {*ldm_parameter_ids, *field_parameter_ids, *latent_parameter_ids}
        )

    @torch.no_grad()
    def encode(self) -> jt.Float[torch.Tensor, " L C "]:
        if self.config.mcmc_space is MCMCSpace.LATENT:
            return self.latents
        elif self.config.mcmc_space is MCMCSpace.FIELD:
            with torch.autocast(self.aabb.device.type, enabled=False):
                mesh = generate_mesh_with_multires_marching_cubes(
                    geometry_callable_field=lambda positions: self.sdf(positions).squeeze(-1),
                    resolution=self.config.sdf_resolution,
                    bounding_box_min=tuple(self.aabb[0].tolist()),
                    bounding_box_max=tuple(self.aabb[1].tolist()),
                )
            mesh = self.transform.inverse()(mesh)
            latent_distribution = self.ldm.encode(mesh)
            if latent_distribution is None:
                return self.latents
            if self.config.encoder_out_mode is EncoderOutMode.MEAN:
                latents = latent_distribution.mean
            elif self.config.encoder_out_mode is EncoderOutMode.SAMPLE:
                latents = latent_distribution.sample()
        return latents

    @jaxtyped()
    def decode(self, latents: jt.Float[torch.Tensor, " L C "] | None = None) -> SDFType:
        latents = self.latents if latents is None else latents
        sdf = self.ldm.decode(latents)
        sdf = self.transform(sdf)
        return sdf

    @jaxtyped()
    @override
    def forward_geo_network(
        self,
        positions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B 1+{self.config.geo_feat_dim} "]:
        geo_outputs = super().forward_geo_network(positions)
        if self.config.mcmc_space is MCMCSpace.LATENT:
            sdf_values, geo_features = torch.split(
                tensor=geo_outputs,
                split_size_or_sections=(1, self.config.geo_feat_dim),
                dim=-1,
            )
            sdf_values = self.decode()(positions)
            geo_outputs = torch.cat([sdf_values, geo_features], dim=-1)
        return geo_outputs

    @jaxtyped()
    @override
    def sdf(
        self,
        positions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        if self.config.mcmc_space is MCMCSpace.LATENT:
            sdf_values = self.decode()(positions)
        elif self.config.mcmc_space is MCMCSpace.FIELD:
            geo_outputs = self.forward_geo_network(positions)
            sdf_values, _ = torch.split(geo_outputs, (1, self.config.geo_feat_dim), dim=-1)
        return sdf_values
