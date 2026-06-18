from collections.abc import Callable
from typing import override

import jaxtyping as jt
import torch
import torch.nn as nn

from ssdp.utils.jaxtyping import jaxtyped

from .sphere_tracer import SphereTracer


class ShadowRenderer(nn.Module):
    def __init__(self, sphere_tracer: SphereTracer) -> None:
        super().__init__()
        self.sphere_tracer = sphere_tracer

    @jaxtyped()
    @override
    def forward(
        self,
        sdf: Callable[
            [jt.Float[torch.Tensor, " *B 3 "]],
            jt.Float[torch.Tensor, " *B 1 "],
        ],
        surface_positions: jt.Float[torch.Tensor, " *B 3 "],
        surface_normals: jt.Float[torch.Tensor, " *B 3 "],
        light_directions: jt.Float[torch.Tensor, " *B 3 "],
        foreground_masks: jt.Bool[torch.Tensor, " *B 1 "],
    ) -> jt.Bool[torch.Tensor, " *B 1 "]:
        """Shadow renderer based on second sphere tracing.

        References:
            [Inigo Quilez](https://iquilezles.org/articles/rmshadows/)
        """
        ray_origins = surface_positions + surface_normals * self.sphere_tracer.distance_threshold
        ray_directions = -light_directions
        surface_positions, convergence_masks = self.sphere_tracer(
            sdf=sdf,
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            foreground_masks=foreground_masks,
        )
        shadow_masks = foreground_masks & convergence_masks
        return shadow_masks
