from collections.abc import Callable
from typing import override

import jaxtyping as jt
import torch
import torch.nn as nn

from ssdp.utils.jaxtyping import jaxtyped


class NormalEstimator(nn.Module):
    def __init__(self, finite_diff_epsilon: float | None = None) -> None:
        super().__init__()
        self.finite_diff_epsilon = finite_diff_epsilon

    @jaxtyped()
    @override
    def forward(
        self,
        sdf: Callable[
            [jt.Float[torch.Tensor, " *B 3 "]],
            jt.Float[torch.Tensor, " *B 1 "],
        ],
        surface_positions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B 3 "]:
        if self.finite_diff_epsilon:
            finite_diff_epsilons = torch.full_like(surface_positions, self.finite_diff_epsilon)
            finite_diff_epsilons = torch.diag_embed(finite_diff_epsilons, dim1=0, dim2=-1)
            pos_sdf_values = sdf(surface_positions + finite_diff_epsilons)
            neg_sdf_values = sdf(surface_positions - finite_diff_epsilons)
            sdf_gradients = (pos_sdf_values - neg_sdf_values) / (self.finite_diff_epsilon * 2.0)
            sdf_gradients = sdf_gradients.squeeze(-1).movedim(0, -1)
        else:
            create_graph = torch.is_grad_enabled()
            with torch.enable_grad():
                surface_positions.requires_grad_(True)
                sdf_values = sdf(surface_positions)
                [sdf_gradients] = torch.autograd.grad(
                    outputs=sdf_values,
                    inputs=surface_positions,
                    grad_outputs=torch.ones_like(sdf_values),
                    create_graph=create_graph,
                )

        surface_normals = nn.functional.normalize(sdf_gradients, p=2, dim=-1)

        return surface_normals
