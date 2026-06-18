from collections.abc import Callable
from typing import override

import jaxtyping as jt
import torch
import torch.nn as nn

from nerfstudio.data.scene_box import SceneBox
from ssdp.utils.jaxtyping import jaxtyped


class SphereTracer(nn.Module):
    def __init__(
        self,
        scene_box: SceneBox | None = None,
        num_iterations: int = 1000,
        distance_threshold: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.scene_box = scene_box
        self.num_iterations = num_iterations
        self.distance_threshold = distance_threshold

    @jaxtyped()
    def _intersect_aabb(
        self,
        ray_origins: jt.Float[torch.Tensor, " *B 3 "],
        ray_directions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> tuple[
        tuple[
            jt.Float[torch.Tensor, " *B 1 "],
            jt.Float[torch.Tensor, " *B 1 "],
        ],
        jt.Bool[torch.Tensor, " *B 1 "],
    ]:
        aabb_min, aabb_max = self.scene_box.aabb
        min_distances = (aabb_min - ray_origins) / ray_directions
        max_distances = (aabb_max - ray_origins) / ray_directions
        min_distances, max_distances = (
            torch.minimum(min_distances, max_distances),
            torch.maximum(min_distances, max_distances),
        )
        min_distances = torch.amax(min_distances, dim=-1, keepdim=True)
        max_distances = torch.amin(max_distances, dim=-1, keepdim=True)
        intersection_masks = (min_distances <= max_distances) & (max_distances >= 0.0)
        return (min_distances, max_distances), intersection_masks

    @jaxtyped()
    @override
    def forward(
        self,
        sdf: Callable[
            [jt.Float[torch.Tensor, " *B 3 "]],
            jt.Float[torch.Tensor, " *B 1 "],
        ],
        ray_origins: jt.Float[torch.Tensor, " *B 3 "],
        ray_directions: jt.Float[torch.Tensor, " *B 3 "],
        foreground_masks: jt.Bool[torch.Tensor, " *B 1 "] | None = None,
        aabb_initialization: bool = False,
    ) -> tuple[
        jt.Float[torch.Tensor, " *B 3 "],
        jt.Bool[torch.Tensor, " *B 1 "],
    ]:
        """Differentiable sphere tracer based on implicit differentiation.

        References:
            - [Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance](https://arxiv.org/abs/2003.09852)
            - [Differentiable Volumetric Rendering: Learning Implicit 3D Representations without 3D Supervision](https://arxiv.org/abs/1912.07372)
        """
        if foreground_masks is None:
            foreground_masks = torch.all(torch.isfinite(ray_origins), dim=-1, keepdim=True)

        if self.scene_box and aabb_initialization:
            (min_distances, _), intersection_masks = self._intersect_aabb(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
            )
            min_distances = torch.clamp(min_distances, min=0.0)
            ray_origins = torch.where(
                condition=intersection_masks,
                input=ray_origins + ray_directions * min_distances,
                other=ray_origins,
            )
            foreground_masks &= intersection_masks

        convergence_masks = torch.zeros_like(foreground_masks)

        ray_positions = ray_origins

        with torch.no_grad():
            for _ in range(self.num_iterations):
                sdf_values = sdf(ray_positions)
                convergence_masks |= torch.abs(sdf_values) < self.distance_threshold

                survival_masks = foreground_masks & ~convergence_masks
                if not torch.any(survival_masks):
                    break

                ray_positions = torch.where(
                    condition=survival_masks,
                    input=ray_positions + ray_directions * sdf_values,
                    other=ray_positions,
                )

                if self.scene_box:
                    inside_masks = self.scene_box.within(ray_positions)
                    inside_masks = inside_masks.unsqueeze(-1)
                    foreground_masks &= inside_masks

        surface_positions = ray_positions

        if torch.is_grad_enabled():
            with torch.enable_grad():
                surface_positions.requires_grad_(True)
                sdf_values = sdf(surface_positions)
                [sdf_gradients] = torch.autograd.grad(
                    outputs=sdf_values,
                    inputs=surface_positions,
                    grad_outputs=torch.ones_like(sdf_values),
                    retain_graph=True,
                    create_graph=False,
                )
                sdf_gradients = torch.sum(sdf_gradients * ray_directions, dim=-1, keepdim=True)
                sdf_gradients = torch.where(
                    condition=sdf_gradients >= 0.0,
                    input=torch.clamp(sdf_gradients, min=+1.0e-6),
                    other=torch.clamp(sdf_gradients, max=-1.0e-6),
                )
                surface_distances = -sdf_values / sdf_gradients
                surface_positions = torch.where(
                    condition=convergence_masks,
                    input=surface_positions + ray_directions * surface_distances,
                    other=surface_positions,
                )

        return surface_positions, convergence_masks
