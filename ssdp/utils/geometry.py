from typing import Any, Self

import jaxtyping as jt
import torch
import torch.nn as nn
import trimesh
from scipy.spatial.transform import RigidTransform

from .aliases import SDFType
from .jaxtyping import jaxtyped


class SimilarityTransform(nn.Module):
    scale: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor

    @jaxtyped()
    def __init__(
        self,
        scale: jt.Float[torch.Tensor, " "],
        rotation: jt.Float[torch.Tensor, " 3 3 "],
        translation: jt.Float[torch.Tensor, " 3 "],
    ) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("rotation", rotation)
        self.register_buffer("translation", translation)

    @classmethod
    def identity(cls, **kwargs: Any) -> Self:
        return cls(
            scale=torch.ones((), **kwargs),
            rotation=torch.eye(3, **kwargs),
            translation=torch.zeros(3, **kwargs),
        )

    def inverse(self) -> Self:
        return type(self)(
            scale=1.0 / self.scale,
            rotation=self.rotation.T,
            translation=-self.rotation.T @ self.translation / self.scale,
        )

    def _transform_sdf(self, sdf: SDFType) -> SDFType:
        @jaxtyped()
        def transformed_sdf(
            positions: jt.Float[torch.Tensor, " *B 3 "],
        ) -> jt.Float[torch.Tensor, " *B 1 "]:
            positions = (positions - self.translation) @ self.rotation
            sdf_values = self.scale * sdf(positions / self.scale)
            return sdf_values

        return transformed_sdf

    def _transform_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        rigid_transform = RigidTransform.from_components(
            rotation=self.rotation.cpu().numpy(),
            translation=self.translation.cpu().numpy(),
        ).as_matrix()
        mesh = mesh.apply_scale(self.scale)
        mesh = mesh.apply_transform(rigid_transform)
        return mesh

    def forward(
        self,
        inputs: trimesh.Trimesh | SDFType,
    ) -> trimesh.Trimesh | SDFType:
        if isinstance(inputs, trimesh.Trimesh):
            return self._transform_mesh(inputs)
        else:
            return self._transform_sdf(inputs)
