import dataclasses
from pathlib import Path
from typing import Annotated

import loguru
import numpy as np
import trimesh
import tyro


@dataclasses.dataclass
class CuboidConfig:
    colors: list[tuple[float, float, float]] = dataclasses.field(
        default_factory=lambda: [
            (1.0, 0.75, 0.25),
            (0.25, 0.5, 1.0),
            (1.0, 0.25, 0.5),
        ],
    )
    radii: tuple[float, float, float] = (0.025, 0.5, 0.5)
    range: tuple[float, float] = (-0.5, 0.5)
    count: int = 3

    def instantiate(self) -> trimesh.Trimesh:
        meshes = [
            trimesh.creation.box(
                bounds=np.add(
                    [np.negative(self.radii), self.radii],
                    [x, 0.0, 0.0],
                ),
                vertex_colors=color,
            )
            for x, color in zip(
                np.linspace(*self.range, self.count),
                self.colors,
                strict=True,
            )
        ]
        mesh = trimesh.util.concatenate(meshes)
        return mesh


@dataclasses.dataclass
class SphereConfig:
    radius: float = 0.125
    range: tuple[float, float] = (-0.375, 0.375)
    count: int = 3

    def instantiate(self) -> trimesh.Trimesh:
        positions = np.meshgrid(*[np.linspace(*self.range, self.count)] * 3)
        positions = np.stack(positions, axis=-1).reshape(-1, 3)
        mesh = trimesh.creation.icosphere(radius=self.radius)
        meshes = [mesh.copy().apply_translation(position) for position in positions]
        colors = np.linspace(0.1, 0.9, self.count)
        indices = np.meshgrid(*[np.arange(self.count)] * 3)
        indices = np.stack(indices, axis=-1).reshape(-1, 3)
        for mesh, (i, j, k) in zip(meshes, indices, strict=True):
            r = (2 * i + j + k) % len(colors)
            g = (i + 2 * j + k) % len(colors)
            b = (i + j + 2 * k) % len(colors)
            mesh.visual.vertex_colors = colors[[r, g, b]]
        mesh: trimesh.Trimesh = trimesh.util.concatenate(meshes)
        return mesh


@dataclasses.dataclass
class MeshCreator:
    output_file: Path
    shape_config: (
        Annotated[CuboidConfig, tyro.conf.subcommand(name="cuboid")]
        | Annotated[SphereConfig, tyro.conf.subcommand(name="sphere")]
    ) = dataclasses.field(
        default_factory=CuboidConfig,
    )

    def __call__(self) -> None:
        mesh = self.shape_config.instantiate()

        assert mesh.is_volume

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(self.output_file)

        loguru.logger.success("Finished!")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        MeshCreator,
        config=(tyro.conf.OmitArgPrefixes,),
    )()
