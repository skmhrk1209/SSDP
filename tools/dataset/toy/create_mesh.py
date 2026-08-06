import dataclasses
from pathlib import Path
from typing import Annotated

import loguru
import numpy as np
import shapely
import trimesh
import tyro

NUM_SECTIONS = 50


@dataclasses.dataclass
class SlabConfig:
    radii: tuple[float, float, float] = (0.05, 0.5, 0.5)
    range: tuple[float, float] = (-0.5, 0.5)
    count: int = 3

    def instantiate(self) -> trimesh.Trimesh:
        meshes = [
            trimesh.creation.box(
                bounds=np.add(
                    [np.negative(self.radii), self.radii],
                    [x, 0.0, 0.0],
                ),
            )
            for x in np.linspace(*self.range, self.count)
        ]
        mesh = trimesh.util.concatenate(meshes)
        return mesh


@dataclasses.dataclass
class TubeConfig:
    thickness: float = 0.01
    height: float = 1.0
    range: tuple[float, float] = (0.1, 0.5)
    count: int = 3

    def instantiate(self) -> trimesh.Trimesh:
        meshes = [
            trimesh.creation.annulus(
                r_min=radius - self.thickness,
                r_max=radius + self.thickness,
                height=self.height,
                sections=NUM_SECTIONS,
            )
            for radius in np.linspace(*self.range, self.count)
        ]
        mesh = trimesh.util.concatenate(meshes)
        return mesh


@dataclasses.dataclass
class HelixConfig:
    major_radius: float = 0.5
    minor_radius: float = 0.1
    range: tuple[float, float] = (-0.5, 0.5)
    count: int = 3

    def instantiate(self) -> trimesh.Trimesh:
        θ = np.linspace(-np.pi, np.pi, NUM_SECTIONS, endpoint=False)
        xy = np.stack([np.cos(θ), np.sin(θ)], axis=-1)
        polygon = shapely.Polygon(xy * self.minor_radius)
        x, y = np.tile(xy * self.major_radius, (self.count, 1)).T
        z = np.linspace(*self.range, self.count * NUM_SECTIONS)
        path = np.stack([x, y, z], axis=-1)
        mesh = trimesh.creation.sweep_polygon(polygon, path)
        return mesh


@dataclasses.dataclass
class SphereConfig:
    radius: float = 0.15
    range: tuple[float, float] = (-0.5, 0.5)
    count: int = 3

    def instantiate(self) -> trimesh.Trimesh:
        positions = np.meshgrid(*[np.linspace(*self.range, self.count)] * 3)
        positions = np.stack(positions, axis=-1).reshape(-1, 3)
        mesh = trimesh.creation.icosphere(radius=self.radius)
        meshes = [mesh.copy().apply_translation(position) for position in positions]
        mesh = trimesh.util.concatenate(meshes)
        return mesh


@dataclasses.dataclass
class MeshCreator:
    output_file: Path
    shape_config: (
        Annotated[SlabConfig, tyro.conf.subcommand(name="slab")]
        | Annotated[TubeConfig, tyro.conf.subcommand(name="tube")]
        | Annotated[HelixConfig, tyro.conf.subcommand(name="helix")]
        | Annotated[SphereConfig, tyro.conf.subcommand(name="sphere")]
    ) = dataclasses.field(
        default_factory=SlabConfig,
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
