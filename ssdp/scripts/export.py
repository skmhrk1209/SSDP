import dataclasses
from pathlib import Path

import loguru
import tyro

from nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes
from nerfstudio.utils.eval_utils import eval_setup
from ssdp.models import NeuSModel


@dataclasses.dataclass
class MarchingCubesExporter:
    config_file: Path
    output_file: Path
    resolution: int = 512
    threshold: float = 0.0

    def __call__(self) -> None:
        _, pipeline, _, _ = eval_setup(self.config_file)

        loguru.logger.success("🚀 Marching Cubes started!")

        model = pipeline.model
        assert isinstance(model, NeuSModel)

        mesh = generate_mesh_with_multires_marching_cubes(
            geometry_callable_field=lambda positions: model.field.sdf(positions).squeeze(-1),
            resolution=self.resolution,
            bounding_box_min=tuple(model.field.aabb[0, ...].tolist()),
            bounding_box_max=tuple(model.field.aabb[1, ...].tolist()),
            isosurface_threshold=self.threshold,
        )

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(self.output_file)

        loguru.logger.success("✨ Marching Cubes finished!")


def entrypoint() -> None:
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(MarchingCubesExporter)()


if __name__ == "__main__":
    entrypoint()
