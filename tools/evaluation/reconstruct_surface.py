import dataclasses
from pathlib import Path

import loguru
import numpy as np
import open3d
import tyro


@dataclasses.dataclass
class SurfaceReconstructor:
    point_file: Path
    output_file: Path
    num_neighbors: int = 100
    max_octree_depth: int = 9
    density_quantile: float = 0.01

    def _reconstruct_surface(
        self,
        point_cloud: open3d.geometry.PointCloud,
    ) -> open3d.geometry.TriangleMesh:
        if not point_cloud.has_normals():
            point_cloud.estimate_normals()
        if self.num_neighbors:
            point_cloud.orient_normals_consistent_tangent_plane(self.num_neighbors)
        mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd=point_cloud,
            depth=self.max_octree_depth,
        )
        if self.density_quantile:
            vertex_mask = densities < np.quantile(densities, self.density_quantile)
            mesh.remove_vertices_by_mask(vertex_mask)
        return mesh

    def __call__(self) -> None:
        point_cloud = open3d.io.read_point_cloud(self.point_file)
        mesh = self._reconstruct_surface(point_cloud)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        open3d.io.write_triangle_mesh(self.output_file, mesh)
        loguru.logger.success(f"Saved the reconstructed mesh to <{self.output_file}>.")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        SurfaceReconstructor,
        config=(tyro.conf.AvoidSubcommands,),
    )()
