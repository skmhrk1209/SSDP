import dataclasses
import json
from pathlib import Path

import jaxtyping as jt
import loguru
import numpy as np
import torch
import tyro

from nerfstudio.utils.eval_utils import eval_setup
from ssdp.fields.ssdp import SSDP
from ssdp.utils.jaxtyping import jaxtyped
from tools.analysis.evaluate_approx_error import CuboidConfig
from tools.analysis.evaluate_training_trajectory import (
    get_world_to_model_matrix,
    transform_points,
)


@dataclasses.dataclass
class SDFErrorEvaluator:
    # NOTE: The geometry reconstructed by a training run of `evaluate_training_trajectory.sh` against the exact
    # slab (`CuboidConfig`), from the learned mean SDF: the thickness of the slab along rays parallel to the x-axis,
    # the learned SDF on the exact faces, and the learned SDF on the plane z = 0 for the pictures
    # (`plot_sdf_error.py`). Every position is in the world coordinates of the dataset (those of the ground-truth
    # mesh), mapped to those of the model to query the field (`get_world_to_model_matrix`; the map has to be
    # a similarity, whose scale converts the learned SDF back to the units of the world). The metrics go to
    # the output file, the slice to the `.npz` file next to it.
    config_file: Path
    output_file: Path
    cuboid_config: CuboidConfig = dataclasses.field(
        default_factory=CuboidConfig,
    )
    # NOTE: The grid of (y, z) over the broad faces of the slab, away from its edges, on which the rays parallel to
    # the x-axis (the thickness) and the points on the exact faces (the learned SDF there) are placed.
    grid_extent: float = 0.4
    num_grid_points: int = 81
    # NOTE: The thickness of the slab on a ray is the length of the set where the learned SDF is negative within
    # this distance of the center of the slab, from the learned SDF sampled at these points along the ray
    # (the zero-crossings are interpolated linearly between the samples); a slab whose set is empty has vanished
    # on that ray.
    search_half_width: float = 0.05
    num_search_samples: int = 101
    # NOTE: The slice z = 0 over these ranges (around the slab, over its extent in y), sampled at this number of
    # points along each axis.
    slice_x_range: tuple[float, float] = (-0.05, 0.05)
    slice_y_range: tuple[float, float] = (-0.5, 0.5)
    num_slice_samples: int = 1001
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    chunk_size: int = 1 << 18

    @jaxtyped()
    def _get_scale(self, matrix: jt.Float[torch.Tensor, " 4 4 "]) -> float:
        # NOTE: The scale of the similarity from the world to the model.
        linear = matrix[:3, :3]
        scale = torch.linalg.det(linear).abs().pow(1.0 / 3.0)
        assert torch.allclose(linear.T @ linear, scale**2.0 * torch.eye(3), atol=1.0e-5), matrix
        return scale.item()

    @jaxtyped()
    @torch.no_grad()
    def _get_sdf_values(
        self,
        field: SSDP,
        matrix: jt.Float[torch.Tensor, " 4 4 "],
        positions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B "]:
        # NOTE: The learned mean SDF (the first output of the geometry network) at the positions given in the world
        # coordinates, in their units.
        flat_positions = transform_points(matrix, positions.reshape(-1, 3))
        sdf_values = torch.cat(
            [
                field.forward_geo_network(flat_positions[start : start + self.chunk_size])[:, 0]
                for start in range(0, len(flat_positions), self.chunk_size)
            ]
        )
        return sdf_values.reshape(positions.shape[:-1]) / self._get_scale(matrix)

    @jaxtyped()
    def _summarize(self, values: jt.Float[torch.Tensor, " P "]) -> dict[str, float]:
        values = values.double()
        summary = dict(mean=values.mean().item())
        for quantile in self.quantiles:
            summary[f"q{round(quantile * 100):02d}"] = torch.quantile(values, quantile).item()
        return summary

    @jaxtyped()
    def _get_grid(
        self,
        device: torch.device,
    ) -> tuple[jt.Float[torch.Tensor, " G G "], jt.Float[torch.Tensor, " G G "]]:
        coords = torch.linspace(
            -self.grid_extent, self.grid_extent, self.num_grid_points, device=device
        )
        return torch.meshgrid(coords, coords, indexing="ij")

    @jaxtyped()
    def _get_negative_lengths(
        self,
        x: jt.Float[torch.Tensor, " X "],
        sdf_values: jt.Float[torch.Tensor, " *R X "],
    ) -> jt.Float[torch.Tensor, " *R "]:
        # NOTE: The length of the set where the SDF is negative, with the zero-crossings interpolated linearly
        # between the samples.
        left_values, right_values = sdf_values[..., :-1], sdf_values[..., 1:]
        crossings = left_values / (left_values - right_values)
        fractions = torch.where(
            (left_values < 0.0) & (right_values < 0.0),
            torch.ones_like(crossings),
            torch.where(
                (left_values >= 0.0) & (right_values >= 0.0),
                torch.zeros_like(crossings),
                torch.where(left_values < 0.0, crossings, 1.0 - crossings),
            ),
        )
        return torch.sum(fractions * torch.diff(x), dim=-1)

    @torch.no_grad()
    def _evaluate_thickness(self, field: SSDP, matrix: torch.Tensor) -> list[dict]:
        device = field.aabb.device
        y, z = self._get_grid(device)

        slabs = []
        for slab_x, _, _ in self.cuboid_config.positions:
            x = torch.linspace(
                slab_x - self.search_half_width,
                slab_x + self.search_half_width,
                self.num_search_samples,
                device=device,
            )
            positions = torch.stack(
                torch.broadcast_tensors(x[None, None, :], y[..., None], z[..., None]),
                dim=-1,
            )
            thicknesses = self._get_negative_lengths(
                x, self._get_sdf_values(field, matrix, positions).flatten(0, 1)
            )
            slabs.append(
                dict(
                    center=slab_x,
                    true_thickness=2.0 * self.cuboid_config.radii[0],
                    thickness=self._summarize(thicknesses),
                    num_vanished_rays=int((thicknesses == 0.0).sum()),
                    num_rays=len(thicknesses),
                )
            )
        return slabs

    @torch.no_grad()
    def _evaluate_faces(self, field: SSDP, matrix: torch.Tensor) -> dict[str, dict]:
        # NOTE: The learned SDF on the exact faces: its absolute value is the error of the learned surface, and
        # its sign tells the side (negative: the learned surface lies outside the exact one).
        y, z = self._get_grid(field.aabb.device)
        sdf_values = {}
        for index, (slab_x, _, _) in enumerate(self.cuboid_config.positions):
            for side, sign in (("front", -1.0), ("back", 1.0)):
                face_x = slab_x + sign * self.cuboid_config.radii[0]
                positions = torch.stack([torch.full_like(y, face_x), y, z], dim=-1)
                sdf_values[f"slab_{index}_{side}"] = self._get_sdf_values(
                    field, matrix, positions
                ).flatten()
        sdf_values["all_faces"] = torch.cat(list(sdf_values.values()))
        return {
            face: dict(
                num_points=len(values),
                signed_sdf=self._summarize(values),
                absolute_sdf=self._summarize(values.abs()),
            )
            for face, values in sdf_values.items()
        }

    @torch.no_grad()
    def _evaluate_slice(self, field: SSDP, matrix: torch.Tensor) -> dict[str, np.ndarray]:
        # NOTE: The learned and the exact SDF on the plane z = 0, as (x, y, values) with the values indexed by (x, y).
        device = field.aabb.device
        x, y = (
            torch.linspace(lower, upper, self.num_slice_samples, device=device)
            for lower, upper in (self.slice_x_range, self.slice_y_range)
        )
        positions = torch.stack(
            torch.broadcast_tensors(x[:, None], y[None, :], torch.zeros((), device=device)),
            dim=-1,
        )
        return dict(
            x=x.cpu().numpy(),
            y=y.cpu().numpy(),
            sdf=self._get_sdf_values(field, matrix, positions).cpu().numpy(),
            exact_sdf=self.cuboid_config.get_sdf()(positions).squeeze(-1).cpu().numpy(),
        )

    def __call__(self) -> None:
        _, pipeline, _, step = eval_setup(self.config_file)
        field: SSDP = pipeline.model.field
        matrix = get_world_to_model_matrix(pipeline)

        record = dict(
            config=dict(
                step=step,
                cuboid_config=dataclasses.asdict(self.cuboid_config),
                grid_extent=self.grid_extent,
                num_grid_points=self.num_grid_points,
                search_half_width=self.search_half_width,
                num_search_samples=self.num_search_samples,
                world_to_model_matrix=matrix.tolist(),
            ),
            metrics=dict(
                slabs=self._evaluate_thickness(field, matrix),
                faces=self._evaluate_faces(field, matrix),
            ),
        )

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as fp:
            json.dump(record, fp, indent=4)
        np.savez_compressed(
            self.output_file.with_suffix(".npz"), **self._evaluate_slice(field, matrix)
        )

        loguru.logger.success(f"Saved the SDF errors to <{self.output_file}>.")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        SDFErrorEvaluator,
        config=(tyro.conf.AvoidSubcommands,),
    )()
