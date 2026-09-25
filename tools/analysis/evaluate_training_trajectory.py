import dataclasses
import json
from pathlib import Path

import jaxtyping as jt
import loguru
import torch
import tqdm
import trimesh
import tyro
from matplotlib import colors
from trimesh.ray.ray_pyembree import RayMeshIntersector

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.model_components.ray_samplers import UniformSampler
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
from ssdp.fields.ssdp import SSDP
from ssdp.utils.jaxtyping import jaxtyped
from tools.analysis.evaluate_approx_error import (
    MonteCarloConfig,
    _compute_metric_errors,
    _compute_metrics,
    _get_bin_values,
    _get_predictive_cdf_values,
    _get_predictive_pmf_values,
    _get_reference_cdf_values,
    _get_transition_params,
)


def get_world_to_model_matrix(pipeline: Pipeline) -> torch.Tensor:
    # NOTE: The 4 x 4 matrix from the world of the dataset (where the ground-truth mesh is) to the coordinates of
    # the model: the transform that the dataparser applied to the poses, if any.
    transform: torch.Tensor | None = pipeline.datamanager.train_dataset.metadata.get("transform")
    if transform is None:
        return torch.eye(4)
    transform = transform.cpu()
    return torch.cat([transform, transform.new_tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)


@jaxtyped()
def transform_points(
    matrix: jt.Float[torch.Tensor, " 4 4 "],
    points: jt.Float[torch.Tensor, " *B 3 "],
) -> jt.Float[torch.Tensor, " *B 3 "]:
    matrix = matrix.to(points)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


@dataclasses.dataclass
class TrainingTrajectoryEvaluator:
    config_file: Path
    target_mesh_file: Path
    output_file: Path
    num_samples: int = 48
    # NOTE: Rays parallel to the x-axis through the slabs, sampled as in `evaluate_approx_error.py`, on the grid
    # {-0.4, -0.3, ..., 0.4}^2 of (y, z), where the exact SDF along the rays equals that of the infinite slabs.
    # They are defined in the world of the dataset (the coordinates of the ground-truth mesh) and mapped to those of
    # the model (`get_world_to_model_matrix`).
    ray_x_range: tuple[float, float] = (-1.0, 1.0)
    num_axis_rays: int = 9
    axis_ray_extent: float = 0.4
    # NOTE: The rays of the dataset cameras on a grid of pixels with this stride (as in `evaluate_uq_metrics.py`),
    # among which those hitting the ground-truth mesh are evaluated.
    camera_ray_stride: int = 100
    # NOTE: The Monte Carlo reference of `evaluate_approx_error.py`, with ten checkpoints per job of at most an hour
    # (`MonteCarloConfig`). The standard error of the Monte Carlo CDF of a ray is then at most
    # 1 / (2 sqrt(num_mc_samples)) before the extrapolation, and the errors of the distances of each ray from it are
    # recorded (`_compute_metric_errors`); the medians and the quartiles over the rays, which are what the figures
    # show, are far more precise.
    monte_carlo: MonteCarloConfig = dataclasses.field(
        default_factory=lambda: MonteCarloConfig(num_mc_samples=1000),
    )
    # NOTE: The paths are seeded at each checkpoint.
    random_seed: int = 42
    # NOTE: The checkpoints are evaluated in slices, e.g., for evaluating them in parallel.
    checkpoint_slice: tuple[int, int] = (0, 100)
    # NOTE: The rays are evaluated in chunks of this size (for the GPU memory only; the path sampling is a loop over
    # the substeps, so smaller chunks cost proportionally more time).
    ray_chunk_size: int = 64
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)

    def _load_checkpoint(self, pipeline: Pipeline, checkpoint_file: Path) -> int:
        loaded_state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
        pipeline.eval()
        return loaded_state["step"]

    def _get_ray_samples(self, ray_bundle: RayBundle) -> RaySamples:
        ray_sampler = UniformSampler(
            num_samples=self.num_samples,
            train_stratified=False,
        )
        ray_samples = ray_sampler(ray_bundle)
        return ray_samples

    def _get_axis_ray_bundle(self, matrix: torch.Tensor, device: torch.device) -> RayBundle:
        coords = torch.linspace(-self.axis_ray_extent, self.axis_ray_extent, self.num_axis_rays)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing="ij"), dim=-1).reshape(-1, 2)
        # NOTE: The origins and the direction in the world, mapped to the coordinates of the model (the length of
        # the direction there scales the parameter of the rays).
        origins = transform_points(
            matrix, torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1)
        )
        directions = torch.tensor([[1.0, 0.0, 0.0]]) @ matrix[:3, :3].T
        lengths = torch.linalg.norm(directions, dim=-1, keepdim=True)

        ray_bundle = RayBundle(
            origins=origins,
            directions=(directions / lengths).expand_as(origins),
            pixel_area=torch.zeros_like(origins[:, :1]),
            camera_indices=torch.zeros_like(origins[:, :1], dtype=torch.long),
            nears=(self.ray_x_range[0] * lengths).expand_as(origins[:, :1]),
            fars=(self.ray_x_range[1] * lengths).expand_as(origins[:, :1]),
        )
        ray_bundle = ray_bundle.to(device)

        return ray_bundle

    def _load_target_mesh(self, matrix: torch.Tensor) -> trimesh.Trimesh:
        target_mesh = trimesh.load_mesh(self.target_mesh_file)
        return target_mesh.apply_transform(matrix.numpy())

    @torch.no_grad()
    def _get_camera_ray_bundle(self, pipeline: Pipeline, matrix: torch.Tensor) -> RayBundle:
        # NOTE: Foreground rays, i.e., the rays hitting the ground-truth mesh between the near and far planes
        # of the collider (the same criterion as `evaluate_uq_metrics.py`).
        intersector = RayMeshIntersector(self._load_target_mesh(matrix), scale_to_box=False)

        ray_bundles = []
        for camera, _ in pipeline.datamanager.fixed_indices_eval_dataloader:
            coords = camera.get_image_coords().to(camera.device)
            coords = coords[~torch.any(coords.long() % self.camera_ray_stride, dim=-1)]
            ray_bundle = pipeline.model.collider(
                camera.generate_rays(camera_indices=0, coords=coords)
            )

            locations, ray_indices, _ = intersector.intersects_location(
                ray_origins=ray_bundle.origins.cpu().numpy(),
                ray_directions=ray_bundle.directions.cpu().numpy(),
                multiple_hits=False,
            )
            locations = ray_bundle.origins.new_tensor(locations)
            ray_indices = torch.as_tensor(ray_indices, device=locations.device)
            depths = torch.linalg.vecdot(
                locations - ray_bundle.origins[ray_indices],
                ray_bundle.directions[ray_indices],
                dim=-1,
            )
            inside = (depths > ray_bundle.nears[ray_indices, 0]) & (
                depths < ray_bundle.fars[ray_indices, 0]
            )
            ray_bundles.append(ray_bundle[ray_indices[inside]])

        ray_bundle = RayBundle(
            origins=torch.cat([bundle.origins for bundle in ray_bundles], dim=0),
            directions=torch.cat([bundle.directions for bundle in ray_bundles], dim=0),
            pixel_area=torch.cat([bundle.pixel_area for bundle in ray_bundles], dim=0),
            camera_indices=torch.cat([bundle.camera_indices for bundle in ray_bundles], dim=0),
            nears=torch.cat([bundle.nears for bundle in ray_bundles], dim=0),
            fars=torch.cat([bundle.fars for bundle in ray_bundles], dim=0),
        )
        loguru.logger.info(f"{len(ray_bundle)} camera rays hit the ground-truth mesh.")

        return ray_bundle

    @jaxtyped()
    def _summarize(self, values: jt.Float[torch.Tensor, " R "]) -> dict[str, float]:
        summary = dict(mean=values.mean().item())
        for quantile in self.quantiles:
            summary[f"q{round(quantile * 100):02d}"] = torch.quantile(values, quantile).item()
        return summary

    @torch.no_grad()
    def _evaluate_ou_params(
        self,
        field: SSDP,
        ray_samples: RaySamples,
    ) -> dict[str, float]:
        # NOTE: The analytic field of `evaluate_approx_error.py` has a single (kappa, tau^2) for all the intervals,
        # whereas the learned ones depend on the interval. They are reduced to a single pair by the median
        # over the intervals of each ray, averaged over the rays.
        _, transition_scales, _, transition_vars = _get_transition_params(
            field=field,
            ray_samples=ray_samples,
            num_sub_samples=1,
        )
        # NOTE: The OU parameters that the renderer effectively uses, i.e., after the floor of the transition variance.
        intervals = ray_samples.deltas.squeeze(-1)
        ou_drifts = -torch.log(transition_scales) / intervals
        ou_diffusions = (
            transition_vars * (2.0 * ou_drifts) / -torch.expm1(-2.0 * ou_drifts * intervals)
        )

        metrics = dict(initial_var=field._get_initial_var().item())
        for name, values in dict(
            transition_scale=transition_scales,
            transition_var=transition_vars,
            ou_drift=ou_drifts,
            ou_diffusion=ou_diffusions,
        ).items():
            metrics[name] = torch.mean(torch.median(values, dim=-1).values).item()
        return metrics

    @torch.no_grad()
    def _get_reference_cdf_values(
        self,
        field: SSDP,
        ray_samples: RaySamples,
    ) -> tuple[
        jt.Float[torch.Tensor, " R S "],
        jt.Float[torch.Tensor, " R S S "],
    ]:
        # NOTE: Monte Carlo first passages of the learned SDE, in chunks of rays.
        cdf_values, cdf_covariances = zip(
            *[
                _get_reference_cdf_values(
                    field=field,
                    ray_samples=ray_samples[start : start + self.ray_chunk_size],
                    monte_carlo=self.monte_carlo,
                )
                for start in range(0, len(ray_samples), self.ray_chunk_size)
            ],
            strict=True,
        )
        return torch.cat(cdf_values, dim=0), torch.cat(cdf_covariances, dim=0)

    @torch.no_grad()
    def _evaluate_approx_error(
        self,
        field: SSDP,
        ray_samples: RaySamples,
    ) -> dict[str, dict[str, float]]:
        # NOTE: The renderer used in training against the Monte Carlo reference of the same learned field,
        # summarized over all the rays.
        bin_values = _get_bin_values(ray_samples)
        cdf_values = _get_predictive_cdf_values(_get_predictive_pmf_values(field, ray_samples))
        reference_cdf_values, reference_cdf_covariances = self._get_reference_cdf_values(
            field, ray_samples
        )
        metrics = _compute_metrics(
            bin_values=bin_values,
            cdf_values_1=cdf_values,
            cdf_values_2=reference_cdf_values,
        )
        # NOTE: The Monte Carlo errors of the distances of each ray, summarized like the distances.
        metrics.update(
            _compute_metric_errors(
                bin_values=bin_values,
                cdf_values_1=cdf_values,
                cdf_values_2=reference_cdf_values,
                cdf_covariances_2=reference_cdf_covariances,
            )
        )
        return {name: self._summarize(values) for name, values in metrics.items()}

    def __call__(self) -> None:
        config, pipeline, checkpoint_file, _ = eval_setup(self.config_file)
        field: SSDP = pipeline.model.field

        checkpoint_files = sorted(checkpoint_file.parent.glob("step-*.ckpt"))[
            slice(*self.checkpoint_slice)
        ]

        # NOTE: The learned parameters are taken on the axis rays, which match the setting of the analytic slabs,
        # and the errors on the camera rays, which are those used in training.
        matrix = get_world_to_model_matrix(pipeline)
        axis_ray_samples = self._get_ray_samples(
            self._get_axis_ray_bundle(matrix, field.aabb.device)
        )
        camera_ray_samples = self._get_ray_samples(self._get_camera_ray_bundle(pipeline, matrix))

        records = []

        for checkpoint_file in tqdm.tqdm(
            iterable=checkpoint_files,
            colour=colors.to_hex("dodgerblue"),
            desc="Evaluating checkpoints...",
        ):
            step = self._load_checkpoint(pipeline, checkpoint_file)
            torch.manual_seed(self.random_seed)

            # NOTE: The renderer is the one used at this training step.
            field.set_progress_ratio(min(1.0, step / config.max_num_iterations))

            metrics = self._evaluate_ou_params(field, axis_ray_samples)
            metrics |= self._evaluate_approx_error(field, camera_ray_samples)

            records.append(
                dict(
                    config=dict(
                        step=step,
                        monte_carlo=dataclasses.asdict(self.monte_carlo),
                        random_seed=self.random_seed,
                    ),
                    metrics=metrics,
                )
            )

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as fp:
            json.dump(records, fp, indent=4)

        loguru.logger.success(f"Saved the training trajectory to <{self.output_file}>.")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        TrainingTrajectoryEvaluator,
        config=(tyro.conf.AvoidSubcommands,),
    )()
