import collections
import dataclasses
import json
from operator import itemgetter
from pathlib import Path
from typing import Any

import jaxtyping as jt
import loguru
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as transforms
import tqdm
import trimesh
import tyro
from matplotlib import colors
from trimesh.ray.ray_pyembree import RayMeshIntersector

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.sdf_field import SDFField
from nerfstudio.model_components.ray_samplers import Sampler, UniformSampler
from nerfstudio.model_components.scene_colliders import AABBBoxCollider, SceneCollider
from nerfstudio.models.base_surface_model import SurfaceModel
from nerfstudio.utils.eval_utils import eval_setup
from ssdp.utils.jaxtyping import jaxtyped


@jaxtyped()
def _searchsorted(
    input: jt.Shaped[torch.Tensor, " *R S "],
    value: jt.Shaped[torch.Tensor, " *R "],
    **kwargs: Any,
) -> jt.Int[torch.Tensor, " *R "]:
    return torch.searchsorted(
        sorted_sequence=input,
        input=value.unsqueeze(-1),
        **kwargs,
    ).squeeze(-1)


@jaxtyped()
def _gather(
    input: jt.Shaped[torch.Tensor, " *R S "],
    index: jt.Int[torch.Tensor, " *R "],
    **kwargs: Any,
) -> jt.Shaped[torch.Tensor, " *R "]:
    return torch.gather(
        input=input,
        index=index.unsqueeze(-1),
        dim=-1,
        **kwargs,
    ).squeeze(-1)


@jaxtyped()
def _compute_mae(
    left_bins: jt.Float[torch.Tensor, " *R S "],
    right_bins: jt.Float[torch.Tensor, " *R S "],
    probabilities: jt.Float[torch.Tensor, " *R S "],
    observed_values: jt.Float[torch.Tensor, " *R "],
) -> jt.Float[torch.Tensor, " *R "]:
    values = (left_bins + right_bins) / 2.0
    expectations = torch.sum(probabilities * values, dim=-1)
    maes = nn.functional.l1_loss(
        input=expectations,
        target=observed_values,
        reduction="none",
    )
    return maes


@jaxtyped()
def _compute_nll(
    observed_left_bins: jt.Float[torch.Tensor, " *R "],
    observed_right_bins: jt.Float[torch.Tensor, " *R "],
    observed_probabilities: jt.Float[torch.Tensor, " *R "],
) -> jt.Float[torch.Tensor, " *R "]:
    nlls = -torch.log(torch.clamp(observed_probabilities, min=1.0e-6))
    nlls += torch.log(torch.clamp(observed_right_bins - observed_left_bins, min=1.0e-6))
    return nlls


@jaxtyped()
def _compute_crps(
    left_bins: jt.Float[torch.Tensor, " *R S "],
    right_bins: jt.Float[torch.Tensor, " *R S "],
    cdf_values: jt.Float[torch.Tensor, " *R S+1 "],
    observed_values: jt.Float[torch.Tensor, " *R "],
) -> jt.Float[torch.Tensor, " *R "]:
    boundaries = torch.cat([left_bins, right_bins[..., -1:]], dim=-1)
    step_values = observed_values.unsqueeze(-1) <= boundaries
    losses = nn.functional.mse_loss(
        input=cdf_values,
        target=step_values.to(cdf_values),
        reduction="none",
    )
    left_losses, right_losses = losses[..., :-1], losses[..., 1:]
    crpss = (left_losses + right_losses) * (right_bins - left_bins) / 2.0
    crpss = torch.sum(crpss, dim=-1)
    return crpss


@jaxtyped()
def _compute_ece(
    observed_cdf_values: jt.Float[torch.Tensor, " R "],
    num_calib_bins: int,
) -> jt.Float[torch.Tensor, " "]:
    observed_cdf_values = torch.sort(observed_cdf_values).values
    expected_probabilities = torch.linspace(
        start=0.0,
        end=1.0,
        steps=num_calib_bins + 1,
        dtype=observed_cdf_values.dtype,
        device=observed_cdf_values.device,
    )[1:-1]
    empirical_counts = torch.searchsorted(observed_cdf_values, expected_probabilities, right=True)
    empirical_probabilities = empirical_counts / len(observed_cdf_values)
    ece = nn.functional.l1_loss(empirical_probabilities, expected_probabilities)
    return ece


@jaxtyped()
def _compute_sharpness(
    left_bins: jt.Float[torch.Tensor, " *R S "],
    right_bins: jt.Float[torch.Tensor, " *R S "],
    probabilities: jt.Float[torch.Tensor, " *R S "],
) -> jt.Float[torch.Tensor, " *R "]:
    values = (left_bins + right_bins) / 2.0
    expectations = torch.sum(probabilities * values, dim=-1)
    values = (left_bins**2.0 + left_bins * right_bins + right_bins**2.0) / 3.0
    variances = torch.sum(probabilities * values, dim=-1)
    variances = variances - expectations**2.0
    sharpnesses = torch.sqrt(torch.clamp(variances, min=1.0e-6))
    return sharpnesses


@jaxtyped()
def _compute_brier_score(
    probabilities: jt.Float[torch.Tensor, " *R "],
    observed_labels: jt.Float[torch.Tensor, " *R "],
) -> jt.Float[torch.Tensor, " *R "]:
    brier_scores = nn.functional.mse_loss(
        input=probabilities,
        target=observed_labels,
        reduction="none",
    )
    return brier_scores


@dataclasses.dataclass
class UQMetricEvaluator:
    target_mesh_file: Path
    meta_file: Path
    config_file: Path
    output_file: Path
    ray_stride: int = 4
    ray_interval: float = 0.005
    num_calib_bins: int = 100
    ray_chunk_size: int = 1 << 12
    foreground_threshold: float = 0.5
    boundary_threshold: float = 50.0

    @jaxtyped()
    def _distance_transform(
        self,
        foreground_masks: jt.Float[torch.Tensor, " H W "],
    ) -> jt.Float[torch.Tensor, " H W "]:
        device = foreground_masks.device
        foreground_masks = foreground_masks.cpu().numpy()
        foreground_masks = foreground_masks > self.foreground_threshold
        inside_distances = scipy.ndimage.distance_transform_edt(foreground_masks)
        outside_distances = scipy.ndimage.distance_transform_edt(~foreground_masks)
        boundary_distances = np.maximum(inside_distances, outside_distances)
        boundary_distances = torch.as_tensor(boundary_distances, device=device)
        return boundary_distances

    @jaxtyped()
    @torch.no_grad()
    def _eval_single_metrics(
        self,
        field: SDFField,
        camera: Cameras,
        sampler: Sampler,
        collider: SceneCollider,
        target_mesh: trimesh.Trimesh,
        foreground_mask: jt.Float[torch.Tensor, " 1 H W "],
        norm_scale_factor: float = 1.0,
    ) -> jt.PyTree[jt.Float[torch.Tensor, " ?R "], " T "]:
        foreground_mask = foreground_mask.squeeze(0)
        boundary_distances = self._distance_transform(foreground_mask)
        non_boundary_mask = boundary_distances > self.boundary_threshold

        pixel_coords = camera.get_image_coords().to(camera.device)
        stride_mask = ~torch.any(pixel_coords.long() % self.ray_stride, dim=-1)

        pixel_mask = non_boundary_mask & stride_mask
        pixel_coords = pixel_coords[pixel_mask, ...]
        camera_foreground_mask = foreground_mask[pixel_mask, ...]

        camera_ray_bundle = camera.generate_rays(
            camera_indices=0,
            coords=pixel_coords,
        )

        intersector = RayMeshIntersector(target_mesh, scale_to_box=False)
        intersections, ray_indices, _ = intersector.intersects_location(
            ray_origins=camera_ray_bundle.origins.cpu().numpy(),
            ray_directions=camera_ray_bundle.directions.cpu().numpy(),
            multiple_hits=False,
        )
        intersections = camera_ray_bundle.origins.new_tensor(intersections)

        camera_intersections = intersections.new_zeros(*camera_ray_bundle.shape, 3)
        camera_intersections[ray_indices, ...] = intersections

        camera_observed_depths = (
            torch.linalg.vecdot(
                x=camera_intersections - camera_ray_bundle.origins,
                y=camera_ray_bundle.directions,
                dim=-1,
            )
            / norm_scale_factor
        )

        camera_intersection_mask = intersections.new_zeros(
            *camera_ray_bundle.shape,
            dtype=torch.bool,
        )
        camera_intersection_mask[ray_indices, ...] = True

        metrics = collections.defaultdict(list)

        for ray_index in tqdm.tqdm(
            iterable=range(0, len(camera_ray_bundle), self.ray_chunk_size),
            colour=colors.to_hex("dodgerblue"),
            desc="Evaluating single metrics...",
            leave=False,
        ):
            start_index = ray_index
            end_index = ray_index + self.ray_chunk_size
            ray_bundle = camera_ray_bundle[start_index:end_index, ...]
            observed_depths = camera_observed_depths[start_index:end_index, ...]
            foreground_mask = camera_foreground_mask[start_index:end_index, ...]
            intersection_mask = camera_intersection_mask[start_index:end_index, ...]

            ray_bundle = collider(ray_bundle)
            ray_samples: RaySamples = sampler(ray_bundle)

            field_outputs = field(ray_samples, return_alphas=True)

            probabilities = ray_samples.get_weights_and_transmittance_from_alphas(
                alphas=field_outputs[FieldHeadNames.ALPHA],
                weights_only=True,
            )
            probabilities = probabilities.squeeze(-1)

            left_bins = ray_samples.frustums.starts.squeeze(-1) / norm_scale_factor
            right_bins = ray_samples.frustums.ends.squeeze(-1) / norm_scale_factor
            boundaries = torch.cat([left_bins, right_bins[..., -1:]], dim=-1)
            observed_indices = _searchsorted(boundaries, observed_depths, right=True) - 1

            aabb_mask = (observed_indices >= 0) & (observed_indices < probabilities.shape[-1])
            intersection_mask &= aabb_mask & (foreground_mask > self.foreground_threshold)

            observed_labels = foreground_mask.to(probabilities)

            brier_scores = _compute_brier_score(
                probabilities=torch.sum(probabilities, dim=-1),
                observed_labels=observed_labels,
            )

            boundaries = boundaries[intersection_mask, ...]
            probabilities = probabilities[intersection_mask, ...]
            observed_depths = observed_depths[intersection_mask, ...]
            observed_indices = observed_indices[intersection_mask, ...]

            probabilities = torch.clamp(probabilities, min=1.0e-6)
            probabilities = nn.functional.normalize(probabilities, p=1, dim=-1)

            cdf_values = torch.cumsum(probabilities, dim=-1)
            cdf_values = nn.functional.pad(cdf_values, (1, 0), mode="constant", value=0.0)

            left_bins = boundaries[..., :-1]
            right_bins = boundaries[..., 1:]
            observed_left_bins = _gather(left_bins, observed_indices)
            observed_right_bins = _gather(right_bins, observed_indices)
            observed_probabilities = _gather(probabilities, observed_indices)
            observed_cdf_values = _gather(cdf_values, observed_indices)
            observed_cdf_values += (
                observed_probabilities
                * (observed_depths - observed_left_bins)
                / torch.clamp(observed_right_bins - observed_left_bins, min=1.0e-6)
            )

            maes = _compute_mae(
                left_bins=left_bins,
                right_bins=right_bins,
                probabilities=probabilities,
                observed_values=observed_depths,
            )
            nlls = _compute_nll(
                observed_left_bins=observed_left_bins,
                observed_right_bins=observed_right_bins,
                observed_probabilities=observed_probabilities,
            )
            crpss = _compute_crps(
                left_bins=left_bins,
                right_bins=right_bins,
                cdf_values=cdf_values,
                observed_values=observed_depths,
            )
            sharpnesses = _compute_sharpness(
                left_bins=left_bins,
                right_bins=right_bins,
                probabilities=probabilities,
            )

            metrics["mae"].append(maes.cpu())
            metrics["nll"].append(nlls.cpu())
            metrics["crps"].append(crpss.cpu())
            metrics["sharpness"].append(sharpnesses.cpu())
            metrics["brier_score"].append(brier_scores.cpu())
            metrics["observed_cdf_values"].append(observed_cdf_values.cpu())

        return metrics

    @jaxtyped()
    def _eval_average_metrics(
        self,
        field: SDFField,
        sampler: Sampler,
        collider: SceneCollider,
        dataloader: FixedIndicesEvalDataloader,
        target_mesh: trimesh.Trimesh,
        foreground_masks: jt.Float[torch.Tensor, " B 1 H W "],
        norm_scale_factor: float = 1.0,
    ) -> jt.PyTree[jt.Float[torch.Tensor, " "], " T "]:
        multi_metrics = collections.defaultdict(list)

        for (camera, _), foreground_mask in tqdm.tqdm(
            iterable=zip(dataloader, foreground_masks, strict=True),
            colour=colors.to_hex("dodgerblue"),
            desc="Evaluating average metrics...",
            total=len(dataloader),
        ):
            foreground_mask = foreground_mask.to(camera.device)
            single_metrics = self._eval_single_metrics(
                field=field,
                camera=camera,
                sampler=sampler,
                collider=collider,
                target_mesh=target_mesh,
                foreground_mask=foreground_mask,
                norm_scale_factor=norm_scale_factor,
            )
            for metric_name, metric_values in single_metrics.items():
                multi_metrics[metric_name].extend(metric_values)

        multi_metrics = dict(
            zip(
                multi_metrics.keys(),
                map(torch.cat, multi_metrics.values()),
                strict=True,
            )
        )
        ece = _compute_ece(
            observed_cdf_values=multi_metrics.pop("observed_cdf_values"),
            num_calib_bins=self.num_calib_bins,
        )
        average_metrics = dict(
            zip(
                multi_metrics.keys(),
                map(torch.mean, multi_metrics.values()),
                strict=True,
            )
        )
        average_metrics.update(ece=ece)

        return average_metrics

    def __call__(self) -> None:
        # NOTE: Since the camera poses are transformed by `SDFStudio` dataparser in nerfstudio,
        # the world coordinate system is also transformed accordingly.
        # https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/data/dataparsers/sdfstudio_dataparser.py#L113
        _, pipeline, _, _ = eval_setup(self.config_file)
        dataset = pipeline.datamanager.train_dataset
        transform_matrix: torch.Tensor | None = dataset.metadata.get("transform")
        if transform_matrix is not None:
            row_vector = transform_matrix.new_tensor([[0.0, 0.0, 0.0, 1.0]])
            transform_matrix = torch.cat([transform_matrix, row_vector], dim=0)

        with open(self.meta_file) as fp:
            meta_data = json.load(fp)

        unnorm_matrix = torch.as_tensor(meta_data["worldtogt"])
        norm_matrix: torch.Tensor = torch.linalg.inv(unnorm_matrix)
        if transform_matrix is not None:
            norm_matrix = transform_matrix @ norm_matrix

        foreground_files: list[str] = list(map(itemgetter("foreground_mask"), meta_data["frames"]))
        foreground_masks = torch.stack(
            [
                transforms.functional.to_dtype(
                    inpt=torchvision.io.decode_image(
                        input=self.meta_file.parent / foreground_file,
                        mode=torchvision.io.ImageReadMode.GRAY,
                    ),
                    dtype=torch.float32,
                    scale=True,
                )
                for foreground_file in foreground_files
            ],
            dim=0,
        )

        target_mesh = trimesh.load_mesh(self.target_mesh_file)
        target_mesh = target_mesh.apply_transform(norm_matrix.numpy())

        # NOTE: Do the same thing as what is done in `ns-eval`.
        # https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/scripts/eval.py#L50
        # https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/pipelines/base_pipeline.py#L420

        model: SurfaceModel = pipeline.model
        datamanager: VanillaDataManager = pipeline.datamanager

        field: SDFField = model.field
        dataloader = datamanager.fixed_indices_eval_dataloader

        scene_aabb = model.collider.scene_box.aabb
        object_aabb = scene_aabb.new_tensor(target_mesh.bounds)

        scene_aabb_min, scene_aabb_max = scene_aabb
        object_aabb_min, object_aabb_max = object_aabb

        aabb_min = torch.maximum(scene_aabb_min, object_aabb_min)
        aabb_max = torch.minimum(scene_aabb_max, object_aabb_max)

        assert torch.all(aabb_min < aabb_max), (
            "The object AABB does not intersect with the scene AABB."
        )

        aabb = torch.stack([aabb_min, aabb_max], dim=0)
        collider = AABBBoxCollider(SceneBox(aabb))

        norm_scale_factors = torch.linalg.norm(norm_matrix[:3, :3], dim=0)
        norm_scale_factor = torch.amin(norm_scale_factors).item()
        ray_interval = self.ray_interval * norm_scale_factor
        max_distance = collider.scene_box.get_diagonal_length()
        num_samples = (max_distance / ray_interval).ceil().long()

        sampler = UniformSampler(
            num_samples=num_samples,
            train_stratified=False,
        )

        metrics = self._eval_average_metrics(
            field=field,
            sampler=sampler,
            collider=collider,
            dataloader=dataloader,
            target_mesh=target_mesh,
            foreground_masks=foreground_masks,
            norm_scale_factor=norm_scale_factor,
        )
        metrics = dict(
            zip(
                metrics.keys(),
                map(float, metrics.values()),
                strict=True,
            )
        )

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as fp:
            json.dump(metrics, fp, indent=4)

        loguru.logger.success(f"Saved the evaluation metrics to <{self.output_file}>.")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        UQMetricEvaluator,
        config=(tyro.conf.AvoidSubcommands,),
    )()
