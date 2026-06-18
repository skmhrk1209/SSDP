import dataclasses
import functools
import json
from pathlib import Path

import cv2 as cv
import jaxtyping as jt
import loguru
import numpy as np
import skimage
import tqdm
import trimesh
import tyro
from matplotlib import colors

from ssdp.utils import git
from ssdp.utils.jaxtyping import jaxtyped


@jaxtyped()
def _expand_matrix_3x3_to_4x4(
    matrices_3x3: jt.Shaped[np.ndarray, " *B 3 3 "],
) -> jt.Shaped[np.ndarray, " *B 4 4 "]:
    matrix_4x4 = np.eye(4)
    matrices_4x4 = np.broadcast_to(
        array=matrix_4x4,
        shape=(*matrices_3x3.shape[:-2], *matrix_4x4.shape[-2:]),
    )
    matrices_4x4 = matrices_4x4.copy()
    matrices_4x4[..., :3, :3] = matrices_3x3
    return matrices_4x4


@dataclasses.dataclass
class MetaDataCreator:
    root_dir: Path
    verbose: bool = False
    dry_run: bool = False
    renormalize_scene: bool = True
    normalized_aabb_extent: float = 1.0
    orthogonalize_extrinsics: bool = True
    normalize_intrinsics: bool = True

    def _create_meta_data(
        self,
        input_scene_dir: Path,
        output_meta_file: Path | None = None,
    ) -> None:
        image_dir = input_scene_dir / "image"
        image_files = sorted(image_dir.glob("*.jpg"))
        image_file = next(iter(image_files))

        image = skimage.io.imread(image_file)
        height, width, _ = image.shape

        if self.renormalize_scene:
            mesh_file = input_scene_dir / "mesh" / "gt_mesh.ply"
            mesh = trimesh.load_mesh(mesh_file)
            norm_matrix, extents = trimesh.bounds.oriented_bounds(mesh)
            scale_factor = self.normalized_aabb_extent / np.max(extents)
            scale_matrix = np.diag([*[scale_factor] * 3, 1.0])
            norm_matrix = scale_matrix @ norm_matrix
        else:
            camera_data = np.load(input_scene_dir / "cameras.npz")
            norm_matrix: np.ndarray = camera_data["scale_mat_0"]

        unnorm_matrix = np.linalg.inv(norm_matrix)

        output_frames = []

        for image_file in image_files:
            frame_id = image_file.stem
            image_file = f"image/{frame_id}.jpg"
            mask_file = f"mask/{frame_id}.png"

            pose_file = input_scene_dir / "pose" / f"{frame_id}.txt"
            pose_matrix = np.loadtxt(pose_file)
            pose_matrix = norm_matrix @ pose_matrix

            intrinsic_file = input_scene_dir / "intrinsic" / f"{frame_id}.txt"
            intrinsic_matrix = np.loadtxt(intrinsic_file)
            intrinsic_matrix = _expand_matrix_3x3_to_4x4(intrinsic_matrix)

            depth_scale_factor = 1.0
            if self.orthogonalize_extrinsics:
                extrinsic_matrix = np.linalg.inv(pose_matrix)
                projection_matrix = intrinsic_matrix @ extrinsic_matrix
                (
                    intrinsic_matrix,
                    rotation_matrix,
                    translation_vector,
                    *_,
                ) = cv.decomposeProjectionMatrix(projection_matrix[..., :-1, :])
                translation_vector = translation_vector.squeeze(-1)
                translation_vector = translation_vector[..., :-1] / translation_vector[..., -1:]
                pose_matrix = _expand_matrix_3x3_to_4x4(rotation_matrix.T)
                pose_matrix[..., :-1, -1] = translation_vector
                if self.normalize_intrinsics:
                    depth_scale_factor = intrinsic_matrix[..., -1, -1].item()
                    intrinsic_matrix /= depth_scale_factor
                intrinsic_matrix = _expand_matrix_3x3_to_4x4(intrinsic_matrix)

            output_frame = dict(
                rgb_path=image_file,
                foreground_mask=mask_file,
                camtoworld=pose_matrix.tolist(),
                intrinsics=intrinsic_matrix.tolist(),
                depth_scale_factor=depth_scale_factor,
            )
            output_frames.append(output_frame)

        output_meta_data = dict(
            camera_model="OPENCV",
            width=width,
            height=height,
            has_mono_prior=False,
            has_foreground_mask=True,
            has_sparse_sfm_points=False,
            worldtogt=unnorm_matrix.tolist(),
            scene_box=dict(aabb=[[-1.0] * 3, [1.0] * 3]),
            frames=output_frames,
            args=dataclasses.asdict(self),
            git=dict(
                branch=git.get_branch(),
                commit_id=git.get_commit_id(),
                remote_url=git.get_remote_url(),
            ),
        )

        output_meta_file = output_meta_file or input_scene_dir / "meta_data.json"

        if not self.dry_run:
            output_meta_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_meta_file, "w") as fp:
                json.dump(output_meta_data, fp, indent=4, default=str)

        loguru.logger.debug(f"Saved the meta data to <{output_meta_file}>.")

    def __call__(self) -> None:
        loguru.logger.remove()
        loguru.logger.add(
            sink=functools.partial(tqdm.tqdm.write, end=""),
            level="DEBUG" if self.verbose else "INFO",
            colorize=True,
        )

        for scene_dir in tqdm.tqdm(
            iterable=list(self.root_dir.iterdir()),
            colour=colors.to_hex("dodgerblue"),
            desc="Processing scenes...",
        ):
            if scene_dir.is_dir():
                self._create_meta_data(scene_dir)

        loguru.logger.success("Finished!")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        MetaDataCreator,
        config=(tyro.conf.AvoidSubcommands,),
    )()
