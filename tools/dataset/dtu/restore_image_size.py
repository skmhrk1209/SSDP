import dataclasses
import functools
import json
import operator
from pathlib import Path
from typing import Any

import jaxtyping as jt
import loguru
import numpy as np
import skimage
import tqdm
import tyro
from matplotlib import colors

from ssdp.utils import git
from ssdp.utils.jaxtyping import jaxtyped


@jaxtyped()
def _get_crop_matrix(crop_offset: tuple[int, int]) -> jt.Float[np.ndarray, " 4 4 "]:
    cy, cx = crop_offset
    crop_matrix = np.asarray(
        [
            [1.0, 0.0, -cx, 0.0],
            [0.0, 1.0, -cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return crop_matrix


@jaxtyped()
def _get_scale_matrix(scale_factor: tuple[float, float]) -> jt.Float[np.ndarray, " 4 4 "]:
    sy, sx = scale_factor
    scale_matrix = np.diag([sx, sy, 1.0, 1.0])
    return scale_matrix


@dataclasses.dataclass
class ImageSizeRestorer:
    root_dir: Path
    verbose: bool = False
    dry_run: bool = False

    @jaxtyped()
    def _get_transform_matrix(
        self,
        input_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> jt.Float[np.ndarray, " 4 4 "]:
        crop_size = min(input_size)
        size_diff = operator.sub(*input_size)
        crop_offset = abs(size_diff) // 2
        crop_offset = (crop_offset, 0) if size_diff > 0 else (0, crop_offset)
        crop_matrix = _get_crop_matrix(crop_offset)
        scale_factor = tuple(np.divide(output_size, crop_size))
        scale_matrix = _get_scale_matrix(scale_factor)
        transform_matrix = scale_matrix @ crop_matrix
        return transform_matrix

    def _restore_image_sizes(
        self,
        input_scene_dir: Path,
        output_meta_file: Path | None = None,
    ) -> None:
        input_meta_file = input_scene_dir / "meta_data.json"
        backed_meta_file = input_scene_dir / ".meta_data.json"
        output_meta_file = output_meta_file or input_meta_file

        if backed_meta_file.exists():
            loguru.logger.debug(
                f"Found the backed up meta data file <{backed_meta_file}>. "
                f"Loading the meta data from it."
            )
            input_meta_file = backed_meta_file

        with open(input_meta_file) as fp:
            input_meta_data = json.load(fp)

        input_width: int = input_meta_data["width"]
        input_height: int = input_meta_data["height"]

        image_dir = input_scene_dir / "image"
        image_file = next(image_dir.glob("*.png"))

        image = skimage.io.imread(image_file)
        output_height, output_width, _ = image.shape

        input_frames: list[dict[str, Any]] = input_meta_data["frames"]
        output_frames = []

        for input_frame in input_frames:
            frame_id = int(input_frame["rgb_path"].split("_")[0])
            image_file = f"image/{frame_id:06d}.png"
            mask_file = f"mask/{frame_id:03d}.png"

            intrinsic_matrix = np.asarray(input_frame["intrinsics"])

            # NOTE: Verify intrinsic matrix conversion is the same as SDFStudio's conversion.
            # https://github.com/autonomousvision/sdfstudio/blob/master/nerfstudio/data/dataparsers/sdfstudio_dataparser.py#L228
            _intrinsic_matrix = intrinsic_matrix.copy()
            _intrinsic_matrix[..., :2, :] *= 1200.0 / 384.0
            _intrinsic_matrix[..., 0, 2] += 200.0

            transform_matrix = self._get_transform_matrix(
                input_size=(output_height, output_width),
                output_size=(input_height, input_width),
            )
            intrinsic_matrix = np.linalg.inv(transform_matrix) @ intrinsic_matrix

            assert np.allclose(intrinsic_matrix, _intrinsic_matrix)

            output_frame = dict(
                input_frame,
                rgb_path=image_file,
                foreground_mask=mask_file,
                intrinsics=intrinsic_matrix.tolist(),
            )
            output_frame.pop("mono_depth_path")
            output_frame.pop("mono_normal_path")
            output_frames.append(output_frame)

        output_meta_data = dict(
            input_meta_data,
            width=output_width,
            height=output_height,
            has_mono_prior=False,
            frames=output_frames,
            args=dataclasses.asdict(self),
            git=dict(
                branch=git.get_branch(),
                commit_id=git.get_commit_id(),
                remote_url=git.get_remote_url(),
            ),
        )

        if output_meta_file == input_meta_file:
            loguru.logger.debug(f"Overwriting the original meta data file <{input_meta_file}>.")
            if not backed_meta_file.exists():
                loguru.logger.debug(
                    f"The original meta data file will be backed up to <{backed_meta_file}>."
                )
                if not self.dry_run:
                    input_meta_file.rename(backed_meta_file)

        if not self.dry_run:
            output_meta_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_meta_file, "w") as fp:
                json.dump(output_meta_data, fp, indent=4, default=str)

        loguru.logger.debug(f"Saved the modified meta data to <{output_meta_file}>.")

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
                self._restore_image_sizes(scene_dir)

        loguru.logger.success("Finished!")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        ImageSizeRestorer,
        config=(tyro.conf.AvoidSubcommands,),
    )()
