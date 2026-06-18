import dataclasses
import json
from operator import itemgetter
from pathlib import Path

import jaxtyping as jt
import kornia
import loguru
import numpy as np
import scipy.ndimage
import skimage
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as transforms
import trimesh
import tyro

from nerfstudio.utils.eval_utils import eval_setup
from ssdp.utils.jaxtyping import jaxtyped


@dataclasses.dataclass
class MeshPostprocessor:
    mesh_file: Path
    meta_file: Path
    config_file: Path | None = None
    output_file: Path | None = None
    input_normalized_mesh: bool = True
    output_normalized_mesh: bool = False
    use_foreground_masks: bool = False
    foreground_threshold: float = 0.5
    fill_foreground_holes: bool = True
    dilation_kernel_radius: float = 50.0
    keep_largest_component: bool = False
    device: torch.device | str | int = "cuda"

    @jaxtyped()
    def _fill_holes(
        self,
        foreground_masks: jt.Float[torch.Tensor, " H W "],
    ) -> jt.Float[torch.Tensor, " H W "]:
        dtype = foreground_masks.dtype
        device = foreground_masks.device
        foreground_masks = foreground_masks.cpu().numpy()
        foreground_masks = foreground_masks > self.foreground_threshold
        foreground_masks = scipy.ndimage.binary_fill_holes(foreground_masks)
        foreground_masks = torch.as_tensor(foreground_masks, dtype=dtype, device=device)
        return foreground_masks

    @jaxtyped()
    def _mask_vertices(
        self,
        vertices: jt.Float[torch.Tensor, " N 3 "],
        foreground_masks: jt.Float[torch.Tensor, " B 1 H W "],
        extrinsic_matrices: jt.Float[torch.Tensor, " B 4 4 "],
        intrinsic_matrices: jt.Float[torch.Tensor, " B 4 4 "],
        dilation_chunk_size: int = 1,
    ) -> jt.Bool[torch.Tensor, " N "]:
        if self.fill_foreground_holes:
            foreground_masks = foreground_masks.squeeze(1)
            foreground_masks = torch.stack(list(map(self._fill_holes, foreground_masks)), dim=0)
            foreground_masks = foreground_masks.unsqueeze(1)

        if self.dilation_kernel_radius:
            dilation_kernel = skimage.morphology.disk(self.dilation_kernel_radius)
            dilation_kernel = foreground_masks.new_tensor(dilation_kernel)
            dilation_function = torch.vmap(
                func=kornia.morphology.dilation,
                chunk_size=dilation_chunk_size,
            )
            foreground_masks = foreground_masks.unsqueeze(1)
            foreground_masks = dilation_function(foreground_masks, kernel=dilation_kernel)
            foreground_masks = foreground_masks.squeeze(1)

        foreground_masks = nn.functional.pad(
            foreground_masks,
            pad=(1, 1, 1, 1),
            mode="constant",
            value=-1.0,
        )

        padding_matrix = intrinsic_matrices.new_tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        intrinsic_matrices = padding_matrix @ intrinsic_matrices
        projection_matrices = intrinsic_matrices @ extrinsic_matrices

        vertices = nn.functional.pad(vertices, (0, 1), mode="constant", value=1.0)
        vertices = torch.einsum("...ij,nj->...ni", projection_matrices, vertices)

        while vertices.shape[-1] > 2:
            vertices = vertices[..., :-1] / (depths := vertices[..., -1:])

        image_size = foreground_masks.new_tensor(foreground_masks.shape[-2:])
        vertices = vertices / (image_size.flip(-1) - 1) * 2.0 - 1.0

        foreground_masks = nn.functional.grid_sample(
            input=foreground_masks,
            grid=vertices.unsqueeze(-2),
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        )
        foreground_masks = foreground_masks.squeeze(-1).squeeze(1)

        foreground_mask = torch.all(torch.abs(foreground_masks) > self.foreground_threshold, dim=0)
        in_frustum_mask = torch.any((foreground_masks >= 0.0) & (depths.squeeze(-1) > 0.0), dim=0)
        vertex_mask = foreground_mask & in_frustum_mask

        return vertex_mask

    def __call__(self) -> None:
        # NOTE: Since the camera poses are transformed by `SDFStudio` dataparser in nerfstudio,
        # the world coordinate system is also transformed accordingly.
        # https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/data/dataparsers/sdfstudio_dataparser.py#L113
        transform_matrix = None
        if self.config_file:
            _, pipeline, _, _ = eval_setup(self.config_file)
            dataset = pipeline.datamanager.train_dataset
            transform_matrix: torch.Tensor | None = dataset.metadata.get("transform")
            if transform_matrix is not None:
                row_vector = transform_matrix.new_tensor([[0.0, 0.0, 0.0, 1.0]])
                transform_matrix = torch.cat([transform_matrix, row_vector], dim=0)

        with open(self.meta_file) as fp:
            meta_data = json.load(fp)

        width: int = meta_data["width"]
        height: int = meta_data["height"]

        unnorm_matrix = torch.as_tensor(meta_data["worldtogt"])
        norm_matrix: torch.Tensor = torch.linalg.inv(unnorm_matrix)

        pose_matrices = torch.as_tensor(list(map(itemgetter("camtoworld"), meta_data["frames"])))

        extrinsic_matrices: torch.Tensor = torch.linalg.inv(pose_matrices)
        intrinsic_matrices = torch.as_tensor(
            list(map(itemgetter("intrinsics"), meta_data["frames"]))
        )

        if self.use_foreground_masks:
            foreground_files: list[str] = list(
                map(itemgetter("foreground_mask"), meta_data["frames"])
            )
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
        else:
            foreground_masks = pose_matrices.new_ones(len(pose_matrices), 1, height, width)

        mesh = trimesh.load_mesh(self.mesh_file)

        if not self.input_normalized_mesh:
            mesh = mesh.apply_transform(norm_matrix.numpy())
        elif transform_matrix is not None:
            untransform_matrix: torch.Tensor = torch.linalg.inv(transform_matrix)
            mesh = mesh.apply_transform(untransform_matrix.numpy())

        foreground_masks = foreground_masks.to(self.device)
        extrinsic_matrices = extrinsic_matrices.to(self.device)
        intrinsic_matrices = intrinsic_matrices.to(self.device)

        mesh_vertices = foreground_masks.new_tensor(mesh.vertices)
        vert_mask = self._mask_vertices(
            vertices=mesh_vertices,
            foreground_masks=foreground_masks,
            extrinsic_matrices=extrinsic_matrices,
            intrinsic_matrices=intrinsic_matrices,
        )

        vert_mask = vert_mask.cpu().numpy()
        face_mask = np.all(vert_mask[mesh.faces], axis=-1)

        mesh.update_faces(face_mask)
        mesh.remove_unreferenced_vertices()

        if self.keep_largest_component:
            meshes = mesh.split(only_watertight=False)
            mesh = max(meshes, key=lambda mesh: mesh.area)

        if not self.output_normalized_mesh:
            mesh = mesh.apply_transform(unnorm_matrix.numpy())

        output_file = self.output_file or self.mesh_file.with_stem(
            f"{self.mesh_file.stem}_postprocessed"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(output_file)

        loguru.logger.success(f"Saved the postprocessed mesh: <{output_file}>.")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        MeshPostprocessor,
        config=(tyro.conf.AvoidSubcommands,),
    )()
