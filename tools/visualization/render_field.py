import dataclasses
from pathlib import Path

import loguru
import torchvision
import tqdm
import tyro
from matplotlib import colors

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.eval_utils import eval_setup


@dataclasses.dataclass
class FieldRenderer:
    config_file: Path
    output_dir: Path
    output_keys: tuple[str, ...] = (
        "surface_colors",
        "surface_depths",
        "surface_normals",
    )
    ray_chunk_size: int = 1 << 12

    def __call__(self) -> None:
        _, pipeline, _, _ = eval_setup(
            config_path=self.config_file,
            eval_num_rays_per_chunk=self.ray_chunk_size,
        )
        datamanager: VanillaDataManager = pipeline.datamanager
        dataloader = datamanager.fixed_indices_eval_dataloader
        dataset: InputDataset = dataloader.dataset

        for camera, inputs in tqdm.tqdm(
            iterable=dataloader,
            colour=colors.to_hex("dodgerblue"),
            desc="Rendering the field...",
        ):
            outputs = pipeline.model.get_outputs_for_camera(camera)
            _, images = pipeline.model.get_image_metrics_and_images(outputs, inputs)
            image_index: int = inputs["image_idx"]
            image_file = dataset.image_filenames[image_index]
            for output_key in self.output_keys:
                image = images[output_key]
                image = image.permute(2, 0, 1)
                output_file = self.output_dir / output_key / image_file.name
                output_file.parent.mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(image, output_file)

        loguru.logger.success("Finished!")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        FieldRenderer,
        config=(tyro.conf.AvoidSubcommands,),
    )()
