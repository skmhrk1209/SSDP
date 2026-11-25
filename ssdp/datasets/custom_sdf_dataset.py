import enum
from pathlib import Path
from typing import Any, override

import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms

from nerfstudio.data.datasets.sdf_dataset import SDFDataset


class ImageDType(enum.StrEnum):
    UINT8 = enum.auto()
    FLOAT32 = enum.auto()


class CustomSDFDataset(SDFDataset):
    def _get_mask(self, filename: Path) -> torch.Tensor:
        mask = torchvision.io.read_image(
            path=str(filename),
            mode=torchvision.io.ImageReadMode.GRAY,
        )
        mask = transforms.functional.to_dtype_image(mask, torch.bool)
        mask = transforms.functional.resize(
            inpt=mask,
            size=np.multiply(mask.shape[-2:], self.scale_factor).astype(int).tolist(),
            interpolation=transforms.InterpolationMode.NEAREST_EXACT,
        )
        mask = mask.movedim(0, -1)
        return mask

    @override
    def get_data(
        self, image_index: int, image_dtype: ImageDType = ImageDType.FLOAT32
    ) -> dict[str, Any]:
        match image_dtype:
            case ImageDType.FLOAT32:
                image = self.get_image_float32(image_index)
            case ImageDType.UINT8:
                image = self.get_image_uint8(image_index)

        data = dict(image_idx=image_index, image=image)

        if self._dataparser_outputs.mask_filenames is not None:
            mask_filename = self._dataparser_outputs.mask_filenames[image_index]
            mask = self._get_mask(mask_filename)
            assert image.shape[:-1] == mask.shape[:-1]
            data.update(mask=mask)
            if self.mask_color:
                image = torch.where(mask, image, image.new_tensor(self.mask_color))
                data.update(image=image)

        metadata = self.get_metadata(data)
        data.update(metadata)

        return data
