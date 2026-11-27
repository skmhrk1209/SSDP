import dataclasses
import enum
from typing import Any, override

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudio, SDFStudioDataParserConfig
from nerfstudio.utils.io import load_from_json


class Split(enum.StrEnum):
    TRAIN = enum.auto()
    TEST = enum.auto()
    VAL = enum.auto()


@dataclasses.dataclass
class CustomSDFStudioDataParserConfig(SDFStudioDataParserConfig):
    _target: type = dataclasses.field(default_factory=lambda: CustomSDFStudioDataParser)


@dataclasses.dataclass
class CustomSDFStudioDataParser(SDFStudio):
    config: CustomSDFStudioDataParserConfig

    @override
    def _generate_dataparser_outputs(self, split: Split = Split.TRAIN) -> DataparserOutputs:
        dataparser_outputs = super()._generate_dataparser_outputs(split)

        if not self.config.include_foreground_mask:
            return dataparser_outputs

        meta_data: dict[str, Any] = load_from_json(self.config.data / "meta_data.json")
        assert meta_data["has_foreground_mask"], f"No foreground masks in <{self.config.data}>."

        frame_indices = list(range(len(meta_data["frames"])))
        if split == Split.VAL and self.config.skip_every_for_val_split >= 1:
            frame_indices = frame_indices[:: self.config.skip_every_for_val_split]

        frames: list[dict[str, Any]] = meta_data["frames"]

        mask_filenames = []

        for frame_index, frame in enumerate(frames):
            if frame_index not in frame_indices:
                continue

            if mask_filename := frame.get("foreground_mask", ""):
                mask_filenames.append(self.config.data / mask_filename)

        assert len(mask_filenames) == len(dataparser_outputs.image_filenames)

        dataparser_outputs = dataclasses.replace(
            dataparser_outputs,
            mask_filenames=mask_filenames,
        )

        return dataparser_outputs
