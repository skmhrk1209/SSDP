import collections
import dataclasses
import itertools
from typing import Any

import toolz

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.base_surface_model import SurfaceModelConfig
from nerfstudio.plugins.types import MethodSpecification
from ssdp.dataparsers import CustomSDFStudioDataParserConfig
from ssdp.datasets import CustomSDFDataset
from ssdp.fields.unis_field import UNISKernelType

from . import _fields, _models  # noqa: F401
from ._base import create_trainer_config
from ._registory import FIELD_REGISTORY, MODEL_REGISTORY, OPTIMIZERS_REGISTORY
from ._utils import asdict


@dataclasses.dataclass
class _ModelEntry:
    name: str
    display_name: str
    base_model_config: SurfaceModelConfig
    model_config_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
    field_config_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _FieldEntry:
    name: str
    display_name: str
    model_config_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
    field_config_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)


def create_method_config(
    model_entry: _ModelEntry,
    field_entry: _FieldEntry,
) -> None:
    model_config_factory = MODEL_REGISTORY[model_entry.name]
    field_config_factory = FIELD_REGISTORY[field_entry.name]
    optimizers_config_factory = toolz.compose(
        OPTIMIZERS_REGISTORY[model_entry.name],
        OPTIMIZERS_REGISTORY[field_entry.name],
    )

    base_trainer_config = create_trainer_config(_BASE_TRAINER_CONFIG)
    base_optimizers_config = base_trainer_config.optimizers

    base_model_config = model_entry.base_model_config
    base_field_config = base_model_config.sdf_field

    model_config = model_config_factory(base_model_config)
    field_config = field_config_factory(base_field_config)
    optimizers_config = optimizers_config_factory(base_optimizers_config)

    assert isinstance(base_trainer_config.pipeline.datamanager, VanillaDataManagerConfig)

    method_name = model_entry.display_name.format(field_name=field_entry.display_name)
    trainer_config = dataclasses.replace(
        base_trainer_config,
        method_name=method_name.lower(),
        pipeline=dataclasses.replace(
            base_trainer_config.pipeline,
            datamanager=dataclasses.replace(
                base_trainer_config.pipeline.datamanager,
                _target=VanillaDataManager[CustomSDFDataset],
                dataparser=CustomSDFStudioDataParserConfig(
                    **dict(
                        asdict(base_trainer_config.pipeline.datamanager.dataparser),
                        include_foreground_mask=True,
                    ),
                ),
                pixel_sampler=dataclasses.replace(
                    base_trainer_config.pipeline.datamanager.pixel_sampler,
                    ignore_mask=True,
                ),
            ),
            model=dataclasses.replace(
                model_config,
                sdf_field=dataclasses.replace(
                    field_config,
                    **model_entry.field_config_overrides,
                    **field_entry.field_config_overrides,
                ),
                **model_entry.model_config_overrides,
                **field_entry.model_config_overrides,
            ),
        ),
        optimizers=optimizers_config,
    )

    method_config = MethodSpecification(
        config=trainer_config,
        description=f"{model_config._target} x {field_config._target}",
    )

    globals()[method_name.lower().replace("-", "_")] = method_config


def _get_trainer_config(name: str) -> TrainerConfig:
    trainer_config = method_configs[name]
    assert isinstance(trainer_config, TrainerConfig)
    return trainer_config


def _get_model_config(name: str) -> SurfaceModelConfig:
    trainer_config = _get_trainer_config(name)
    model_config = trainer_config.pipeline.model
    assert isinstance(model_config, SurfaceModelConfig)
    return model_config


_BASE_TRAINER_CONFIG = _get_trainer_config("neus-facto")

_MODEL_ENTRIES = [
    _ModelEntry(
        name="neus_model",
        display_name="{field_name}",
        base_model_config=_get_model_config("neus"),
    ),
    _ModelEntry(
        name="neus_facto_model",
        display_name="{field_name}-Facto",
        base_model_config=_get_model_config("neus-facto"),
    ),
]

_FIELD_ETNRIES = [
    _FieldEntry(
        name="neus_field",
        display_name="NeuS",
        model_config_overrides=dict(
            color_nll_loss_mult=1.0 / 3.0,
            color_nlml_loss_mult=0.0,
            eikonal_loss_mult=0.1 * 2.0,
        ),
    ),
    _FieldEntry(
        name="oav_field",
        display_name="OaV",
        model_config_overrides=dict(
            color_nll_loss_mult=1.0 / 3.0,
            color_nlml_loss_mult=0.0,
            eikonal_loss_mult=0.1 * 2.0,
        ),
    ),
    *(
        _FieldEntry(
            name="unis_field",
            display_name=f"UNIS-C{index}",
            model_config_overrides=dict(
                color_nll_loss_mult=1.0 / 3.0,
                color_nlml_loss_mult=0.0,
                eikonal_loss_mult=0.1 * 2.0,
            ),
            field_config_overrides=dict(
                kernel_type=kernel_type,
            ),
        )
        for index, kernel_type in enumerate(UNISKernelType)
    ),
    _FieldEntry(
        name="ssdp_field",
        display_name="SSDP",
        model_config_overrides=dict(
            color_nll_loss_mult=0.0,
            color_nlml_loss_mult=1.0,
            eikonal_loss_mult=0.1 * 2.0,
        ),
    ),
]

collections.deque(
    itertools.starmap(
        create_method_config,
        itertools.product(_MODEL_ENTRIES, _FIELD_ETNRIES),
    ),
    maxlen=0,
)
