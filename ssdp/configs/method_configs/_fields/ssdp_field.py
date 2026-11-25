import dataclasses

from nerfstudio.fields.base_field import FieldConfig
from ssdp.fields import SSDPFieldConfig

from ....utils.types import OptimizersConfig
from .._registory import FIELD_REGISTORY, OPTIMIZERS_REGISTORY
from .._utils import asdict
from ._base import override_field_config as override_base_field_config
from ._base import override_optimizers_config as override_base_optimizers_config


@FIELD_REGISTORY.register()
def create_field_config(base_field_config: FieldConfig) -> SSDPFieldConfig:
    field_config = SSDPFieldConfig(**asdict(base_field_config))
    field_config = override_base_field_config(field_config)
    field_config = dataclasses.replace(
        field_config,
        num_ou_layers=field_config.num_layers_color,
    )
    return field_config


@OPTIMIZERS_REGISTORY.register()
def create_optimizers_config(base_optimizers_config: OptimizersConfig) -> OptimizersConfig:
    base_optimizers_config = override_base_optimizers_config(base_optimizers_config)
    return base_optimizers_config
