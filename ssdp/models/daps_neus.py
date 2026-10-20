import dataclasses

from nerfstudio.fields.sdf_field import SDFFieldConfig
from ssdp.fields.ldm_mixin import LDMMixinConfig

from .daps_mixin import DAPSMixin, DAPSMixinConfig
from .neus import NeuS, NeuSConfig


@dataclasses.dataclass
class DAPSNeuSConfig(DAPSMixinConfig, NeuSConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: DAPSNeuS,
    )
    sdf_field: SDFFieldConfig = dataclasses.field(
        default_factory=LDMMixinConfig,
    )


class DAPSNeuS(DAPSMixin, NeuS):
    config: DAPSNeuSConfig
