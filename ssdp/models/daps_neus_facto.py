import dataclasses

from nerfstudio.fields.sdf_field import SDFFieldConfig
from ssdp.fields.ldm_mixin import LDMMixinConfig

from .daps_mixin import DAPSMixin, DAPSMixinConfig
from .neus_facto import NeuSFacto, NeuSFactoConfig


@dataclasses.dataclass
class DAPSNeuSFactoConfig(DAPSMixinConfig, NeuSFactoConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: DAPSNeuSFacto,
    )
    sdf_field: SDFFieldConfig = dataclasses.field(
        default_factory=LDMMixinConfig,
    )


class DAPSNeuSFacto(DAPSMixin, NeuSFacto):
    config: DAPSNeuSFactoConfig
