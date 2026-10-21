import dataclasses

from nerfstudio.fields.sdf_field import SDFFieldConfig
from ssdp.fields.ldm_mixin import LDMMixinConfig

from .map_mixin import MAPMixin, MAPMixinConfig
from .neus_facto import NeuSFacto, NeuSFactoConfig


@dataclasses.dataclass
class MAPNeuSFactoConfig(MAPMixinConfig, NeuSFactoConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: MAPNeuSFacto,
    )
    sdf_field: SDFFieldConfig = dataclasses.field(
        default_factory=LDMMixinConfig,
    )


class MAPNeuSFacto(MAPMixin, NeuSFacto):
    config: MAPNeuSFactoConfig
