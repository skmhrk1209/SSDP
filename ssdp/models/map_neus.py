import dataclasses

from nerfstudio.fields.sdf_field import SDFFieldConfig
from ssdp.fields.ldm_mixin import LDMMixinConfig

from .map_mixin import MAPMixin, MAPMixinConfig
from .neus import NeuS, NeuSConfig


@dataclasses.dataclass
class MAPNeuSConfig(MAPMixinConfig, NeuSConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: MAPNeuS,
    )
    sdf_field: SDFFieldConfig = dataclasses.field(
        default_factory=LDMMixinConfig,
    )


class MAPNeuS(MAPMixin, NeuS):
    config: MAPNeuSConfig
