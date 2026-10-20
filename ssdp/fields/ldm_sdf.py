import dataclasses

from .ldm_mixin import LDMMixin, LDMMixinConfig
from .sdf import SDF, SDFConfig


@dataclasses.dataclass
class LDMSDFConfig(LDMMixinConfig, SDFConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: LDMSDF,
    )


class LDMSDF(LDMMixin, SDF):
    config: LDMSDFConfig
