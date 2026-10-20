import dataclasses

from .ldm_mixin import LDMMixin, LDMMixinConfig
from .ssdp import SSDP, SSDPConfig


@dataclasses.dataclass
class LDMSSDPConfig(LDMMixinConfig, SSDPConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: LDMSSDP,
    )


class LDMSSDP(LDMMixin, SSDP):
    config: LDMSSDPConfig
