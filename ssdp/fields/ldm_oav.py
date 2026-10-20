import dataclasses

from .ldm_mixin import LDMMixin, LDMMixinConfig
from .oav import OaV, OaVConfig


@dataclasses.dataclass
class LDMOaVConfig(LDMMixinConfig, OaVConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: LDMOaV,
    )


class LDMOaV(LDMMixin, OaV):
    config: LDMOaVConfig
