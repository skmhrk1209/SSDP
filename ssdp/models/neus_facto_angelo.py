import dataclasses

from .neuralangelo_mixin import NeuralangeloMixin, NeuralangeloMixinConfig
from .neus_facto import NeuSFacto, NeuSFactoConfig


@dataclasses.dataclass
class NeuSFactoAngeloConfig(NeuralangeloMixinConfig, NeuSFactoConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: NeuSFactoAngelo,
    )


class NeuSFactoAngelo(NeuralangeloMixin, NeuSFacto):
    pass
