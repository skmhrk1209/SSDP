import dataclasses

from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.models.neus_facto import NeuSFactoModel, NeuSFactoModelConfig
from ssdp.fields import SDFConfig

from .surface_model_mixin import SurfaceModelMixin, SurfaceModelMixinConfig


@dataclasses.dataclass
class NeuSFactoConfig(SurfaceModelMixinConfig, NeuSFactoModelConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: NeuSFacto,
    )
    sdf_field: SDFFieldConfig = dataclasses.field(
        default_factory=SDFConfig,
    )


class NeuSFacto(SurfaceModelMixin, NeuSFactoModel):
    pass
