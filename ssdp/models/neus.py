import dataclasses

from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from ssdp.fields import SDFConfig

from .surface_model_mixin import SurfaceModelMixin, SurfaceModelMixinConfig


@dataclasses.dataclass
class NeuSConfig(SurfaceModelMixinConfig, NeuSModelConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: NeuS,
    )
    sdf_field: SDFFieldConfig = dataclasses.field(
        default_factory=SDFConfig,
    )


class NeuS(SurfaceModelMixin, NeuSModel):
    pass
