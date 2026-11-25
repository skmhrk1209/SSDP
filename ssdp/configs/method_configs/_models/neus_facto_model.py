import dataclasses

from nerfstudio.models.base_model import ModelConfig
from ssdp.models import NeuSModelConfig
from ssdp.models.neus_model import SamplerType

from ....utils.types import OptimizersConfig
from .._registory import MODEL_REGISTORY, OPTIMIZERS_REGISTORY
from .._utils import asdict
from ._base import override_model_config as override_base_model_config
from ._base import override_optimizers_config as override_base_optimizers_config


@MODEL_REGISTORY.register()
def create_model_config(base_model_config: ModelConfig) -> NeuSModelConfig:
    model_config = NeuSModelConfig(**asdict(base_model_config))
    model_config = override_base_model_config(model_config)
    model_config = dataclasses.replace(
        model_config,
        sampler_type=SamplerType.PROPOSAL,
    )
    return model_config


@OPTIMIZERS_REGISTORY.register()
def create_optimizers_config(base_optimizers_config: OptimizersConfig) -> OptimizersConfig:
    base_optimizers_config = override_base_optimizers_config(base_optimizers_config)
    return base_optimizers_config
