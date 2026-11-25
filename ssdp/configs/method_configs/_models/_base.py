import dataclasses

from ssdp.models import NeuSModelConfig

from ....utils.types import OptimizersConfig


def override_model_config[T: NeuSModelConfig](model_config: T) -> T:
    model_config = dataclasses.replace(
        model_config,
        near_plane=0.0,
        far_plane=5.0,
        background_model="mlp",
    )
    return model_config


def override_optimizers_config(optimizers_config: OptimizersConfig) -> OptimizersConfig:
    return optimizers_config
