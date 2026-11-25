import dataclasses
import math

from ssdp.fields import NeuSFieldConfig, SSDPFieldConfig

from ....utils.types import OptimizersConfig


def _variance_to_beta(variance: float) -> float:
    return math.log(variance * 3.0 / math.pi**2.0) / -20.0


def override_field_config[T: NeuSFieldConfig](field_config: T) -> T:
    field_config = dataclasses.replace(
        field_config,
        bias=0.2,
        inside_outside=False,
        beta_init=_variance_to_beta(SSDPFieldConfig.prior_initial_var),
    )
    return field_config


def override_optimizers_config(optimizers_config: OptimizersConfig) -> OptimizersConfig:
    return optimizers_config
