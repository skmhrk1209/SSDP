from typing import TypedDict

from nerfstudio.engine.optimizers import OptimizerConfig
from nerfstudio.engine.schedulers import (
    SchedulerConfig,
)


class OptimizerGroupConfig(TypedDict):
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig


type OptimizersConfig = dict[str, OptimizerGroupConfig]
