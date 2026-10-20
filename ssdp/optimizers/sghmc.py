import dataclasses
from typing import Any

import torch
from torch.optim.optimizer import ParamsT

from nerfstudio.engine.optimizers import OptimizerConfig

from .psghmc import PSGHMC


@dataclasses.dataclass
class SGHMCConfig(OptimizerConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: SGHMC,
    )
    momentum: float = 0.0

    def setup(self, params: ParamsT) -> torch.optim.Optimizer:
        return self._target(params, lr=self.lr, momentum=self.momentum)


class SGHMC(PSGHMC):
    def __init__(self, params: ParamsT, lr: float, momentum: float = 0.0) -> None:
        super().__init__(params, lr=lr, momentum=momentum)

    def _init_state(self, param: torch.Tensor) -> None:
        pass

    def _get_preconditioner(self, param: torch.Tensor, group: dict[str, Any]) -> torch.Tensor:
        return torch.ones_like(param)
