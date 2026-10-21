import dataclasses

import torch
from torch.optim.optimizer import ParamsT

from nerfstudio.engine.optimizers import OptimizerConfig

from .sghmc import SGHMC


@dataclasses.dataclass
class SGLDConfig(OptimizerConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: SGLD,
    )
    temperature: float = 1.0

    def setup(self, params: ParamsT) -> torch.optim.Optimizer:
        return self._target(params, lr=self.lr, temperature=self.temperature)


class SGLD(SGHMC):
    def __init__(
        self,
        params: ParamsT,
        lr: float,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            momentum=0.0,
            temperature=temperature,
        )
