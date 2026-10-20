import dataclasses

import torch
from torch.optim.optimizer import ParamsT

from nerfstudio.engine.optimizers import OptimizerConfig

from .psghmc import PSGHMC


@dataclasses.dataclass
class PSGLDConfig(OptimizerConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: PSGLD,
    )
    alpha: float = 0.99

    def setup(self, params: ParamsT) -> torch.optim.Optimizer:
        return self._target(
            params,
            lr=self.lr,
            alpha=self.alpha,
            eps=self.eps,
        )


class PSGLD(PSGHMC):
    def __init__(
        self,
        params: ParamsT,
        lr: float,
        alpha: float = 0.99,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            momentum=0.0,
            alpha=alpha,
            eps=eps,
        )
