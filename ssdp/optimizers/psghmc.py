import dataclasses

import torch
import torch.nn as nn
from torch.optim.optimizer import ParamsT

from nerfstudio.engine.optimizers import OptimizerConfig


@dataclasses.dataclass
class PSGHMCConfig(OptimizerConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: PSGHMC,
    )
    momentum: float = 0.9
    alpha: float = 0.99
    temperature: float = 1.0

    def setup(self, params: ParamsT) -> torch.optim.Optimizer:
        return self._target(
            params,
            lr=self.lr,
            momentum=self.momentum,
            alpha=self.alpha,
            eps=self.eps,
            temperature=self.temperature,
        )


class PSGHMC(torch.optim.SGD):
    def __init__(
        self,
        params: ParamsT,
        lr: float,
        momentum: float = 0.0,
        alpha: float = 0.99,
        eps: float = 1.0e-8,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(params, lr=lr, momentum=momentum)
        self.defaults.update(alpha=alpha, eps=eps, temperature=temperature)
        for group in self.param_groups:
            group.update(alpha=alpha, eps=eps, temperature=temperature)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            group: dict[str, float]

            lr = group["lr"]
            momentum = group["momentum"]
            temperature = group["temperature"]

            for param in group["params"]:
                param: nn.Parameter

                if param.grad is None:
                    continue

                self._init_state(param)

                noise = torch.randn_like(param.grad)
                preconditioner = self._get_preconditioner(param, group)
                noise_scale = (2.0 * (1.0 - momentum) * temperature / lr * preconditioner).sqrt()

                param.grad.mul_(preconditioner)
                param.grad.add_(noise_scale * noise)

        super().step()

    def _init_state(self, param: torch.Tensor) -> None:
        state: dict[str, torch.Tensor] = self.state[param]
        if not state:
            state.update(square_avg=torch.zeros_like(param))

    def _get_preconditioner(self, param: torch.Tensor, group: dict[str, float]) -> torch.Tensor:
        state: dict[str, torch.Tensor] = self.state[param]
        state["square_avg"].mul_(group["alpha"])
        state["square_avg"].addcmul_(param.grad, param.grad, value=1.0 - group["alpha"])
        preconditioner = (state["square_avg"].sqrt() + group["eps"]).reciprocal()
        return preconditioner
