import dataclasses
import enum
import math
from typing import Any, override

import jaxtyping as jt
import torch
import torch.nn as nn

from nerfstudio.cameras.rays import RaySamples
from ssdp.utils.jaxtyping import jaxtyped

from .sdf import SDF, SDFConfig


class UNISKernelType(enum.StrEnum):
    LOGISTIC = enum.auto()  # UNIS-C0 (Eq. 19): algorithmically equivalent to NeuS
    LAPLACE = enum.auto()  # UNIS-C1 (Eq. 20): first-order unbiased version of VolSDF
    ALGEBRAIC = enum.auto()  # UNIS-C2 (Eq. 21)
    SOFTPLUS = enum.auto()  # UNIS-C3 (Eq. 22)


# Scaling factors c = h'(0) / h(0)^2 in Eq. (10) for the increasing kernels below.
SCALING_FACTORS: dict[UNISKernelType, float] = {
    UNISKernelType.LOGISTIC: 1.0,
    UNISKernelType.LAPLACE: 2.0,
    UNISKernelType.ALGEBRAIC: 2.0,
    UNISKernelType.SOFTPLUS: 1.0 / (2.0 * math.log(2.0) ** 2.0),
}


class SoftplusAntiderivative(torch.autograd.Function):
    # H(x) = ∫_{-∞}^{x} softplus(u) du = -Li₂(-eˣ), whose derivative is softplus(x) itself.
    # Forward: Li₂(z) = Σ_{n≥0} Bₙ uⁿ⁺¹ / (n+1)! with u = -log(1-z) (Bernoulli series).
    # For z = -eˣ and x ≤ 0, u = -softplus(x) ∈ [-log 2, 0), so the truncation error is < 1e-14.
    # For x > 0, the inversion formula H(x) + H(-x) = x² / 2 + π² / 6 reduces it to x ≤ 0.
    # Backward: softplus(x), exactly.
    COEFFICIENTS: tuple[tuple[int, float], ...] = (
        (1, 1.0),
        (2, 1.0 / 4.0),
        (3, 1.0 / 36.0),
        (5, -1.0 / 3600.0),
        (7, 1.0 / 211680.0),
        (9, -1.0 / 10886400.0),
        (11, 1.0 / 526901760.0),
        (13, -691.0 / (2730.0 * 6227020800.0)),
    )

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        dtype = inputs.dtype
        inputs = inputs.to(torch.float64)
        softplus_values = nn.functional.softplus(-torch.abs(inputs))
        series_values = sum(
            coefficient * softplus_values**exponent
            for exponent, coefficient in SoftplusAntiderivative.COEFFICIENTS
        )
        outputs = torch.where(
            condition=inputs > 0.0,
            input=inputs**2.0 / 2.0 + math.pi**2.0 / 6.0 - series_values,
            other=series_values,
        )
        return outputs.to(dtype)

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> torch.Tensor:
        (inputs,) = ctx.saved_tensors
        return grad_outputs * nn.functional.softplus(inputs)


@jaxtyped()
def antiderivative(
    inputs: jt.Float[torch.Tensor, " *B "],
    kernel_type: UNISKernelType,
) -> jt.Float[torch.Tensor, " *B "]:
    # H(x) = ∫_{-∞}^{x} h(u) du for the increasing kernel h, so that H(-∞) = 0.
    if kernel_type is UNISKernelType.LOGISTIC:
        outputs = nn.functional.softplus(inputs)
    if kernel_type is UNISKernelType.LAPLACE:
        outputs = torch.where(
            condition=inputs <= 0.0,
            input=torch.exp(torch.clamp(inputs, max=0.0)) / 2.0,
            other=inputs + torch.exp(-torch.clamp(inputs, min=0.0)) / 2.0,
        )
    if kernel_type is UNISKernelType.ALGEBRAIC:
        sqrt_values = torch.sqrt(1.0 + inputs**2.0)
        outputs = torch.where(
            condition=inputs >= 0.0,
            input=inputs + sqrt_values,
            other=1.0 / (sqrt_values - inputs),
        )
        outputs = outputs / 2.0
    if kernel_type is UNISKernelType.SOFTPLUS:
        outputs = SoftplusAntiderivative.apply(inputs)
    return outputs


@dataclasses.dataclass
class UNISConfig(SDFConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: UNIS,
    )
    kernel_type: UNISKernelType = UNISKernelType.LAPLACE


class UNIS(SDF):
    """UNIS-Facto: the k = 0 density mappings of UNIS (Deng et al., ICCV 2025) within NeuS-Facto.

    UNIS maps an SDF f to the density σ = c s (f')^{k+1} h(s f (f')^k) (Eq. 10). For k = 0, σ is the
    derivative of R(f(t)) := c H(-s f(t)) along the ray, where H' = h, so the opacity of each interval is
    given in closed form by 1 - exp(-(R(f_{i+1}) - R(f_i))). This is the exact interval integration of
    σ under the local-planarity assumption stated in their Appendix F, with the endpoint SDF values
    estimated as in NeuS-Facto. The logistic kernel (UNIS-C0, Eq. 19) reduces identically to NeuS's
    opacity, while the Laplace, algebraic, and softplus kernels correspond to UNIS-C1/C2/C3 (Eqs. 20-22).
    """

    config: UNISConfig

    @jaxtyped()
    def _get_cumulative_optical_depth(
        self,
        sdf_values: jt.Float[torch.Tensor, " *B 1 "],
        inv_s: jt.Float[torch.Tensor, " 1 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        # R(f) = c H(-s f): the optical depth accumulated from f = +∞ down to f.
        scaling_factor = SCALING_FACTORS[self.config.kernel_type]
        cumulative_optical_depths = antiderivative(-sdf_values * inv_s, self.config.kernel_type)
        cumulative_optical_depths = cumulative_optical_depths * scaling_factor
        return cumulative_optical_depths

    @jaxtyped()
    def _get_opacity(
        self,
        prev_sdf_values: jt.Float[torch.Tensor, " *B 1 "],
        next_sdf_values: jt.Float[torch.Tensor, " *B 1 "],
        inv_s: jt.Float[torch.Tensor, " 1 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        prev_cumulative_optical_depths = self._get_cumulative_optical_depth(prev_sdf_values, inv_s)
        next_cumulative_optical_depths = self._get_cumulative_optical_depth(next_sdf_values, inv_s)
        optical_depths = next_cumulative_optical_depths - prev_cumulative_optical_depths
        # NOTE: Exiting intervals (increasing SDF) are clamped to zero opacity as in NeuS.
        opacities = -torch.expm1(-optical_depths)
        opacities = torch.clamp(opacities, min=0.0, max=1.0)
        return opacities

    @jaxtyped()
    @override
    def get_alpha(
        self,
        ray_samples: RaySamples,
        sdf_means: jt.Float[torch.Tensor, " *B 1 "],
        sdf_gradients: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        inv_s = self.deviation_network.get_variance()

        # NOTE: The endpoint SDF values are estimated exactly as in `SDFField.get_alpha` of Nerfstudio,
        # including the cosine annealing of NeuS, so that only the opacity formula differs from NeuS-Facto.
        true_cos = torch.sum(ray_samples.frustums.directions * sdf_gradients, dim=-1, keepdim=True)
        cos_anneal_ratio = self._cos_anneal_ratio
        iter_cos = -(
            nn.functional.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
            + nn.functional.relu(-true_cos) * cos_anneal_ratio
        )
        next_sdf_values = sdf_means + iter_cos * ray_samples.deltas * 0.5
        prev_sdf_values = sdf_means - iter_cos * ray_samples.deltas * 0.5

        alphas = self._get_opacity(
            prev_sdf_values=prev_sdf_values,
            next_sdf_values=next_sdf_values,
            inv_s=inv_s,
        )
        return alphas
