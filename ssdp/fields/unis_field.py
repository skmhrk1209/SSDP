import dataclasses
import enum
import functools
import itertools
import math
import operator
from collections.abc import Callable, Iterator
from fractions import Fraction
from typing import Any, override

import jaxtyping as jt
import torch
import torch.nn as nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from ssdp.utils.function import signature_of
from ssdp.utils.jaxtyping import jaxtyped

from .neus_field import NeuSField, NeuSFieldConfig, RaySampleLocation


def _bernoulli_numbers() -> Iterator[Fraction]:
    number = Fraction(1)
    numbers = [number]
    yield number
    for n in itertools.count(1):
        total = functools.reduce(
            operator.add,
            (math.comb(n + 1, k) * numbers[k] for k in range(n)),
        )
        number = -total / (n + 1)
        numbers.append(number)
        yield number


class _SoftplusAntiderivative(torch.autograd.Function):
    @staticmethod
    @functools.cache
    def _get_coefficients(dtype: torch.dtype) -> tuple[float, ...]:
        # NOTE: Truncate the series in `_forward`.
        # Each term cₙ f(x)ⁿ⁺¹ is largest at x = 0 over x ≤ 0, where f(0) = log 2 and F(0) = π² / 12.
        # The spacing of `dtype` in [1 / 2, 1) is ε / 2, so adding a value below ε / 4 cannot change F(0).
        tolerance = torch.finfo(dtype).eps / 4.0
        coefficients = []
        for n, b_n in enumerate(_bernoulli_numbers()):
            coefficient = b_n * (-1) ** n / math.factorial(n + 1)
            if coefficient and abs(coefficient) * math.log(2.0) ** (n + 1) < tolerance:
                break
            coefficients.append(float(coefficient))
        return tuple(coefficients)

    @staticmethod
    @jaxtyped()
    def _forward(inputs: jt.Float[torch.Tensor, " *B "]) -> jt.Float[torch.Tensor, " *B "]:
        # NOTE: Compute the antiderivative F(x) = ∫₋∞ˣ f(t) dt of the softplus function f.
        # For x ≤ 0, F(x) = Σₙ cₙ f(x)ⁿ⁺¹, where cₙ = bₙ (-1)ⁿ / (n + 1)! and bₙ is the n-th Bernoulli number.
        # For x > 0, since f(x) - f(-x) = x, we use F(x) + F(-x) = x² / 2 + 2 F(0).
        outputs = nn.functional.softplus(-torch.abs(inputs))
        coefficients = _SoftplusAntiderivative._get_coefficients(inputs.dtype)
        outputs = functools.reduce(
            operator.add,
            (outputs ** (n + 1) * c_n for n, c_n in enumerate(coefficients) if c_n),
        )
        outputs = torch.where(
            condition=inputs < 0.0,
            input=outputs,
            other=inputs**2.0 / 2.0 + math.pi**2.0 / 6.0 - outputs,
        )
        return outputs

    @staticmethod
    @jaxtyped()
    @override
    def forward(
        ctx: torch.autograd.function.BackwardCFunction,
        inputs: jt.Float[torch.Tensor, " *B "],
    ) -> jt.Float[torch.Tensor, " *B "]:
        ctx.save_for_backward(inputs)
        return _SoftplusAntiderivative._forward(inputs)

    @staticmethod
    @jaxtyped()
    @override
    def backward(
        ctx: torch.autograd.function.BackwardCFunction,
        output_gradients: jt.Float[torch.Tensor, " *B "],
    ) -> jt.Float[torch.Tensor, " *B "]:
        (inputs,) = ctx.saved_tensors
        input_gradients = output_gradients * nn.functional.softplus(inputs)
        return input_gradients

    @signature_of(_forward)
    @classmethod
    @override
    def apply(cls, *args: Any, **kwargs: Any) -> Any:
        return super().apply(*args, **kwargs)


class UNISKernelType(enum.StrEnum):
    LOGISTIC = enum.auto()
    LAPLACE = enum.auto()
    ALGEBRAIC = enum.auto()
    SOFTPLUS = enum.auto()


@dataclasses.dataclass
class UNISFieldConfig(NeuSFieldConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: UNISField,
    )
    kernel_type: UNISKernelType = UNISKernelType.LAPLACE


class UNISField(NeuSField):
    config: UNISFieldConfig

    @property
    def _kernel_antiderivative(self) -> Callable[[torch.Tensor], torch.Tensor]:
        @jaxtyped()
        def antiderivative(
            inputs: jt.Float[torch.Tensor, " *B "],
        ) -> jt.Float[torch.Tensor, " *B "]:
            match self.config.kernel_type:
                case UNISKernelType.LOGISTIC:
                    outputs = nn.functional.softplus(inputs)
                case UNISKernelType.LAPLACE:
                    outputs = torch.where(
                        condition=inputs < 0.0,
                        input=torch.exp(torch.clamp(inputs, max=0.0)) / 2.0,
                        other=inputs + torch.exp(-torch.clamp(inputs, min=0.0)) / 2.0,
                    )
                case UNISKernelType.ALGEBRAIC:
                    outputs = torch.sqrt(1.0 + inputs**2.0)
                    outputs = torch.where(
                        condition=inputs < 0.0,
                        # Just for numerical stability.
                        input=1.0 / (outputs - torch.clamp(inputs, max=0.0)),
                        other=inputs + outputs,
                    )
                    outputs = outputs / 2.0
                case UNISKernelType.SOFTPLUS:
                    outputs = _SoftplusAntiderivative.apply(inputs)
            return outputs

        return antiderivative

    @property
    def _kernel_scale_factor(self) -> float:
        match self.config.kernel_type:
            case UNISKernelType.LOGISTIC:
                kernel_scale_factor = 1.0
            case UNISKernelType.LAPLACE:
                kernel_scale_factor = 2.0
            case UNISKernelType.ALGEBRAIC:
                kernel_scale_factor = 2.0
            case UNISKernelType.SOFTPLUS:
                kernel_scale_factor = 1.0 / (2.0 * math.log(2.0) ** 2.0)
        return kernel_scale_factor

    @jaxtyped()
    def _get_optical_depth(
        self,
        sdf_values: jt.Float[torch.Tensor, " *B 1 "],
        sdf_scales: jt.Float[torch.Tensor, " *B 1 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        optical_depths = self._kernel_antiderivative(-sdf_values / sdf_scales)
        optical_depths = optical_depths * self._kernel_scale_factor
        return optical_depths

    @jaxtyped()
    @override
    def get_opacity(
        self,
        intervals: jt.Float[torch.Tensor, " *B 1 "],
        directions: jt.Float[torch.Tensor, " *B 3 "],
        sdf_values: jt.Float[torch.Tensor, " *B 1 "],
        sdf_gradients: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B 1 "]:
        sdf_gradients = torch.sum(directions * sdf_gradients, dim=-1, keepdim=True)
        sdf_gradients = torch.lerp(
            input=-nn.functional.relu(-sdf_gradients * 0.5 + 0.5),
            end=-nn.functional.relu(-sdf_gradients),
            weight=self._cos_anneal_ratio,
        )

        prev_sdf_values = sdf_values - sdf_gradients * intervals * 0.5
        next_sdf_values = sdf_values + sdf_gradients * intervals * 0.5

        sdf_scales = self.sdf_scale.expand_as(sdf_values)
        prev_optical_depths = self._get_optical_depth(
            sdf_values=prev_sdf_values,
            sdf_scales=sdf_scales,
        )
        next_optical_depths = self._get_optical_depth(
            sdf_values=next_sdf_values,
            sdf_scales=sdf_scales,
        )

        opacities = -torch.expm1(prev_optical_depths - next_optical_depths)
        opacities = torch.clamp(opacities, min=0.0, max=1.0)

        return opacities

    @override
    def get_outputs(
        self,
        ray_samples: RaySamples,
        return_alphas: bool = True,
        return_colors: bool = True,
        return_gradients: bool | None = None,
        ray_sample_location: RaySampleLocation = RaySampleLocation.START,
    ) -> dict[FieldHeadNames | str, torch.Tensor]:
        outputs = super().get_outputs(
            ray_samples=ray_samples,
            return_alphas=False,
            return_colors=return_colors,
            return_gradients=return_gradients or return_alphas,
            ray_sample_location=ray_sample_location,
        )

        if return_alphas:
            assert ray_samples.deltas is not None
            intervals = torch.clamp(ray_samples.deltas, min=0.0)
            directions = ray_samples.frustums.directions
            sdf_values = outputs[FieldHeadNames.SDF]
            sdf_gradients = outputs[FieldHeadNames.GRADIENT]
            opacities = self.get_opacity(
                intervals=intervals,
                directions=directions,
                sdf_values=sdf_values,
                sdf_gradients=sdf_gradients,
            )
            outputs |= {FieldHeadNames.ALPHA: opacities}

        return outputs
