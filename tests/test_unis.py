"""Tests for `ssdp.fields.unis` (UNIS-Facto).

Run with `uv run --with pytest pytest tests/test_unis.py` or `uv run python tests/test_unis.py`.
"""

import math
from collections.abc import Callable

import numpy as np
import scipy.special
import torch

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.sdf_field import SDFField
from ssdp.fields.unis import (
    SCALING_FACTORS,
    UNIS,
    SoftplusAntiderivative,
    UNISConfig,
    UNISKernelType,
    antiderivative,
)

# NOTE: `tinycudann` requires CUDA.
DEVICE = torch.device("cuda")


def _reference_kernel(inputs: np.ndarray, kernel_type: UNISKernelType) -> np.ndarray:
    # Increasing kernels h(x) of UNIS: logistic and Laplace CDFs, and Eqs. (17) and (18).
    if kernel_type is UNISKernelType.LOGISTIC:
        return scipy.special.expit(inputs)
    if kernel_type is UNISKernelType.LAPLACE:
        return np.where(inputs <= 0.0, np.exp(inputs) / 2.0, 1.0 - np.exp(-inputs) / 2.0)
    if kernel_type is UNISKernelType.ALGEBRAIC:
        return 0.5 + 0.5 * inputs / np.sqrt(1.0 + inputs**2.0)
    if kernel_type is UNISKernelType.SOFTPLUS:
        return np.logaddexp(0.0, inputs)
    raise ValueError(kernel_type)


def _create_field(kernel_type: UNISKernelType, inv_s: float) -> UNIS:
    config = UNISConfig(
        kernel_type=kernel_type,
        use_grid_feature=True,
        num_layers=2,
        num_layers_color=2,
        hidden_dim=256,
        bias=0.2,
        inside_outside=False,
        use_appearance_embedding=False,
    )
    field: UNIS = config.setup(
        aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        spatial_distortion=SceneContraction(order=float("inf")),
        num_images=1,
        use_average_appearance_embedding=False,
    )
    field = field.to(DEVICE)
    # `LearnedVariance.get_variance` returns exp(10 * variance) =: inv_s.
    field.deviation_network.variance.data.fill_(math.log(inv_s) / 10.0)
    return field


def _create_ray_samples(
    num_rays: int = 64,
    num_samples: int = 48,
    generator: torch.Generator | None = None,
) -> RaySamples:
    origins = torch.rand(num_rays, 1, 3, generator=generator) * 0.5 - 0.25
    directions = torch.randn(num_rays, 1, 3, generator=generator)
    directions = torch.nn.functional.normalize(directions, p=2, dim=-1)
    bins = torch.sort(torch.rand(num_rays, num_samples + 1, 1, generator=generator) * 0.5, dim=1)[
        0
    ]
    starts, ends = bins[:, :-1], bins[:, 1:]
    frustums = Frustums(
        origins=origins.expand(num_rays, num_samples, 3),
        directions=directions.expand(num_rays, num_samples, 3),
        starts=starts,
        ends=ends,
        pixel_area=torch.ones(num_rays, num_samples, 1),
    )
    ray_samples = RaySamples(
        frustums=frustums,
        camera_indices=torch.zeros(num_rays, num_samples, 1, dtype=torch.long),
        deltas=ends - starts,
    )
    return ray_samples.to(DEVICE)


def test_logistic_kernel_reduces_to_neus() -> None:
    # UNIS-C0 (Eq. 19) must reproduce the discrete opacity of NeuS (inherited from Nerfstudio).
    generator = torch.Generator().manual_seed(0)
    for inv_s in (1.0, 20.0, 400.0):
        for cos_anneal_ratio in (0.0, 0.3, 1.0):
            field = _create_field(UNISKernelType.LOGISTIC, inv_s=inv_s)
            field.set_cos_anneal_ratio(cos_anneal_ratio)
            ray_samples = _create_ray_samples(generator=generator)
            sdf_values = torch.randn(64, 48, 1, generator=generator).to(DEVICE) * 0.1
            sdf_gradients = torch.randn(64, 48, 3, generator=generator).to(DEVICE)

            alphas = field.get_alpha(ray_samples, sdf_values, sdf_gradients)
            alphas_neus = SDFField.get_alpha(field, ray_samples, sdf_values, sdf_gradients)

            true_cos = torch.sum(
                ray_samples.frustums.directions * sdf_gradients, dim=-1, keepdim=True
            )
            iter_cos = -(
                torch.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
                + torch.relu(-true_cos) * cos_anneal_ratio
            )
            next_sdf_values = sdf_values + iter_cos * ray_samples.deltas * 0.5
            prev_sdf_values = sdf_values - iter_cos * ray_samples.deltas * 0.5
            prev_cdf_values = torch.sigmoid(prev_sdf_values.double() * inv_s)
            next_cdf_values = torch.sigmoid(next_sdf_values.double() * inv_s)
            alphas_ratio = torch.clamp(1.0 - next_cdf_values / prev_cdf_values, 0.0, 1.0)
            assert torch.allclose(alphas.double(), alphas_ratio, atol=1.0e-5), (
                inv_s,
                cos_anneal_ratio,
            )

            # NOTE: Nerfstudio stabilizes the ratio as (p + 1e-5) / (c + 1e-5), which perturbs the
            # opacity by O(1e-5 / c), so the comparison is restricted to intervals with c = prev_cdf > 0.1.
            masks = prev_cdf_values > 0.1
            assert torch.any(masks)
            assert torch.allclose(alphas[masks], alphas_neus[masks], atol=1.0e-4), (
                inv_s,
                cos_anneal_ratio,
            )


def test_antiderivative_matches_kernel() -> None:
    # H' = h for every kernel, both via autograd and via central finite differences.
    inputs = torch.linspace(-30.0, 30.0, 6001, dtype=torch.float64, requires_grad=True)
    epsilon = 1.0e-4
    for kernel_type in UNISKernelType:
        kernel_values = _reference_kernel(inputs.detach().numpy(), kernel_type)
        kernel_values = torch.as_tensor(kernel_values)

        outputs = antiderivative(inputs, kernel_type)
        (gradients,) = torch.autograd.grad(outputs.sum(), inputs)
        assert torch.allclose(gradients, kernel_values, atol=1.0e-6), kernel_type

        with torch.no_grad():
            finite_differences = (
                antiderivative(inputs + epsilon, kernel_type)
                - antiderivative(inputs - epsilon, kernel_type)
            ) / (2.0 * epsilon)
        assert torch.allclose(finite_differences, kernel_values, atol=1.0e-6), kernel_type

        # H is non-negative and non-decreasing with H(-∞) = 0, so that empty space is transparent.
        # NOTE: The algebraic kernel decays only polynomially, h(x) ~ 1 / (4 x^2) as x -> -∞, and
        # thus H(x) ~ -1 / (4 x), whereas the other kernels decay exponentially.
        assert torch.all(outputs >= 0.0), kernel_type
        assert torch.all(outputs[1:] >= outputs[:-1]), kernel_type
        if kernel_type is UNISKernelType.ALGEBRAIC:
            assert math.isclose(outputs[0].item(), 1.0 / (4.0 * 30.0), rel_tol=1.0e-3), kernel_type
        else:
            assert outputs[0].item() < 1.0e-12, kernel_type
        kernel_value_0 = _reference_kernel(np.zeros(()), kernel_type).item()
        kernel_derivative_0 = (
            _reference_kernel(np.full((), epsilon), kernel_type)
            - _reference_kernel(np.full((), -epsilon), kernel_type)
        ).item() / (2.0 * epsilon)
        scaling_factor = kernel_derivative_0 / kernel_value_0**2.0
        # NOTE: The Laplace kernel has a jump in h'' at 0, so the central difference is O(epsilon).
        assert math.isclose(SCALING_FACTORS[kernel_type], scaling_factor, rel_tol=1.0e-3), (
            kernel_type
        )

    # The generated Bernoulli series starts with v + v² / 4 + v³ / 36 - v⁵ / 3600 + v⁷ / 211680.
    exponents, coefficients = zip(*SoftplusAntiderivative.COEFFICIENTS, strict=True)
    assert exponents == (1, 2, 3, 5, 7, 9, 11, 13)
    assert np.allclose(
        coefficients[:5], [1.0, 1.0 / 4.0, 1.0 / 36.0, -1.0 / 3600.0, 1.0 / 211680.0]
    )

    # The softplus antiderivative is -Li₂(-eˣ) = -spence(1 + eˣ) (SciPy's convention).
    with torch.no_grad():
        outputs = antiderivative(inputs, UNISKernelType.SOFTPLUS)
    references = -scipy.special.spence(1.0 + np.exp(inputs.detach().numpy()))
    assert np.allclose(outputs.numpy(), references, rtol=1.0e-12, atol=1.0e-12)
    assert math.isclose(outputs[3000].item(), math.pi**2.0 / 12.0, rel_tol=1.0e-12)

    # Single precision inputs are evaluated in double precision internally.
    with torch.no_grad():
        outputs_fp32 = antiderivative(inputs.detach().float(), UNISKernelType.SOFTPLUS)
    assert outputs_fp32.dtype == torch.float32
    assert torch.allclose(outputs_fp32.double(), outputs, rtol=1.0e-6, atol=1.0e-6)


def test_opacity_boundary_behavior() -> None:
    generator = torch.Generator().manual_seed(0)
    for kernel_type in UNISKernelType:
        field = _create_field(kernel_type, inv_s=20.0)
        inv_s = field.deviation_network.get_variance()

        # Far outside the surface, intervals are transparent, except for the algebraic kernel,
        # whose polynomial tail yields R(f) ~ 1 / (2 s f) and thus a small residual opacity.
        prev_sdf_values = torch.full((1000, 1), 2.0, device=DEVICE)
        next_sdf_values = prev_sdf_values - 0.01
        alphas = field._get_opacity(prev_sdf_values, next_sdf_values, inv_s)
        assert torch.all(alphas >= 0.0), kernel_type
        if kernel_type is UNISKernelType.ALGEBRAIC:
            alphas_tail = (1.0 / next_sdf_values - 1.0 / prev_sdf_values) / (2.0 * inv_s)
            assert torch.allclose(alphas, alphas_tail, rtol=1.0e-2), kernel_type
        else:
            assert torch.all(alphas < 1.0e-6), kernel_type

        # Exiting intervals (increasing SDF) have zero opacity, as in NeuS.
        prev_sdf_values = torch.randn(1000, 1, generator=generator).to(DEVICE) * 0.1
        next_sdf_values = (
            prev_sdf_values + torch.rand(1000, 1, generator=generator).to(DEVICE) * 0.1
        )
        alphas = field._get_opacity(prev_sdf_values, next_sdf_values, inv_s)
        assert torch.all(alphas == 0.0), kernel_type

        # Entering intervals have opacities in [0, 1), increasing with the penetration depth.
        next_sdf_values = (
            prev_sdf_values - torch.rand(1000, 1, generator=generator).to(DEVICE) * 0.1
        )
        alphas = field._get_opacity(prev_sdf_values, next_sdf_values, inv_s)
        assert torch.all(alphas >= 0.0) and torch.all(alphas < 1.0), kernel_type
        assert torch.all(torch.isfinite(alphas)), kernel_type
        deeper_alphas = field._get_opacity(prev_sdf_values, next_sdf_values - 0.05, inv_s)
        assert torch.all(deeper_alphas >= alphas), kernel_type

        # Deep inside the surface, intervals are opaque.
        prev_sdf_values = torch.full((1000, 1), -2.0, device=DEVICE)
        next_sdf_values = prev_sdf_values - 1.0
        alphas = field._get_opacity(prev_sdf_values, next_sdf_values, inv_s)
        assert torch.all(alphas > 1.0 - 1.0e-6), kernel_type

        # Extreme inputs do not produce NaNs or infinities in the forward pass.
        sdf_values = torch.tensor([[-1.0e6], [-1.0e3], [0.0], [1.0e3], [1.0e6]], device=DEVICE)
        alphas = field._get_opacity(sdf_values, sdf_values - 1.0, inv_s)
        assert torch.all(torch.isfinite(alphas)), kernel_type


def test_gradients_flow_to_field_and_deviation_network() -> None:
    generator = torch.Generator().manual_seed(0)
    for kernel_type in UNISKernelType:
        field = _create_field(kernel_type, inv_s=20.0)
        field.train()
        ray_samples = _create_ray_samples(generator=generator)

        outputs = field.get_outputs(ray_samples, return_alphas=True)
        alphas = outputs[FieldHeadNames.ALPHA]
        assert torch.all(torch.isfinite(alphas)), kernel_type
        alphas.sum().backward()

        variance_grad = field.deviation_network.variance.grad
        assert variance_grad is not None and torch.all(torch.isfinite(variance_grad)), kernel_type
        assert torch.any(variance_grad != 0.0), kernel_type

        geo_parameters = [
            parameter
            for name, parameter in field.named_parameters()
            if name.startswith(("glin", "encoding"))
        ]
        assert len(geo_parameters) > 0
        for parameter in geo_parameters:
            assert parameter.grad is not None and torch.all(torch.isfinite(parameter.grad)), (
                kernel_type
            )
        assert any(torch.any(parameter.grad != 0.0) for parameter in geo_parameters), kernel_type


if __name__ == "__main__":
    tests: list[Callable[[], None]] = [
        test_logistic_kernel_reduces_to_neus,
        test_antiderivative_matches_kernel,
        test_opacity_boundary_behavior,
        test_gradients_flow_to_field_and_deviation_network,
    ]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
