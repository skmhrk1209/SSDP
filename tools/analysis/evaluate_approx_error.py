import dataclasses
import enum
import itertools
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import override

import jaxtyping as jt
import loguru
import numpy as np
import torch
import torch.nn as nn
import tqdm
import tyro
from matplotlib import colors

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import UniformSampler
from ssdp.fields.ssdp import SSDP, SSDPConfig
from ssdp.utils.jaxtyping import jaxtyped


class Variant(enum.StrEnum):
    MC = enum.auto()
    NA = enum.auto()
    NA_UP = enum.auto()
    BF = enum.auto()
    BF_UP = enum.auto()

    @property
    def is_survival_approx(self) -> bool:
        return self in (Variant.NA, Variant.NA_UP)

    @property
    def is_up_cross_approx(self) -> bool:
        return self in (Variant.NA, Variant.BF)


class Interpolation(enum.StrEnum):
    STEP = enum.auto()
    LINEAR = enum.auto()


@dataclasses.dataclass
class MonteCarloConfig:
    # NOTE: The Monte Carlo reference of all the experiments (`_get_reference_cdf_values` and
    # `evaluate_up_cross_prob.py`), recorded in their outputs.
    # The number of paths is set by the computation time of each experiment (one job of at most an hour, a power
    # of ten). The standard error of a mass p is then at most sqrt(p / num_mc_samples) before the extrapolation.
    num_mc_samples: int
    # NOTE: The zero-crossings are detected from the signs at the substeps of the sampling intervals, and also at
    # the substeps thinned out by the strides, for extrapolating the reference to infinitely many substeps
    # (`_get_extrapolation_weights`). At the largest kappa dt of the experiments (10, the top of the sweep of exp 1
    # in kappa dt; the trained fields reach 4.7 with kappa = 112 on the sampling interval 1/24, the maps 4.2),
    # a substep is at most 1/100 of the correlation length 1/kappa, and the coarsest level of the extrapolation
    # (n / 4) at most 1/10 of it, where the expansion behind the extrapolation holds (the drift is nearly constant
    # over a substep).
    num_sub_samples: int = 1000
    sub_sample_strides: tuple[int, ...] = (1, 2, 4)


def _configure_variant(field: SSDP, variant: Variant) -> None:
    field.set_progress_ratio(1.0)
    field.config.deterministic_end_ratio = 0.0
    field.config.survival_approx_end_ratio = 1.0 if variant.is_survival_approx else 0.0
    field.config.up_cross_approx_end_ratio = 1.0 if variant.is_up_cross_approx else 0.0
    field.config.up_cross_anneal_end_ratio = 0.0


@jaxtyped()
def _get_bin_values(ray_samples: RaySamples) -> jt.Float[torch.Tensor, " *R S "]:
    left_values = ray_samples.frustums.starts.squeeze(-1)
    rght_values = ray_samples.frustums.ends.squeeze(-1)
    bin_values = torch.cat([left_values, rght_values[..., -1:]], dim=-1)
    return bin_values


@jaxtyped()
def _get_ray_positions(
    ray_samples: RaySamples,
    num_sub_samples: int = 1,
) -> jt.Float[torch.Tensor, " *R S 3 "]:
    origins = ray_samples.frustums.origins[..., :1, :]
    directions = ray_samples.frustums.directions[..., :1, :]
    bin_values = _get_bin_values(ray_samples)
    bin_values = nn.functional.interpolate(
        input=bin_values.unsqueeze(0),
        size=(bin_values.shape[-1] - 1) * num_sub_samples + 1,
        mode="linear",
        align_corners=True,
    ).squeeze(0)
    positions = origins + directions * bin_values.unsqueeze(-1)
    return positions


@jaxtyped()
@torch.no_grad()
def _get_predictive_pmf_values(
    field: SSDP,
    ray_samples: RaySamples,
) -> jt.Float[torch.Tensor, " *R S "]:
    outputs = field(ray_samples, return_alphas=True)
    pmf_values = ray_samples.get_weights_and_transmittance_from_alphas(
        alphas=outputs[FieldHeadNames.ALPHA],
        weights_only=True,
    )
    pmf_values = pmf_values.squeeze(-1)
    return pmf_values


@jaxtyped()
@torch.no_grad()
def _get_transition_params(
    field: SSDP,
    ray_samples: RaySamples,
    num_sub_samples: int,
) -> tuple[
    jt.Float[torch.Tensor, " *R S "],
    jt.Float[torch.Tensor, " *R S-1 "],
    jt.Float[torch.Tensor, " *R S-1 "],
    jt.Float[torch.Tensor, " *R S-1 "],
]:
    ray_positions = _get_ray_positions(ray_samples)
    geo_outputs = field.forward_geo_network(ray_positions)
    _, geo_features = torch.split(
        tensor=geo_outputs,
        split_size_or_sections=(1, field.config.geo_feat_dim),
        dim=-1,
    )
    ou_drifts, ou_diffusions = field._get_ou_params(geo_features)

    ray_positions = _get_ray_positions(ray_samples, num_sub_samples)
    geo_outputs = field.forward_geo_network(ray_positions)
    marginal_means, _ = torch.split(
        tensor=geo_outputs,
        split_size_or_sections=(1, field.config.geo_feat_dim),
        dim=-1,
    )

    ray_intervals: torch.Tensor = ray_samples.deltas
    ray_intervals = torch.clamp(ray_intervals, min=0.0)

    marginal_means = marginal_means.squeeze(-1)
    ray_intervals = ray_intervals.squeeze(-1)

    ou_drifts = torch.repeat_interleave(ou_drifts, num_sub_samples, dim=-1)
    ou_diffusions = torch.repeat_interleave(ou_diffusions, num_sub_samples, dim=-1)
    ray_intervals = torch.repeat_interleave(
        ray_intervals / num_sub_samples, num_sub_samples, dim=-1
    )

    # NOTE: The variance floor is divided by the number of substeps
    # so that the variance composed over a coarse interval does not exceed the floor of the coarse transition.
    min_var_epsilon = field.config.min_var_epsilon
    field.config.min_var_epsilon = min_var_epsilon / num_sub_samples

    (
        transition_scales,
        transition_shifts,
        transition_vars,
    ) = field._get_transition_params(
        ou_drifts=ou_drifts,
        ou_diffusions=ou_diffusions,
        marginal_means=marginal_means,
        intervals=ray_intervals,
    )

    field.config.min_var_epsilon = min_var_epsilon

    return (
        marginal_means,
        transition_scales,
        transition_shifts,
        transition_vars,
    )


@jaxtyped()
@torch.no_grad()
def _get_empirical_obs_values(
    field: SSDP,
    ray_samples: RaySamples,
    num_mc_samples: int,
    num_sub_samples: int,
    sub_sample_strides: tuple[int, ...],
) -> tuple[
    jt.Float[torch.Tensor, " T {num_mc_samples} *R "],
    jt.Bool[torch.Tensor, " T {num_mc_samples} *R "],
]:
    (
        marginal_means,
        transition_scales,
        transition_shifts,
        transition_vars,
    ) = _get_transition_params(
        field=field,
        ray_samples=ray_samples,
        num_sub_samples=num_sub_samples,
    )

    mc_samples = field.get_mc_samples(
        initial_means=marginal_means[..., 0],
        transition_scales=transition_scales,
        transition_shifts=transition_shifts,
        transition_vars=transition_vars,
        num_mc_samples=num_mc_samples,
    )

    ray_ends = ray_samples.frustums.ends.squeeze(-1)

    obs_values_list, valid_flags_list = [], []

    # NOTE: The reference is free from the model under validation: the zero-crossings are detected only from the signs
    # at the substeps, so that it misses those inside a substep and converges slowly in the number of substeps.
    # They are also detected on the substeps thinned out by the strides, which are exact paths on coarser substeps,
    # for extrapolating the reference to infinitely many substeps (`_extrapolate_values`).
    for sub_sample_stride in sub_sample_strides:
        assert not num_sub_samples % sub_sample_stride
        sub_mc_samples = mc_samples[..., ::sub_sample_stride]
        cross_flags = (sub_mc_samples[..., :-1] > 0.0) & (sub_mc_samples[..., 1:] <= 0.0)
        valid_flags = torch.any(cross_flags, dim=-1)

        # NOTE: The first substep with a crossing (uint8 keeps the memory of this step small).
        obs_indices = torch.argmax(cross_flags.to(torch.uint8), dim=-1)
        obs_indices //= num_sub_samples // sub_sample_stride
        obs_values = torch.gather(
            input=ray_ends.expand(num_mc_samples, *ray_ends.shape),
            index=obs_indices.unsqueeze(-1),
            dim=-1,
        ).squeeze(-1)

        obs_values_list.append(obs_values)
        valid_flags_list.append(valid_flags)

    return torch.stack(obs_values_list, dim=0), torch.stack(valid_flags_list, dim=0)


@jaxtyped()
def _get_predictive_cdf_values(
    pmf_values: jt.Float[torch.Tensor, " *R S "],
) -> jt.Float[torch.Tensor, " *R S+1 "]:
    # NOTE: The CDF of the first-passage time censored at the far end of the ray, i.e., min(H, t_N). It ends at
    # the probability of the first passage inside the ray, and the rest of the mass is at the far end.
    cdf_values = torch.cumsum(pmf_values, dim=-1)
    cdf_values = torch.nn.functional.pad(
        input=cdf_values,
        pad=(1, 0),
        mode="constant",
        value=0.0,
    )
    return cdf_values


@jaxtyped()
def _get_empirical_cdf_indicators(
    bin_values: jt.Float[torch.Tensor, " *R S "],
    obs_values: jt.Float[torch.Tensor, " M *R "],
    valid_flags: jt.Bool[torch.Tensor, " M *R "],
) -> jt.Float[torch.Tensor, " M *R S "]:
    # NOTE: The indicator of the first passage of each path at or before each bin value, whose mean over the paths
    # is the empirical CDF.
    obs_values = obs_values.unsqueeze(-1)
    valid_flags = valid_flags.unsqueeze(-1)
    return ((obs_values <= bin_values) & valid_flags).to(torch.float64)


@jaxtyped()
def _normalize_cdf_values(
    cdf_values: jt.Float[torch.Tensor, " *R S "],
    epsilon: float = 1.0e-6,
) -> jt.Float[torch.Tensor, " *R S "]:
    # NOTE: The CDF conditioned on the first passage inside the ray, normalized as in `evaluate_uq_metrics.py`:
    # the probabilities of the intervals are floored and then normalized, so that the result is always a distribution
    # (the uniform one if there is no mass at all).
    pmf_values = torch.diff(cdf_values, dim=-1)
    pmf_values = torch.clamp(pmf_values, min=epsilon)
    pmf_values = nn.functional.normalize(pmf_values, p=1, dim=-1)
    cdf_values = torch.cumsum(pmf_values, dim=-1)
    cdf_values = nn.functional.pad(cdf_values, (1, 0), mode="constant", value=0.0)
    return cdf_values


@jaxtyped()
def _get_extrapolation_weights(
    sub_sample_strides: tuple[int, ...],
) -> jt.Float[torch.Tensor, " T "]:
    # NOTE: The detection from the signs at n substeps misses the zero-crossings inside a substep:
    # P(n) = P + c_1 n^(-1/2) + c_2 n^(-1) + ... The weights remove the leading terms from the probabilities on
    # the substeps thinned out by the strides, e.g., (8 P(n) - 6 P(n / 4) + P(n / 16)) / 3 for the strides (1, 4, 16).
    strides = torch.tensor(sub_sample_strides, dtype=torch.float64)
    orders = torch.arange(len(sub_sample_strides)).to(strides)
    return torch.linalg.solve(
        strides ** (orders.unsqueeze(-1) / 2.0), torch.eye(len(strides)).to(strides)[:, 0]
    )


@jaxtyped()
def _extrapolate_values(
    values: jt.Float[torch.Tensor, " T *B "],
    sub_sample_strides: tuple[int, ...],
) -> jt.Float[torch.Tensor, " *B "]:
    weights = _get_extrapolation_weights(sub_sample_strides)
    return torch.tensordot(weights.to(values), values, dims=1)


@jaxtyped()
@torch.no_grad()
def _get_reference_cdf_values(
    field: SSDP,
    ray_samples: RaySamples,
    monte_carlo: MonteCarloConfig,
    num_mc_chunks: int = 1,
) -> tuple[
    jt.Float[torch.Tensor, " *R S "],
    jt.Float[torch.Tensor, " *R S S "],
]:
    # NOTE: The Monte Carlo reference: the CDFs of the first passages detected on the substeps and on those
    # thinned out by the strides, extrapolated to infinitely many substeps, with the covariance of the extrapolated
    # CDF over the bins. The strides share the paths, so that the extrapolation is applied to the indicators of each
    # path: their mean over the paths is the reference, and their covariance divided by the number of paths is
    # that of the reference. The paths are drawn in equal chunks (for the memory only).
    assert not monte_carlo.num_mc_samples % num_mc_chunks
    bin_values = _get_bin_values(ray_samples)
    weights = _get_extrapolation_weights(monte_carlo.sub_sample_strides)
    sums, sums_of_products = 0.0, 0.0
    for _ in range(num_mc_chunks):
        obs_values, valid_flags = _get_empirical_obs_values(
            field=field,
            ray_samples=ray_samples,
            num_mc_samples=monte_carlo.num_mc_samples // num_mc_chunks,
            num_sub_samples=monte_carlo.num_sub_samples,
            sub_sample_strides=monte_carlo.sub_sample_strides,
        )
        indicators = sum(
            weight
            * _get_empirical_cdf_indicators(
                bin_values=bin_values,
                obs_values=stride_obs_values,
                valid_flags=stride_valid_flags,
            )
            for weight, stride_obs_values, stride_valid_flags in zip(
                weights.tolist(), obs_values, valid_flags, strict=True
            )
        )
        flat_indicators = indicators.reshape(indicators.shape[0], -1, indicators.shape[-1])
        sums = sums + torch.sum(indicators, dim=0)
        sums_of_products = sums_of_products + torch.einsum(
            "mrj,mrk->rjk", flat_indicators, flat_indicators
        ).reshape(*indicators.shape[1:], indicators.shape[-1])
    cdf_values = sums / monte_carlo.num_mc_samples
    cdf_covariances = sums_of_products / monte_carlo.num_mc_samples
    cdf_covariances = cdf_covariances - cdf_values.unsqueeze(-1) * cdf_values.unsqueeze(-2)
    return cdf_values, cdf_covariances / monte_carlo.num_mc_samples


@jaxtyped()
def _split_bin(
    values: jt.Float[torch.Tensor, " *R S "],
    interpolation: Interpolation,
) -> tuple[
    jt.Float[torch.Tensor, " *R S-1 "],
    jt.Float[torch.Tensor, " *R S-1 "],
]:
    match interpolation:
        case Interpolation.LINEAR:
            left_values = values[..., :-1]
            rght_values = values[..., 1:]
        case Interpolation.STEP:
            left_values = values[..., :-1]
            rght_values = values[..., :-1]
    return left_values, rght_values


@jaxtyped()
def _integrate_piecewise_linear(
    bin_values: jt.Float[torch.Tensor, " *R S "],
    left_values: jt.Float[torch.Tensor, " *R S-1 "],
    rght_values: jt.Float[torch.Tensor, " *R S-1 "],
    integrand_order: int,
) -> jt.Float[torch.Tensor, " *R "]:
    left_bin_values = bin_values[..., :-1]
    rght_bin_values = bin_values[..., 1:]
    match integrand_order:
        case 1:
            abs_values = torch.abs(left_values) + torch.abs(rght_values)
            integrals = torch.where(
                left_values * rght_values < 0.0,
                (left_values**2.0 + rght_values**2.0) / abs_values,
                abs_values,
            )
            integrals = integrals * (rght_bin_values - left_bin_values) / 2.0
        case 2:
            integrals = left_values**2.0 + left_values * rght_values + rght_values**2.0
            integrals = integrals * (rght_bin_values - left_bin_values) / 3.0
        case _:
            raise NotImplementedError(integrand_order)
    integrals = torch.sum(integrals, dim=-1)
    return integrals


@jaxtyped()
def _compute_cdf_distance(
    bin_values: jt.Float[torch.Tensor, " *R S "],
    cdf_values_1: jt.Float[torch.Tensor, " *R S "],
    cdf_values_2: jt.Float[torch.Tensor, " *R S "],
    interpolation_1: Interpolation,
    interpolation_2: Interpolation,
    integrand_order: int,
) -> jt.Float[torch.Tensor, " *R "]:
    left_cdf_values_1, rght_cdf_values_1 = _split_bin(
        values=cdf_values_1,
        interpolation=interpolation_1,
    )
    left_cdf_values_2, rght_cdf_values_2 = _split_bin(
        values=cdf_values_2,
        interpolation=interpolation_2,
    )
    left_values = left_cdf_values_1 - left_cdf_values_2
    rght_values = rght_cdf_values_1 - rght_cdf_values_2
    distances = _integrate_piecewise_linear(
        bin_values=bin_values,
        left_values=left_values,
        rght_values=rght_values,
        integrand_order=integrand_order,
    )
    return distances


@jaxtyped()
def _compute_wasserstein_distance(
    bin_values: jt.Float[torch.Tensor, " *R S "],
    cdf_values_1: jt.Float[torch.Tensor, " *R S "],
    cdf_values_2: jt.Float[torch.Tensor, " *R S "],
    interpolation_1: Interpolation = Interpolation.LINEAR,
    interpolation_2: Interpolation = Interpolation.LINEAR,
) -> jt.Float[torch.Tensor, " *R "]:
    return _compute_cdf_distance(
        bin_values=bin_values,
        cdf_values_1=cdf_values_1,
        cdf_values_2=cdf_values_2,
        interpolation_1=interpolation_1,
        interpolation_2=interpolation_2,
        integrand_order=1,
    )


@jaxtyped()
def _compute_cramer_distance(
    bin_values: jt.Float[torch.Tensor, " *R S "],
    cdf_values_1: jt.Float[torch.Tensor, " *R S "],
    cdf_values_2: jt.Float[torch.Tensor, " *R S "],
    interpolation_1: Interpolation = Interpolation.LINEAR,
    interpolation_2: Interpolation = Interpolation.LINEAR,
) -> jt.Float[torch.Tensor, " *R "]:
    return _compute_cdf_distance(
        bin_values=bin_values,
        cdf_values_1=cdf_values_1,
        cdf_values_2=cdf_values_2,
        interpolation_1=interpolation_1,
        interpolation_2=interpolation_2,
        integrand_order=2,
    )


@jaxtyped()
def _compute_metrics(
    bin_values: jt.Float[torch.Tensor, " *R S "],
    cdf_values_1: jt.Float[torch.Tensor, " *R S "],
    cdf_values_2: jt.Float[torch.Tensor, " *R S "],
) -> dict[str, jt.Float[torch.Tensor, " *R "]]:
    # NOTE: The distance between the distributions conditioned on the first passage inside the ray, the probabilities
    # of that first passage, and their squared distance.
    hit_probs_1, hit_probs_2 = cdf_values_1[..., -1], cdf_values_2[..., -1]
    metrics = dict(
        conditional_cramer_distance=_compute_cramer_distance(
            bin_values=bin_values,
            cdf_values_1=_normalize_cdf_values(cdf_values_1),
            cdf_values_2=_normalize_cdf_values(cdf_values_2),
        ),
        squared_distance=(hit_probs_1 - hit_probs_2) ** 2.0,
        hit_prob_1=hit_probs_1,
        hit_prob_2=hit_probs_2,
    )
    return metrics


@jaxtyped()
def _compute_metric_errors(
    bin_values: jt.Float[torch.Tensor, " *R S "],
    cdf_values_1: jt.Float[torch.Tensor, " *R S "],
    cdf_values_2: jt.Float[torch.Tensor, " *R S "],
    cdf_covariances_2: jt.Float[torch.Tensor, " *R S S "],
) -> dict[str, jt.Float[torch.Tensor, " *R "]]:
    # NOTE: The Monte Carlo errors of the distances of `_compute_metrics` from the covariance Sigma of the reference
    # CDF (the second one). A distance expanded to the second order in the noise of the reference (taken Gaussian),
    # with its gradient g and Hessian H in the reference CDF, has the bias tr(H Sigma) / 2 (the floor of
    # the distance where the renderer agrees with the reference, where g vanishes) and the variance
    # g^T Sigma g + tr((H Sigma)^2) / 2. The hit probability of the reference is linear in its CDF.
    with torch.enable_grad():
        cdf_values_2 = cdf_values_2.detach().requires_grad_(True)
        metrics = _compute_metrics(
            bin_values=bin_values,
            cdf_values_1=cdf_values_1.detach(),
            cdf_values_2=cdf_values_2,
        )
        errors = dict(hit_prob_2_stderr=torch.sqrt(cdf_covariances_2[..., -1, -1]))
        for name in ("conditional_cramer_distance", "squared_distance"):
            (grads,) = torch.autograd.grad(metrics[name].sum(), cdf_values_2, create_graph=True)
            hessians = torch.stack(
                [
                    torch.autograd.grad(
                        grads[..., index].sum(), cdf_values_2, retain_graph=True, allow_unused=True
                    )[0]
                    if grads[..., index].requires_grad
                    else torch.zeros_like(grads)
                    for index in range(grads.shape[-1])
                ],
                dim=-2,
            )
            grads, hessians = grads.detach(), hessians.detach()
            products = hessians @ cdf_covariances_2
            errors[f"{name}_bias"] = torch.diagonal(products, dim1=-2, dim2=-1).sum(dim=-1) / 2.0
            errors[f"{name}_stderr"] = torch.sqrt(
                torch.einsum("...j,...jk,...k->...", grads, cdf_covariances_2, grads)
                + torch.diagonal(products @ products, dim1=-2, dim2=-1).sum(dim=-1) / 2.0
            )
    return errors


@dataclasses.dataclass
class CuboidConfig:
    radii: tuple[float, float, float] = (0.025, 0.5, 0.5)
    range: tuple[float, float] = (-0.5, 0.5)
    count: int = 3

    @property
    def positions(self) -> list[tuple[float, float, float]]:
        return [[x, 0.0, 0.0] for x in np.linspace(*self.range, self.count).tolist()]

    @jaxtyped()
    def get_sdf(
        self,
    ) -> Callable[[jt.Float[torch.Tensor, " *B 3 "]], jt.Float[torch.Tensor, " *B 1 "]]:
        @jaxtyped()
        def sdf(p: jt.Float[torch.Tensor, " *B 3 "]) -> jt.Float[torch.Tensor, " *B 1 "]:
            c = p.new_tensor(self.positions)
            r = p.new_tensor(self.radii)
            p = p.unsqueeze(-2) - c
            q = torch.abs(p) - r
            d1 = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1, keepdim=True)
            d2 = torch.clamp(torch.amax(q, dim=-1, keepdim=True), max=0.0)
            d = torch.amin(d1 + d2, dim=-2)
            return d

        return sdf


@dataclasses.dataclass
class AnalyticSSDPConfig(SSDPConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: AnalyticSSDP,
    )


class AnalyticSSDP(SSDP):
    sdf: Callable[
        [jt.Float[torch.Tensor, " *B 3 "]],
        jt.Float[torch.Tensor, " *B 1 "],
    ]
    ou_drift: float
    ou_diffusion: float

    @override
    @jaxtyped()
    def forward_geo_network(
        self,
        positions: jt.Float[torch.Tensor, " *B 3 "],
    ) -> jt.Float[torch.Tensor, " *B 1+{self.config.geo_feat_dim} "]:
        sdf_values = self.sdf(positions)
        geo_features = positions.new_zeros(*positions.shape[:-1], self.config.geo_feat_dim)
        geo_outputs = torch.cat([sdf_values, geo_features], dim=-1)
        return geo_outputs

    @override
    @jaxtyped()
    def _get_ou_params(
        self,
        geo_features: jt.Float[torch.Tensor, " *R S {self.config.geo_feat_dim} "],
    ) -> tuple[
        jt.Float[torch.Tensor, " *R S-1 "],
        jt.Float[torch.Tensor, " *R S-1 "],
    ]:
        ou_drifts = torch.full_like(geo_features[..., :-1, 0], self.ou_drift)
        ou_diffusions = torch.full_like(geo_features[..., :-1, 0], self.ou_diffusion)
        return ou_drifts, ou_diffusions


def _create_field(min_var_epsilon: float, device: str) -> AnalyticSSDP:
    config = AnalyticSSDPConfig(min_var_epsilon=min_var_epsilon)
    field: AnalyticSSDP = config.setup(
        aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        num_images=1,
    )
    field = field.to(device)
    field = field.eval()
    return field


@dataclasses.dataclass
class ApproxErrorEvaluator:
    output_file: Path
    cuboid_config: CuboidConfig = dataclasses.field(
        default_factory=CuboidConfig,
    )
    ray_x_range: tuple[float, float] = (-1.0, 1.0)
    num_samples: int = 48
    # NOTE: The sample points are shifted by this fraction of a sampling interval, i.e., the phase of the slabs
    # (centered at sample points of the unshifted ray) relative to the sample points: at the phase 0.5 the centers
    # are in the middle of a sampling interval, where a slab thinner than the interval contains no sample point.
    phase: float = 0.0
    # NOTE: One scene per job of at most an hour (`MonteCarloConfig`). The standard error of the Monte Carlo CDF is
    # then at most 1 / (2 sqrt(num_mc_samples)) before the extrapolation; the errors of the distances from it are
    # recorded with them (`_compute_metric_errors`).
    monte_carlo: MonteCarloConfig = dataclasses.field(
        default_factory=lambda: MonteCarloConfig(num_mc_samples=100_000),
    )
    # NOTE: The paths of a grid point are drawn in this many chunks (for the GPU memory only; the path sampling
    # is a loop over the substeps, so more chunks cost proportionally more time).
    num_mc_chunks: int = 2
    # NOTE: The grid of kappa in [1e-2, 1e2] and tau in [1e-3, 1e1] of the OU process (the diffusion is tau^2 as in
    # `SSDP._get_transition_params`), uniform in log with 11 points each. It covers the parameters learned by
    # the training runs of `evaluate_training_trajectory.sh` (kappa reaches 112 at the end of some runs, which
    # `plot_approx_error.py` clips onto the top edge).
    ou_drifts: tuple[float, ...] = tuple(np.logspace(-2.0, 2.0, 11).tolist())
    ou_diffusions: tuple[float, ...] = tuple((np.logspace(-3.0, 1.0, 11) ** 2.0).tolist())
    initial_var: float = 1.0e-2
    # NOTE: The floor of the transition variance of the analytic field, far below every transition variance on
    # the grid (down to 5e-9 at tau = 1e-3), so that it never binds. The floor of the configuration of the paper
    # (1e-6) would bind at tau below about 1e-2.
    min_var_epsilon: float = 1.0e-12
    random_seed: int = 42
    device: str = "cuda"

    @torch.no_grad()
    def _set_initial_var(self, field: SSDP, initial_var: float) -> None:
        field.initial_var.fill_(math.log(math.expm1(initial_var - field.config.min_var_epsilon)))

    def _get_interval(self) -> float:
        return (self.ray_x_range[1] - self.ray_x_range[0]) / self.num_samples

    def _get_transition_params(
        self,
        ou_drift: float,
        ou_diffusion: float,
    ) -> tuple[float, float]:
        interval = self._get_interval()
        transition_scale = math.exp(-ou_drift * interval)
        transition_var = ou_diffusion / (2.0 * ou_drift) * -math.expm1(-2.0 * ou_drift * interval)
        return transition_scale, transition_var

    def _get_ray_samples(self) -> RaySamples:
        shift = self.phase * self._get_interval()
        ray_bundle = RayBundle(
            origins=torch.tensor([[0.0, 0.0, 0.0]]),
            directions=torch.tensor([[1.0, 0.0, 0.0]]),
            pixel_area=torch.full((1, 1), 0.0),
            camera_indices=torch.full((1, 1), 0),
            nears=torch.full((1, 1), self.ray_x_range[0] + shift),
            fars=torch.full((1, 1), self.ray_x_range[1] + shift),
        )
        ray_bundle = ray_bundle.to(self.device)
        ray_sampler = UniformSampler(
            num_samples=self.num_samples,
            train_stratified=False,
        )
        ray_samples = ray_sampler(ray_bundle)
        return ray_samples

    def __call__(self) -> None:
        field = _create_field(self.min_var_epsilon, self.device)
        field.sdf = self.cuboid_config.get_sdf()

        self._set_initial_var(field, self.initial_var)

        ray_samples = self._get_ray_samples()
        bin_values = _get_bin_values(ray_samples)

        records = []

        for ou_drift, ou_diffusion in tqdm.tqdm(
            iterable=list(itertools.product(self.ou_drifts, self.ou_diffusions)),
            colour=colors.to_hex("dodgerblue"),
            desc="Evaluating approximation errors...",
        ):
            torch.manual_seed(self.random_seed)

            field.ou_drift, field.ou_diffusion = ou_drift, ou_diffusion

            _configure_variant(field, Variant.MC)
            reference_cdf_values, reference_cdf_covariances = _get_reference_cdf_values(
                field=field,
                ray_samples=ray_samples,
                monte_carlo=self.monte_carlo,
                num_mc_chunks=self.num_mc_chunks,
            )

            # NOTE: alpha and gamma of a sampling interval, i.e., what the renderers receive, are recorded as well.
            transition_scale, transition_var = self._get_transition_params(ou_drift, ou_diffusion)
            config = dict(
                ou_drift=ou_drift,
                ou_diffusion=ou_diffusion,
                transition_scale=transition_scale,
                transition_var=transition_var,
                initial_var=self.initial_var,
                num_samples=self.num_samples,
                phase=self.phase,
                monte_carlo=dataclasses.asdict(self.monte_carlo),
                random_seed=self.random_seed,
            )

            # NOTE: Each renderer (the first one) against the reference (the second one).
            for variant in Variant:
                if variant is Variant.MC:
                    continue
                _configure_variant(field, variant)
                cdf_values = _get_predictive_cdf_values(
                    _get_predictive_pmf_values(field, ray_samples)
                )
                metrics = _compute_metrics(
                    bin_values=bin_values,
                    cdf_values_1=cdf_values,
                    cdf_values_2=reference_cdf_values,
                )
                # NOTE: The Monte Carlo errors of the distances are recorded with them (they are not drawn).
                metrics.update(
                    _compute_metric_errors(
                        bin_values=bin_values,
                        cdf_values_1=cdf_values,
                        cdf_values_2=reference_cdf_values,
                        cdf_covariances_2=reference_cdf_covariances,
                    )
                )
                records.append(
                    dict(
                        config=config,
                        variant=variant,
                        metrics={name: values.mean().item() for name, values in metrics.items()},
                    )
                )

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as fp:
            json.dump(records, fp, indent=4)

        loguru.logger.success(f"Saved the approximation errors to <{self.output_file}>.")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        ApproxErrorEvaluator,
        config=(tyro.conf.AvoidSubcommands,),
    )()
