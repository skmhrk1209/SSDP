import dataclasses
import enum
import functools
import itertools
import math
from typing import Any, override

import jaxtyping as jt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.distributions.utils import broadcast_all

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from ssdp.utils.jaxtyping import jaxtyped

from .sdf import SDF, SDFConfig


@jaxtyped()
def log1mexp(
    inputs: jt.Float[torch.Tensor, " *B "],
    epsilon: float = 1.0e-6,
) -> jt.Float[torch.Tensor, " *B "]:
    inputs = torch.clamp(inputs, max=-epsilon)
    return torch.where(
        condition=inputs <= math.log(0.5),
        input=torch.log1p(-torch.exp(inputs)),
        other=torch.log(-torch.expm1(inputs)),
    )


@jaxtyped()
def log_ndtr(
    inputs: jt.Float[torch.Tensor, " *B "],
    dtype: torch.dtype = torch.float64,
) -> jt.Float[torch.Tensor, " *B "]:
    # NOTE: `torch.special.log_ndtr` produces NaN for values less than `-46371` in FP32.
    if dtype == torch.float32:
        LOG_NDTR_FP32_MIN = -46371
        inputs = torch.clamp(inputs, min=LOG_NDTR_FP32_MIN)
    return torch.special.log_ndtr(inputs.to(dtype)).to(inputs.dtype)


class Normal(torch.distributions.Normal):
    @jaxtyped()
    def log_cdf(
        self,
        inputs: jt.Float[torch.Tensor, " ... "] | float,
    ) -> jt.Float[torch.Tensor, " ... "]:
        return log_ndtr((inputs - self.mean) / self.stddev)

    @jaxtyped()
    def log_ccdf(
        self,
        inputs: jt.Float[torch.Tensor, " ... "] | float,
    ) -> jt.Float[torch.Tensor, " ... "]:
        return log_ndtr(-(inputs - self.mean) / self.stddev)


class Logistic(torch.distributions.TransformedDistribution):
    def __init__(self, loc: torch.Tensor | float, scale: torch.Tensor | float) -> None:
        loc, scale = broadcast_all(loc, scale)
        base_distribution = torch.distributions.Uniform(
            low=torch.zeros_like(loc),
            high=torch.ones_like(loc),
        )
        transforms = [
            torch.distributions.SigmoidTransform().inv,
            torch.distributions.AffineTransform(loc, scale),
        ]
        super().__init__(base_distribution, transforms)


class TruncatedPositive(torch.distributions.Distribution):
    def __init__(
        self,
        base_distribution: torch.distributions.Distribution,
    ) -> None:
        super().__init__(
            batch_shape=base_distribution.batch_shape,
            event_shape=base_distribution.event_shape,
        )
        self.base_distribution = base_distribution

    @jaxtyped()
    def cdf(
        self,
        inputs: jt.Float[torch.Tensor, " ... "] | float,
        epsilon: float = 1.0e-6,
    ) -> jt.Float[torch.Tensor, " ... "]:
        inputs = torch.clamp(inputs, min=0.0)
        cdf_values_x = self.base_distribution.cdf(inputs)
        cdf_values_0 = self.base_distribution.cdf(0.0)
        cdf_values_0 = torch.clamp(cdf_values_0, max=1.0 - epsilon)
        outputs = (cdf_values_x - cdf_values_0) / (1.0 - cdf_values_0)
        return outputs

    @jaxtyped()
    def icdf(
        self,
        inputs: jt.Float[torch.Tensor, " ... "] | float,
        epsilon: float = 1.0e-6,
    ) -> jt.Float[torch.Tensor, " ... "]:
        cdf_values_0 = self.base_distribution.cdf(0.0)
        cdf_values_x = inputs * (1.0 - cdf_values_0) + cdf_values_0
        cdf_values_x = torch.clamp(cdf_values_x, min=epsilon, max=1.0 - epsilon)
        outputs = self.base_distribution.icdf(cdf_values_x)
        outputs = torch.clamp(outputs, min=0.0)
        return outputs


class QuadratureMode(enum.Enum):
    GL = enum.auto()
    QMC = enum.auto()


@dataclasses.dataclass
class SSDPConfig(SDFConfig):
    _target: type = dataclasses.field(
        default_factory=lambda: SSDP,
    )
    num_layers_ou: int = 4
    num_quad_samples: int = 32
    pos_val_epsilon: float = 1.0e-6
    min_var_epsilon: float = 1.0e-6
    prior_initial_var: float = 1.0e-3
    deterministic_end_ratio: float = 0.5
    up_cross_approx_end_ratio: float = 1.0
    up_cross_anneal_end_ratio: float = 1.0
    survival_approx_end_ratio: float = 1.0
    quadrature_mode: QuadratureMode = QuadratureMode.GL


class SSDP(SDF):
    config: SSDPConfig
    gl_samples: torch.Tensor
    gl_weights: torch.Tensor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        assert (
            self.config.deterministic_end_ratio
            <= self.config.up_cross_approx_end_ratio
            <= self.config.up_cross_anneal_end_ratio
            <= self.config.survival_approx_end_ratio
        )

        self.ou_network = torchvision.ops.MLP(
            in_channels=self.config.geo_feat_dim * 2,
            hidden_channels=[*[self.config.hidden_dim] * self.config.num_layers_ou, 2],
            activation_layer=nn.Softplus,
        )

        prior_initial_var = torch.tensor(self.config.prior_initial_var)
        prior_initial_var = torch.log(torch.expm1(prior_initial_var))
        self.initial_var = nn.Parameter(prior_initial_var)

        if self.config.quadrature_mode is QuadratureMode.GL:
            gl_samples, gl_weights = np.polynomial.legendre.leggauss(self.config.num_quad_samples)
            self.register_buffer("gl_samples", torch.as_tensor(gl_samples, dtype=torch.float32))
            self.register_buffer("gl_weights", torch.as_tensor(gl_weights, dtype=torch.float32))

        elif self.config.quadrature_mode is QuadratureMode.QMC:
            self.sobol_engine = torch.quasirandom.SobolEngine(
                dimension=1,
                scramble=True,
            )

        if self.config.weight_norm:

            def apply_weight_norm(module: nn.Module) -> None:
                if isinstance(module, nn.Linear):
                    nn.utils.weight_norm(module)

            self.ou_network.apply(apply_weight_norm)

    def _is_deterministic(self) -> bool:
        return self.get_progress_ratio() <= self.config.deterministic_end_ratio

    def _is_up_cross_approx(self) -> bool:
        return self.get_progress_ratio() <= self.config.up_cross_approx_end_ratio

    def _is_up_cross_anneal(self) -> bool:
        return self.get_progress_ratio() <= self.config.up_cross_anneal_end_ratio

    def _is_survival_approx(self) -> bool:
        return self.get_progress_ratio() <= self.config.survival_approx_end_ratio

    def _get_initial_var(self) -> jt.Float[torch.Tensor, " "]:
        initial_var = nn.functional.softplus(self.initial_var)
        initial_var = initial_var + self.config.min_var_epsilon
        return initial_var

    def _safe_log(self, inputs: jt.Float[torch.Tensor, " *B "]) -> jt.Float[torch.Tensor, " *B "]:
        return torch.log(torch.clamp(inputs, min=self.config.pos_val_epsilon))

    @jaxtyped()
    def _get_ou_params(
        self,
        geo_features: jt.Float[torch.Tensor, " *R S {self.config.geo_feat_dim} "],
    ) -> tuple[
        jt.Float[torch.Tensor, " *R S-1 "],
        jt.Float[torch.Tensor, " *R S-1 "],
    ]:
        ou_features = torch.cat([geo_features[..., :-1, :], geo_features[..., 1:, :]], dim=-1)
        ou_params = self.ou_network(ou_features)

        ou_drifts, ou_diffusions = torch.unbind(ou_params, dim=-1)

        ou_drifts = nn.functional.softplus(ou_drifts)
        ou_drifts = ou_drifts + self.config.pos_val_epsilon

        ou_diffusions = nn.functional.softplus(ou_diffusions)
        ou_diffusions = ou_diffusions + self.config.pos_val_epsilon

        return ou_drifts, ou_diffusions

    @jaxtyped()
    def _get_transition_params(
        self,
        ou_drifts: jt.Float[torch.Tensor, " *R S "],
        ou_diffusions: jt.Float[torch.Tensor, " *R S "],
        marginal_means: jt.Float[torch.Tensor, " *R S+1 "],
        intervals: jt.Float[torch.Tensor, " *R S "],
    ) -> tuple[
        jt.Float[torch.Tensor, " *R S "],
        jt.Float[torch.Tensor, " *R S "],
        jt.Float[torch.Tensor, " *R S "],
    ]:
        if self._is_deterministic():
            transition_scales = torch.ones_like(ou_drifts)
            transition_vars = torch.zeros_like(ou_diffusions)
        else:
            transition_scales = torch.exp(-ou_drifts * intervals)
            transition_vars = ou_diffusions / (2.0 * ou_drifts)
            transition_vars = transition_vars * -torch.expm1(-2.0 * ou_drifts * intervals)
            transition_vars = torch.clamp(transition_vars, min=self.config.min_var_epsilon)

        transition_shifts = marginal_means[..., 1:] - transition_scales * marginal_means[..., :-1]

        return transition_scales, transition_shifts, transition_vars

    @jaxtyped()
    def _get_log_down_zero_cross_prob(
        self,
        ssdp_samples: jt.Float[torch.Tensor, " S *R "],
        transition_scales: jt.Float[torch.Tensor, " *R "],
        transition_shifts: jt.Float[torch.Tensor, " *R "],
        transition_vars: jt.Float[torch.Tensor, " *R "],
    ) -> jt.Float[torch.Tensor, " S *R "]:
        forward_means = transition_scales * ssdp_samples + transition_shifts
        forward_kernels = Normal(
            loc=forward_means,
            scale=torch.sqrt(transition_vars),
        )
        log_down_zero_cross_probs = forward_kernels.log_cdf(0.0)
        return log_down_zero_cross_probs

    @jaxtyped()
    def _get_log_up_zero_cross_prob(
        self,
        ssdp_samples: jt.Float[torch.Tensor, " S *R "],
        transition_scales: jt.Float[torch.Tensor, " *R "],
        transition_shifts: jt.Float[torch.Tensor, " *R "],
        transition_vars: jt.Float[torch.Tensor, " *R "],
    ) -> jt.Float[torch.Tensor, " S *R "]:
        assert not self._is_up_cross_approx()
        forward_means = transition_scales * ssdp_samples + transition_shifts
        forward_kernels = Normal(
            loc=forward_means,
            scale=torch.sqrt(transition_vars),
        )
        xi = 2.0 * transition_scales / transition_vars * ssdp_samples
        log_up_zero_cross_probs = (
            # Eq. (26) in the paper.
            forward_kernels.log_ccdf(xi * forward_kernels.variance)
            + xi * (xi * forward_kernels.variance / 2.0 - forward_kernels.mean)
        )
        if self._is_up_cross_anneal():
            neg_anneal_ratio = self.get_cosine_anneal_ratio(
                start_ratio=self.config.up_cross_approx_end_ratio,
                end_ratio=self.config.up_cross_anneal_end_ratio,
            )
            log_up_zero_cross_probs = torch.add(
                input=log_up_zero_cross_probs,
                other=math.log(max(neg_anneal_ratio, self.config.pos_val_epsilon)),
            )
        return log_up_zero_cross_probs

    @jaxtyped()
    def _get_log_zero_cross_prob(
        self,
        ssdp_samples: jt.Float[torch.Tensor, " S *R "],
        transition_scales: jt.Float[torch.Tensor, " *R "],
        transition_shifts: jt.Float[torch.Tensor, " *R "],
        transition_vars: jt.Float[torch.Tensor, " *R "],
    ) -> jt.Float[torch.Tensor, " S *R "]:
        log_down_zero_cross_probs = self._get_log_down_zero_cross_prob(
            ssdp_samples=ssdp_samples,
            transition_scales=transition_scales,
            transition_shifts=transition_shifts,
            transition_vars=transition_vars,
        )

        if self._is_up_cross_approx():
            return log_down_zero_cross_probs

        log_up_zero_cross_probs = self._get_log_up_zero_cross_prob(
            ssdp_samples=ssdp_samples,
            transition_scales=transition_scales,
            transition_shifts=transition_shifts,
            transition_vars=transition_vars,
        )

        log_zero_cross_probs = torch.logaddexp(
            input=log_down_zero_cross_probs,
            other=log_up_zero_cross_probs,
        )

        return log_zero_cross_probs

    @jaxtyped()
    def _get_log_snis_weight(
        self,
        ssdp_samples: jt.Float[torch.Tensor, " S *R "],
        transition_scales: jt.Float[torch.Tensor, " *R "],
        transition_shifts: jt.Float[torch.Tensor, " *R "],
        transition_vars: jt.Float[torch.Tensor, " *R "],
        prior_means: jt.Float[torch.Tensor, " *R "],
        prior_vars: jt.Float[torch.Tensor, " *R "],
    ) -> jt.Float[torch.Tensor, " S *R "]:
        assert not self._is_survival_approx()

        backward_vars = 1.0 / (1.0 / prior_vars + transition_scales**2.0 / transition_vars)
        backward_vars = torch.clamp(backward_vars, min=self.config.min_var_epsilon)
        backward_means = backward_vars * (
            prior_means / prior_vars
            + transition_scales / transition_vars * (ssdp_samples - transition_shifts)
        )
        backward_kernels = Normal(
            loc=backward_means,
            scale=torch.sqrt(backward_vars),
        )
        xi = 2.0 * transition_scales / transition_vars * ssdp_samples
        log_up_snis_weights = (
            # Eq. (32) in the paper.
            backward_kernels.log_ccdf(xi * backward_kernels.variance)
            - backward_kernels.log_ccdf(0.0)
            + xi * (xi * backward_kernels.variance / 2.0 - backward_kernels.mean)
        )

        log_snis_weights = log1mexp(
            inputs=log_up_snis_weights,
            epsilon=self.config.pos_val_epsilon,
        )

        return log_snis_weights

    @jaxtyped()
    def _get_marginal_var(
        self,
        transition_scales: jt.Float[torch.Tensor, " *R S "],
        transition_vars: jt.Float[torch.Tensor, " *R S "],
    ) -> jt.Float[torch.Tensor, " *R S+1 "]:
        initial_var = self._get_initial_var()
        initial_vars = initial_var.expand(*transition_scales.shape[:-1])
        marginal_vars = itertools.accumulate(
            iterable=zip(
                torch.unbind(transition_scales, dim=-1),
                torch.unbind(transition_vars, dim=-1),
                strict=True,
            ),
            func=lambda marginal_vars, transition_params: (
                transition_params[0] ** 2.0 * marginal_vars + transition_params[1]
            ),
            initial=initial_vars,
        )
        marginal_vars = torch.stack(list(marginal_vars), dim=-1)
        return marginal_vars

    @jaxtyped()
    def _get_opacity_deterministic(
        self,
        marginal_means: jt.Float[torch.Tensor, " *R S "],
    ) -> jt.Float[torch.Tensor, " *R S-1 "]:
        assert self._is_deterministic()

        marginal_variance = self._get_initial_var()
        marginal_distributions = Normal(
            loc=marginal_means[..., :-1],
            scale=torch.sqrt(marginal_variance),
        )

        proposal_distributions = TruncatedPositive(marginal_distributions)
        opacities = proposal_distributions.cdf(
            inputs=marginal_means[..., :-1] - marginal_means[..., 1:],
            epsilon=self.config.pos_val_epsilon,
        )

        return opacities

    @jaxtyped()
    def _get_opacity_approx(
        self,
        marginal_means: jt.Float[torch.Tensor, " *R S "],
        marginal_vars: jt.Float[torch.Tensor, " *R S "],
        transition_scales: jt.Float[torch.Tensor, " *R S-1 "],
        transition_shifts: jt.Float[torch.Tensor, " *R S-1 "],
        transition_vars: jt.Float[torch.Tensor, " *R S-1 "],
    ) -> jt.Float[torch.Tensor, " *R S-1 "]:
        assert self._is_survival_approx()

        marginal_distributions = Normal(
            loc=marginal_means[..., :-1],
            scale=torch.sqrt(marginal_vars[..., :-1]),
        )

        if self.config.quadrature_mode is QuadratureMode.GL:
            cdf_samples = (self.gl_samples + 1.0) / 2.0
            cdf_weights = self.gl_weights / 2.0
            cdf_samples = cdf_samples.reshape(-1, *[1] * marginal_means.ndim)
            cdf_weights = cdf_weights.reshape(-1, *[1] * marginal_means.ndim)

        elif self.config.quadrature_mode is QuadratureMode.QMC:
            cdf_samples = self.sobol_engine.draw(self.config.num_quad_samples)
            cdf_samples = cdf_samples.reshape(-1, *[1] * marginal_means.ndim)
            cdf_samples = cdf_samples.to(marginal_means)
            cdf_weights = torch.full_like(cdf_samples, 1.0 / self.config.num_quad_samples)

        log_cdf_weights = torch.log(cdf_weights)

        proposal_distributions = TruncatedPositive(marginal_distributions)
        ssdp_samples = proposal_distributions.icdf(cdf_samples)
        log_zero_cross_probs = self._get_log_zero_cross_prob(
            ssdp_samples=ssdp_samples,
            transition_scales=transition_scales,
            transition_shifts=transition_shifts,
            transition_vars=transition_vars,
        )
        log_zero_cross_probs = log_zero_cross_probs + log_cdf_weights
        opacities = torch.sum(torch.exp(log_zero_cross_probs), dim=0)

        return opacities

    @jaxtyped()
    def _get_opacity_bayesian(
        self,
        initial_means: jt.Float[torch.Tensor, " *R "],
        transition_scales: jt.Float[torch.Tensor, " *R S "],
        transition_shifts: jt.Float[torch.Tensor, " *R S "],
        transition_vars: jt.Float[torch.Tensor, " *R S "],
    ) -> jt.Float[torch.Tensor, " *R S "]:
        assert not self._is_survival_approx()

        initial_var = self._get_initial_var()
        initial_vars = initial_var.expand_as(initial_means)

        if self.config.quadrature_mode is QuadratureMode.GL:
            cdf_samples = (self.gl_samples + 1.0) / 2.0
            cdf_weights = self.gl_weights / 2.0
            cdf_samples = cdf_samples.reshape(-1, *[1] * initial_means.ndim)
            cdf_weights = cdf_weights.reshape(-1, *[1] * initial_means.ndim)

        elif self.config.quadrature_mode is QuadratureMode.QMC:
            cdf_samples = self.sobol_engine.draw(self.config.num_quad_samples)
            cdf_samples = cdf_samples.reshape(-1, *[1] * initial_means.ndim)
            cdf_samples = cdf_samples.to(initial_means)
            cdf_weights = torch.full_like(cdf_samples, 1.0 / self.config.num_quad_samples)

        log_cdf_weights = torch.log(cdf_weights)

        prior_means = initial_means
        prior_vars = initial_vars

        opacities_list = []

        @jaxtyped()
        def get_log_snis_weight(
            ssdp_samples: jt.Float[torch.Tensor, " S *R "],
        ) -> jt.Float[torch.Tensor, " S *R "]:
            return torch.zeros_like(ssdp_samples)

        for transition_scales, transition_shifts, transition_vars in zip(
            torch.unbind(transition_scales, dim=-1),
            torch.unbind(transition_shifts, dim=-1),
            torch.unbind(transition_vars, dim=-1),
            strict=True,
        ):
            prior_distributions = Normal(
                loc=prior_means,
                scale=torch.sqrt(prior_vars),
            )
            proposal_distributions = TruncatedPositive(prior_distributions)

            ssdp_samples = proposal_distributions.icdf(cdf_samples)
            ssdp_samples = torch.clamp(ssdp_samples, min=self.config.pos_val_epsilon)

            log_snis_weights = get_log_snis_weight(ssdp_samples)
            log_snis_weights = log_snis_weights + log_cdf_weights
            log_norm_constants = torch.logsumexp(log_snis_weights, dim=0)

            log_zero_cross_probs = self._get_log_zero_cross_prob(
                ssdp_samples=ssdp_samples,
                transition_scales=transition_scales,
                transition_shifts=transition_shifts,
                transition_vars=transition_vars,
            )
            log_zero_cross_probs = log_zero_cross_probs + log_snis_weights
            log_zero_cross_probs = torch.logsumexp(log_zero_cross_probs, dim=0)

            opacities = torch.exp(log_zero_cross_probs - log_norm_constants)
            opacities_list.append(opacities)

            log_ssdp_samples = torch.log(ssdp_samples)

            log_posterior_means = log_ssdp_samples + log_snis_weights
            log_posterior_means = torch.logsumexp(log_posterior_means, dim=0)
            posterior_means = torch.exp(log_posterior_means - log_norm_constants)

            log_posterior_vars = log_ssdp_samples * 2.0 + log_snis_weights
            log_posterior_vars = torch.logsumexp(log_posterior_vars, dim=0)
            posterior_vars = torch.exp(log_posterior_vars - log_norm_constants)
            posterior_vars = posterior_vars - posterior_means**2.0
            posterior_vars = torch.clamp(posterior_vars, min=self.config.min_var_epsilon)

            get_log_snis_weight = functools.partial(
                self._get_log_snis_weight,
                transition_scales=transition_scales,
                transition_shifts=transition_shifts,
                transition_vars=transition_vars,
                prior_means=prior_means,
                prior_vars=prior_vars,
            )

            prior_means = transition_scales * posterior_means + transition_shifts
            prior_vars = transition_scales**2.0 * posterior_vars + transition_vars
            prior_vars = torch.clamp(prior_vars, min=self.config.min_var_epsilon)

        opacities = torch.stack(opacities_list, dim=-1)

        return opacities

    @override
    def get_outputs(
        self,
        ray_samples: RaySamples,
        return_alphas: bool = False,
    ) -> dict[str, torch.Tensor]:
        outputs = super().get_outputs(ray_samples, return_alphas=False)

        if return_alphas:
            ray_positions = ray_samples.frustums.get_start_positions()
            ray_directions = ray_samples.frustums.directions

            ray_intervals: torch.Tensor = ray_samples.deltas
            ray_intervals = torch.clamp(ray_intervals, min=0.0)
            ray_positions = ray_positions + ray_directions * ray_intervals

            marginal_means = outputs[FieldHeadNames.SDF]
            geo_features = outputs["geo_features"]

            extra_ray_positions = ray_positions[..., -1:, :]
            extra_geo_outputs = self.forward_geo_network(extra_ray_positions)
            extra_marginal_means, extra_geo_features = torch.split(
                tensor=extra_geo_outputs,
                split_size_or_sections=(1, self.config.geo_feat_dim),
                dim=-1,
            )

            marginal_means = torch.cat([marginal_means, extra_marginal_means], dim=-2)
            geo_features = torch.cat([geo_features, extra_geo_features], dim=-2)

            marginal_means = marginal_means.squeeze(-1)
            ray_intervals = ray_intervals.squeeze(-1)

            if self._is_deterministic():
                opacities = self._get_opacity_deterministic(marginal_means)

            else:
                ou_drifts, ou_diffusions = self._get_ou_params(geo_features)

                (
                    transition_scales,
                    transition_shifts,
                    transition_vars,
                ) = self._get_transition_params(
                    ou_drifts=ou_drifts,
                    ou_diffusions=ou_diffusions,
                    marginal_means=marginal_means,
                    intervals=ray_intervals,
                )

                if self._is_survival_approx():
                    marginal_vars = self._get_marginal_var(
                        transition_scales=transition_scales,
                        transition_vars=transition_vars,
                    )
                    opacities = self._get_opacity_approx(
                        marginal_means=marginal_means,
                        marginal_vars=marginal_vars,
                        transition_scales=transition_scales,
                        transition_shifts=transition_shifts,
                        transition_vars=transition_vars,
                    )

                else:
                    opacities = self._get_opacity_bayesian(
                        initial_means=marginal_means[..., 0],
                        transition_scales=transition_scales,
                        transition_shifts=transition_shifts,
                        transition_vars=transition_vars,
                    )

            opacities = torch.clamp(opacities, min=0.0, max=1.0)
            opacities = opacities.unsqueeze(-1)

            outputs |= {FieldHeadNames.ALPHA: opacities}

        return outputs
