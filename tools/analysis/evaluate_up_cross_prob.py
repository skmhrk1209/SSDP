import dataclasses
import itertools
import json
import math
from pathlib import Path

import jaxtyping as jt
import loguru
import numpy as np
import torch
import tqdm
import tyro
from matplotlib import colors

from ssdp.fields.ssdp import SSDP
from ssdp.utils.jaxtyping import jaxtyped
from tools.analysis.evaluate_approx_error import (
    MonteCarloConfig,
    Variant,
    _configure_variant,
    _create_field,
)


@dataclasses.dataclass
class UpCrossProbEvaluator:
    # NOTE: Proposition 3.1 alone, on a single sampling interval [0, dt] with constant kappa and tau. With the mean
    # mu(t) = mu_0 + a t + b t^2 and S(0) ~ N(mu_0, sigma^2) truncated to S(0) > 0 (the path is alive at the start of
    # the interval, as in Eq. (35)), the probability of the up-crossings
    #     P(inf_{(0, dt]} S <= 0, S(dt) > 0 | S(0) > 0)
    # is estimated by Monte Carlo and compared with Eq. (26) averaged over the truncated distribution of S(0).
    # The Bayesian filter, the other intervals and the scene do not enter. Both probabilities depend only on kappa dt,
    # mu_0 / sigma_st, sigma / sigma_st, a / (kappa sigma_st) and b / (kappa^2 sigma_st), where
    # sigma_st = tau / sqrt(2 kappa) is the stationary std, so that the quantities below are these groups themselves
    # for the defaults (kappa = 1 and tau^2 = 2).
    output_file: Path
    # NOTE: kappa and tau^2 of the OU process.
    ou_drift: float = 1.0
    ou_diffusion: float = 2.0
    # NOTE: The lengths of the interval, as the normalized sampling interval kappa dt. `evaluate_up_cross_prob.sh`
    # sweeps two variables uniformly in log, a decade on each side of one: kappa dt itself (the default here), and
    # the normalized quadratic variation Omega_i / sigma_st^2 = exp(2 kappa dt) - 1, where Omega_i = Theta(t_{i+1})
    # of Proposition 3.1 is the quadratic variation of the martingale part of the residual over the interval (the
    # variance of the Brownian motion behind the bridge), in which the bound of the interpolation residual
    # (Omega_i^2) and the crossing probability of the bridge are exact; that sweep is passed here converted to
    # kappa dt. Both groups are recorded. In Omega_i / sigma_st^2, the errors reach the resolution of the reference
    # at 1/10 (see `plot_up_cross_prob.py`) and have stopped growing at 10.
    normalized_sampling_intervals: tuple[float, ...] = tuple(
        float(f"{x:.4g}") for x in np.logspace(-1.0, 1.0, 11)
    )
    # NOTE: mu_0, sigma, a and b, all of whose combinations are evaluated on the same paths of the residual, on
    # three decades (10^k for k = -1, 0, 1) around the stationary standard deviation, which cover the values of
    # the trained fields of `evaluate_training_trajectory.sh` (|a| up to about 30 and sigma up to about 60 in these
    # units after training). The shapes of the mean are those of a live path approaching a surface: the mean starts
    # above the boundary (mu_0 > 0) and does not rise (a <= 0, with a = 0 as the mean that does not approach), and
    # curves upward or not at all (b >= 0: a mean that keeps falling never comes back, so that the up-crossings are
    # only those by the noise, which b = 0 covers).
    initial_means: tuple[float, ...] = (0.1, 1.0, 10.0)
    initial_stds: tuple[float, ...] = (0.1, 1.0, 10.0)
    linear_coeffs: tuple[float, ...] = (0.0, -0.1, -1.0, -10.0)
    quadratic_coeffs: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0)
    # NOTE: One interval ratio per job of at most an hour (`MonteCarloConfig`).
    monte_carlo: MonteCarloConfig = dataclasses.field(
        default_factory=lambda: MonteCarloConfig(num_mc_samples=100_000_000),
    )
    # NOTE: The paths are drawn in this many chunks (for the GPU memory only).
    num_mc_chunks: int = 1000
    # NOTE: The number of nodes of the Gauss-Legendre quadrature over the CDF of S(0) (the scheme of
    # `SSDP._get_opacity_approx`; see `_get_predictive_probs`): the smallest power of ten whose error against
    # the midpoint rule on 2^18 points is below a third of the standard error of the Monte Carlo probability for
    # every combination (`check_quadrature.py`: at most 0.0015 of it; 100 nodes reach 2.2 at Omega / sigma_st^2 = 10).
    num_quad_nodes: int = 1000
    # NOTE: The floor of the transition variance of the field (`ApproxErrorEvaluator.min_var_epsilon`), far below
    # every transition variance of the substeps here.
    min_var_epsilon: float = 1.0e-12
    random_seed: int = 42
    device: str = "cuda"

    def _create_field(self) -> SSDP:
        # NOTE: Only the transition kernel, the sampler and the crossing probabilities of the field are used.
        # The renderer with the up-crossing term is configured, for which the field is not in its deterministic phase.
        field = _create_field(self.min_var_epsilon, self.device)
        _configure_variant(field, Variant.BF_UP)
        return field

    @jaxtyped()
    def _get_transition_params(
        self,
        field: SSDP,
        marginal_means: jt.Float[torch.Tensor, " *R S "],
        interval: float,
    ) -> tuple[
        jt.Float[torch.Tensor, " *R S-1 "],
        jt.Float[torch.Tensor, " *R S-1 "],
        jt.Float[torch.Tensor, " *R S-1 "],
    ]:
        return field._get_transition_params(
            ou_drifts=torch.full_like(marginal_means[..., :-1], self.ou_drift),
            ou_diffusions=torch.full_like(marginal_means[..., :-1], self.ou_diffusion),
            marginal_means=marginal_means,
            intervals=torch.full_like(marginal_means[..., :-1], interval),
        )

    @jaxtyped()
    @torch.no_grad()
    def _get_predictive_probs(
        self,
        field: SSDP,
        initial_means: jt.Float[torch.Tensor, " C "],
        initial_stds: jt.Float[torch.Tensor, " C "],
        final_means: jt.Float[torch.Tensor, " C "],
        interval: float,
    ) -> tuple[
        jt.Float[torch.Tensor, " C "],
        jt.Float[torch.Tensor, " C "],
    ]:
        transition_scales, transition_shifts, transition_vars = self._get_transition_params(
            field=field,
            marginal_means=torch.stack([initial_means, final_means], dim=-1),
            interval=interval,
        )
        # NOTE: Eq. (26), and Eq. (23) for the down-crossings, are averaged over S(0) truncated to S(0) > 0 by
        # Gauss-Legendre quadrature in its CDF, as `SSDP._get_opacity_approx` does: the nodes are the quantiles
        # s_k = mu_0 - sigma Phi^{-1}((1 - u_k) P(S(0) > 0)) of the truncated normal distribution at
        # the Gauss-Legendre nodes u_k on [0, 1] (written through the upper tail so that the nodes stay finite where
        # P(S(0) > 0) is tiny; `log_ndtr` keeps that tail where `ndtr` underflows).
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(self.num_quad_nodes)
        cdf_nodes = initial_means.new_tensor((gl_nodes + 1.0) / 2.0).unsqueeze(-1)
        log_weights = torch.log(initial_means.new_tensor(gl_weights / 2.0)).unsqueeze(-1)
        positive_log_masses = torch.special.log_ndtr(initial_means / initial_stds)
        ssdp_samples = initial_means - initial_stds * torch.special.ndtri(
            (1.0 - cdf_nodes) * torch.exp(positive_log_masses)
        )
        cross_probs = [
            torch.sum(
                torch.exp(
                    log_weights
                    + get_log_cross_prob(
                        ssdp_samples=ssdp_samples,
                        transition_scales=transition_scales.squeeze(-1),
                        transition_shifts=transition_shifts.squeeze(-1),
                        transition_vars=transition_vars.squeeze(-1),
                    )
                ),
                dim=0,
            )
            for get_log_cross_prob in (
                field._get_log_up_cross_prob,
                field._get_log_down_cross_prob,
            )
        ]
        return tuple(cross_probs)

    @jaxtyped()
    @torch.no_grad()
    def _get_empirical_probs(
        self,
        field: SSDP,
        interval: float,
    ) -> tuple[
        jt.Float[torch.Tensor, " N M D L Q T "],
        jt.Float[torch.Tensor, " N M D L Q "],
    ]:
        kwargs = dict(dtype=torch.float64, device=self.device)
        sub_values = torch.linspace(0.0, interval, self.monte_carlo.num_sub_samples + 1, **kwargs)

        # NOTE: The paths of the residual with a zero mean. The decay of the initial value is that of the substeps.
        transition_scales, transition_shifts, transition_vars = self._get_transition_params(
            field=field,
            marginal_means=torch.zeros_like(sub_values),
            interval=interval / self.monte_carlo.num_sub_samples,
        )
        decays = torch.cumprod(
            torch.nn.functional.pad(transition_scales, (1, 0), value=1.0), dim=-1
        )

        up_cross_probs, down_cross_probs = [], []

        assert not self.monte_carlo.num_mc_samples % self.num_mc_chunks
        for _ in range(self.num_mc_chunks):
            mc_samples = field.get_mc_samples(
                initial_means=sub_values.new_zeros(()),
                transition_scales=transition_scales,
                transition_shifts=transition_shifts,
                transition_vars=transition_vars,
                num_mc_samples=self.monte_carlo.num_mc_samples // self.num_mc_chunks,
            )
            # NOTE: R(t) = exp(-kappa t) R(0) + Z(t), where the OU process Z from zero does not depend on R(0), so that
            # the same Z is shared by all the combinations of mu_0, sigma, a and b. S(0) = mu_0 + R(0) is drawn from
            # the normal distribution truncated to S(0) > 0 through the same uniform variable for every (mu_0, sigma).
            mc_samples = mc_samples - decays * mc_samples[..., :1]
            uniform_samples = torch.rand_like(mc_samples[..., 0])

            chunk_up_cross_probs = mc_samples.new_zeros(
                len(self.initial_means),
                len(self.initial_stds),
                len(self.linear_coeffs),
                len(self.quadratic_coeffs),
                len(self.monte_carlo.sub_sample_strides),
            )
            chunk_down_cross_probs = chunk_up_cross_probs.new_zeros(
                chunk_up_cross_probs.shape[:-1]
            )

            for (mean_index, initial_mean), (std_index, initial_std) in itertools.product(
                enumerate(self.initial_means), enumerate(self.initial_stds)
            ):
                # NOTE: The quantiles of the truncated normal distribution, through the upper tail as in
                # `_get_predictive_probs`.
                positive_mass = torch.special.ndtr(
                    sub_values.new_tensor(initial_mean / initial_std)
                )
                initial_residuals = -initial_std * torch.special.ndtri(
                    (1.0 - uniform_samples) * positive_mass
                )
                residuals = mc_samples + decays * initial_residuals.unsqueeze(-1)

                for (linear_index, linear_coeff), (
                    quadratic_index,
                    quadratic_coeff,
                ) in itertools.product(
                    enumerate(self.linear_coeffs),
                    enumerate(self.quadratic_coeffs),
                ):
                    # NOTE: The zero-crossings are detected only from the signs at the (thinned-out) substeps.
                    paths = (
                        initial_mean
                        + residuals
                        + (linear_coeff * sub_values + quadratic_coeff * sub_values**2.0)
                    )
                    final_flags = paths[..., -1] > 0.0
                    for stride_index, stride in enumerate(self.monte_carlo.sub_sample_strides):
                        cross_flags = torch.amin(paths[..., ::stride], dim=-1) <= 0.0
                        chunk_up_cross_probs[
                            mean_index, std_index, linear_index, quadratic_index, stride_index
                        ] = torch.mean((cross_flags & final_flags).to(paths), dim=-1)
                    chunk_down_cross_probs[
                        mean_index, std_index, linear_index, quadratic_index
                    ] = torch.mean((~final_flags).to(paths), dim=-1)

            up_cross_probs.append(chunk_up_cross_probs)
            down_cross_probs.append(chunk_down_cross_probs)

        return torch.stack(up_cross_probs, dim=0), torch.stack(down_cross_probs, dim=0)

    def __call__(self) -> None:
        field = self._create_field()
        stationary_std = math.sqrt(self.ou_diffusion / (2.0 * self.ou_drift))

        records = []

        for normalized_sampling_interval in tqdm.tqdm(
            iterable=self.normalized_sampling_intervals,
            colour=colors.to_hex("dodgerblue"),
            desc="Evaluating up-crossing probabilities...",
        ):
            torch.manual_seed(self.random_seed)

            interval = normalized_sampling_interval / self.ou_drift

            # NOTE: The combinations are flattened in the order of `params` below. The detection from the signs at
            # the substeps misses the zero-crossings inside a substep, so that the probabilities on the substeps
            # thinned out by the strides are extrapolated to infinitely many substeps in `plot_up_cross_prob.py`.
            up_cross_probs, down_cross_probs = self._get_empirical_probs(field, interval)
            up_cross_probs, down_cross_probs = (
                up_cross_probs.flatten(1, 4),
                down_cross_probs.flatten(1, 4),
            )

            params = list(
                itertools.product(
                    self.initial_means,
                    self.initial_stds,
                    self.linear_coeffs,
                    self.quadratic_coeffs,
                )
            )
            initial_means, initial_stds, linear_coeffs, quadratic_coeffs = (
                up_cross_probs.new_tensor(params).unbind(-1)
            )
            predictive_up_cross_probs, predictive_down_cross_probs = self._get_predictive_probs(
                field=field,
                initial_means=initial_means,
                initial_stds=initial_stds,
                final_means=initial_means
                + linear_coeffs * interval
                + quadratic_coeffs * interval**2.0,
                interval=interval,
            )

            for index, (initial_mean, initial_std, linear_coeff, quadratic_coeff) in enumerate(
                params
            ):
                records.append(
                    dict(
                        config=dict(
                            ou_drift=self.ou_drift,
                            ou_diffusion=self.ou_diffusion,
                            interval=interval,
                            initial_mean=initial_mean,
                            initial_std=initial_std,
                            linear_coeff=linear_coeff,
                            quadratic_coeff=quadratic_coeff,
                            # NOTE: The dimensionless groups (the normalized quadratic variation is derived from
                            # the swept normalized sampling interval).
                            normalized_sampling_interval=normalized_sampling_interval,
                            normalized_quadratic_variation=math.expm1(
                                2.0 * normalized_sampling_interval
                            ),
                            normalized_initial_mean=initial_mean / stationary_std,
                            normalized_initial_std=initial_std / stationary_std,
                            normalized_linear_coeff=linear_coeff
                            / (self.ou_drift * stationary_std),
                            normalized_quadratic_coeff=quadratic_coeff
                            / (self.ou_drift**2.0 * stationary_std),
                            monte_carlo=dataclasses.asdict(self.monte_carlo),
                            random_seed=self.random_seed,
                        ),
                        metrics=dict(
                            predictive_up_cross_prob=predictive_up_cross_probs[index].item(),
                            predictive_down_cross_prob=predictive_down_cross_probs[index].item(),
                            # NOTE: One probability per stride of `monte_carlo.sub_sample_strides`.
                            empirical_up_cross_probs=up_cross_probs[:, index].mean(dim=0).tolist(),
                            # NOTE: The down-crossings need no substep and Eq. (23) is exact, which checks the paths.
                            # The standard error is that of the mean over the chunks.
                            empirical_down_cross_prob=down_cross_probs[:, index]
                            .mean(dim=0)
                            .item(),
                            empirical_down_cross_prob_stderr=(
                                torch.std(down_cross_probs[:, index], dim=0)
                                / math.sqrt(self.num_mc_chunks)
                            ).item(),
                        ),
                    )
                )

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as fp:
            json.dump(records, fp, indent=4)

        loguru.logger.success(f"Saved the up-crossing probabilities to <{self.output_file}>.")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        UpCrossProbEvaluator,
        config=(tyro.conf.AvoidSubcommands,),
    )()
