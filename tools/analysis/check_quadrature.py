import dataclasses
import itertools
import json
import math
from pathlib import Path

import jaxtyping as jt
import loguru
import torch
import tyro
from torch.distributions import Normal

from ssdp.fields.ssdp import SSDP
from ssdp.utils.jaxtyping import jaxtyped
from tools.analysis.evaluate_up_cross_prob import UpCrossProbEvaluator


@jaxtyped()
@torch.no_grad()
def _get_midpoint_probs(
    evaluator: UpCrossProbEvaluator,
    field: SSDP,
    initial_means: jt.Float[torch.Tensor, " C "],
    initial_stds: jt.Float[torch.Tensor, " C "],
    final_means: jt.Float[torch.Tensor, " C "],
    interval: float,
    num_samples: int,
    num_stds: float,
) -> jt.Float[torch.Tensor, " C "]:
    # NOTE: The brute-force reference for the quadrature of `UpCrossProbEvaluator._get_predictive_probs`: the midpoint
    # rule on a uniform grid over the given number of standard deviations of S(0), truncated to S(0) > 0 and
    # normalized by P(S(0) > 0).
    transition_scales, transition_shifts, transition_vars = evaluator._get_transition_params(
        field=field,
        marginal_means=torch.stack([initial_means, final_means], dim=-1),
        interval=interval,
    )
    lower_values = torch.clamp(initial_means - num_stds * initial_stds, min=0.0)
    upper_values = torch.maximum(initial_means + num_stds * initial_stds, lower_values)
    ssdp_samples = (torch.arange(num_samples).to(initial_means) + 0.5) / num_samples
    ssdp_samples = lower_values + (upper_values - lower_values) * ssdp_samples.unsqueeze(-1)
    log_weights = Normal(loc=initial_means, scale=initial_stds).log_prob(ssdp_samples)
    log_weights = log_weights + torch.log((upper_values - lower_values) / num_samples)
    log_weights = log_weights - torch.special.log_ndtr(initial_means / initial_stds)
    log_cross_probs = field._get_log_up_cross_prob(
        ssdp_samples=ssdp_samples,
        transition_scales=transition_scales.squeeze(-1),
        transition_shifts=transition_shifts.squeeze(-1),
        transition_vars=transition_vars.squeeze(-1),
    )
    return torch.sum(torch.exp(log_weights + log_cross_probs), dim=0)


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.flatten().double()
    return dict(
        median=values.median().item(),
        q95=values.quantile(0.95).item(),
        max=values.max().item(),
    )


@dataclasses.dataclass
class QuadratureChecker:
    # NOTE: The check behind the number of nodes of the quadrature of the formula in exp 1
    # (`UpCrossProbEvaluator.num_quad_nodes`): the Gauss-Legendre quadrature over the CDF of S(0) against the midpoint
    # rule on a fine uniform grid (2^18 points over ten standard deviations of S(0); 2^20 points or fourteen standard
    # deviations change it by less than 1e-10), for all the combinations of `UpCrossProbEvaluator` at these
    # normalized sampling intervals (the ends and the middle of both sweeps of `evaluate_up_cross_prob.sh`, the one
    # in the normalized quadratic variation converted). The error is reported relative to the standard error of
    # the Monte Carlo probability at the path count of exp 1.
    output_file: Path

    quad_nodes: tuple[int, ...] = (10, 32, 100, 1000)
    midpoint_samples: int = 1 << 18
    midpoint_stds: float = 10.0
    normalized_sampling_intervals: tuple[float, ...] = tuple(
        sorted({0.1, 1.0, 10.0, *(math.log1p(x) / 2.0 for x in (0.1, 1.0, 10.0))})
    )

    device: str = "cuda"

    def _check_quadrature(self) -> list[dict]:
        evaluator = UpCrossProbEvaluator(output_file=self.output_file, device=self.device)
        field = evaluator._create_field()
        params = torch.tensor(
            list(
                itertools.product(
                    evaluator.initial_means,
                    evaluator.initial_stds,
                    evaluator.linear_coeffs,
                    evaluator.quadratic_coeffs,
                )
            ),
            dtype=torch.float64,
            device=self.device,
        )
        initial_means, initial_stds, linear_coeffs, quadratic_coeffs = params.unbind(-1)

        records = []
        for normalized_sampling_interval in self.normalized_sampling_intervals:
            interval = normalized_sampling_interval / evaluator.ou_drift
            final_means = (
                initial_means + linear_coeffs * interval + quadratic_coeffs * interval**2.0
            )
            reference_probs = _get_midpoint_probs(
                evaluator=evaluator,
                field=field,
                initial_means=initial_means,
                initial_stds=initial_stds,
                final_means=final_means,
                interval=interval,
                num_samples=self.midpoint_samples,
                num_stds=self.midpoint_stds,
            )
            # NOTE: The standard error of the probability at the path count of exp 1, over the combinations with at
            # least one expected crossing (the others are not drawn).
            num_mc_samples = evaluator.monte_carlo.num_mc_samples
            stderrs = torch.sqrt(reference_probs / num_mc_samples)
            flags = reference_probs >= 1.0 / num_mc_samples
            previous_probs = None
            for num_quad_nodes in self.quad_nodes:
                evaluator.num_quad_nodes = num_quad_nodes
                probs, _ = evaluator._get_predictive_probs(
                    field, initial_means, initial_stds, final_means, interval
                )
                record = dict(
                    normalized_sampling_interval=normalized_sampling_interval,
                    num_quad_nodes=num_quad_nodes,
                    num_combinations=int(flags.sum()),
                    error_over_stderr=_quantiles(
                        (probs - reference_probs).abs()[flags] / stderrs[flags]
                    ),
                    max_error=(probs - reference_probs).abs().max().item(),
                )
                if previous_probs is not None:
                    record["change_over_stderr"] = _quantiles(
                        (probs - previous_probs).abs()[flags] / stderrs[flags]
                    )
                previous_probs = probs
                records.append(record)
                loguru.logger.info(
                    f"quadrature: kappa dt {normalized_sampling_interval:.4g}, {num_quad_nodes} nodes: "
                    f"|error| / stderr max {record['error_over_stderr']['max']:.2e}, "
                    f"median {record['error_over_stderr']['median']:.2e}"
                )
        return records

    def __call__(self) -> None:
        results = dict(quadrature=self._check_quadrature())
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as fp:
            json.dump(results, fp, indent=4)
        loguru.logger.success(f"Saved the checks to <{self.output_file}>.")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        QuadratureChecker,
        config=(tyro.conf.AvoidSubcommands,),
    )()
