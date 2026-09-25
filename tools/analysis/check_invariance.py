import dataclasses
import json
import math
from pathlib import Path

import loguru
import numpy as np
import tyro

from tools.analysis.evaluate_approx_error import MonteCarloConfig
from tools.analysis.evaluate_up_cross_prob import UpCrossProbEvaluator


@dataclasses.dataclass
class InvarianceChecker:
    # NOTE: The check of the reduction of exp 1 to the dimensionless groups: `UpCrossProbEvaluator` run at different
    # (kappa, tau^2) with the same groups kappa dt, mu_0 / sigma_st, sigma / sigma_st, a / (kappa sigma_st) and
    # b / (kappa^2 sigma_st) has to give the same probabilities as the standard form (kappa = 1 and tau^2 = 2, the
    # first setting) up to rounding errors, with the same random seed. The records of each setting are written to
    # the folder named after the output file, which gets the largest differences to the standard form.
    output_file: Path
    # NOTE: (kappa, tau^2) of the settings, the standard form first.
    ou_params: tuple[tuple[float, float], ...] = (
        (1.0, 2.0),
        (10.0, 0.25),
        (100.0, 0.02),
        (0.1, 8.0),
    )
    # NOTE: The groups: the ends and the middle of the sweep in kappa dt, and a few values of the others including
    # the starts at and below the boundary and both signs of the coefficients.
    normalized_sampling_intervals: tuple[float, ...] = (0.1, 1.0, 10.0)
    normalized_initial_means: tuple[float, ...] = (-1.0, 0.0, 1.0)
    normalized_initial_stds: tuple[float, ...] = (0.1, 1.0)
    normalized_linear_coeffs: tuple[float, ...] = (-1.0, 1.0)
    normalized_quadratic_coeffs: tuple[float, ...] = (-1.0, 1.0)
    # NOTE: Fewer paths than exp 1 (the differences are rounding errors, not sampling errors).
    monte_carlo: MonteCarloConfig = dataclasses.field(
        default_factory=lambda: MonteCarloConfig(num_mc_samples=1_000_000),
    )
    num_mc_chunks: int = 10
    device: str = "cuda"

    def _get_records(self, ou_drift: float, ou_diffusion: float) -> list[dict]:
        stationary_std = math.sqrt(ou_diffusion / (2.0 * ou_drift))
        evaluator = UpCrossProbEvaluator(
            output_file=self.output_file.with_suffix("")
            / f"k-{ou_drift:g}_d-{ou_diffusion:g}.json",
            ou_drift=ou_drift,
            ou_diffusion=ou_diffusion,
            normalized_sampling_intervals=self.normalized_sampling_intervals,
            initial_means=tuple(x * stationary_std for x in self.normalized_initial_means),
            initial_stds=tuple(x * stationary_std for x in self.normalized_initial_stds),
            linear_coeffs=tuple(
                x * ou_drift * stationary_std for x in self.normalized_linear_coeffs
            ),
            quadratic_coeffs=tuple(
                x * ou_drift**2.0 * stationary_std for x in self.normalized_quadratic_coeffs
            ),
            monte_carlo=self.monte_carlo,
            num_mc_chunks=self.num_mc_chunks,
            device=self.device,
        )
        evaluator()
        with open(evaluator.output_file) as fp:
            return json.load(fp)

    def __call__(self) -> None:
        records = [self._get_records(*ou_params) for ou_params in self.ou_params]

        results = []
        for (ou_drift, ou_diffusion), setting in zip(self.ou_params, records, strict=True):
            pairs = list(zip(setting, records[0], strict=True))
            result = dict(
                ou_drift=ou_drift,
                ou_diffusion=ou_diffusion,
                num_records=len(setting),
                max_formula_diff=max(
                    abs(
                        a["metrics"]["predictive_up_cross_prob"]
                        - b["metrics"]["predictive_up_cross_prob"]
                    )
                    for a, b in pairs
                ),
                max_mc_diff=max(
                    float(
                        np.abs(
                            np.array(a["metrics"]["empirical_up_cross_probs"])
                            - np.array(b["metrics"]["empirical_up_cross_probs"])
                        ).max()
                    )
                    for a, b in pairs
                ),
            )
            results.append(result)
            loguru.logger.info(
                f"invariance: kappa {ou_drift:g}, tau^2 {ou_diffusion:g}: {result['num_records']} records, "
                f"max |formula diff| {result['max_formula_diff']:.1e}, max |MC diff| {result['max_mc_diff']:.1e}"
            )

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as fp:
            json.dump(dict(invariance=results), fp, indent=4)
        loguru.logger.success(f"Saved the checks to <{self.output_file}>.")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        InvarianceChecker,
        config=(tyro.conf.AvoidSubcommands,),
    )()
