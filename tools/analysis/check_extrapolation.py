import dataclasses
import json
from pathlib import Path

import loguru
import numpy as np
import pandas as pd
import tyro

from tools.analysis.evaluate_approx_error import MonteCarloConfig, _get_extrapolation_weights
from tools.analysis.evaluate_up_cross_prob import UpCrossProbEvaluator
from tools.analysis.plot_up_cross_prob import (
    Axis,
    UpCrossProbPlotter,
    get_extrapolation_stats,
    get_resolved_flags,
    load_records,
)


@dataclasses.dataclass
class ExtrapolationChecker:
    # NOTE: The check of the residual of the extrapolation of the Monte Carlo reference of exp 1 to infinitely many
    # substeps (`_get_extrapolation_weights`): `UpCrossProbEvaluator` run with four times the substeps of exp 1 and
    # the strides extended by two levels, so that the paths thinned out to the substeps of exp 1 give its extrapolation
    # (the weights depend only on the ratios of the strides) and the finer levels a finer one, on the same paths.
    # The difference between the two is what the residual of exp 1 changes by (7/8 of it if the remainder is
    # O(n^(-3/2))), and has to stay below a third of the standard error of exp 1 (the rule of the design document for
    # the errors that are not drawn). The nesting of the events does not cancel the sampling noise of the difference
    # (the weights amplify the crossings detected between the levels, which differ between the two extrapolations),
    # so the difference is reported with its own standard error (`get_extrapolation_stats`): the noise of the check
    # relative to the standard error of exp 1, and the mean of the signed difference over the combinations with
    # its standard error (the systematic part, at the noise reduced by the number of the combinations). The difference
    # is also compared with the errors that the figure of exp 1 draws (those of the formula, floored at twice the
    # standard error), which bounds what the residual can change in the figure. The ratio of
    # the differences between three consecutive extrapolations and the slope of the detected probabilities against
    # the substeps check the orders behind the extrapolation. The records go to the folder named after the output
    # file (reused when present, so that the points can be evaluated by separate jobs), which gets the statistics;
    # the standard errors of exp 1 are read from its data.
    output_file: Path
    data_dir: Path
    # NOTE: The ends and the middle of the sweep in kappa dt of exp 1 (the top has the largest kappa h).
    normalized_sampling_intervals: tuple[float, ...] = (0.1, 1.0, 10.0)
    # NOTE: The paths of exp 1 with four times its substeps; the coarsest level is that of exp 1.
    monte_carlo: MonteCarloConfig = dataclasses.field(
        default_factory=lambda: MonteCarloConfig(
            num_mc_samples=100_000_000, num_sub_samples=4000, sub_sample_strides=(1, 2, 4, 8, 16)
        ),
    )
    num_mc_chunks: int = 1000
    # NOTE: The population of the figure of exp 1 (`UpCrossProbPlotter`).
    max_relative_stderr: float = UpCrossProbPlotter.max_relative_stderr
    device: str = "cuda"

    def _get_records(self, normalized_sampling_interval: float) -> list[dict]:
        # NOTE: The records of a point are kept next to the output file and reused, so that the points can be
        # evaluated by separate jobs and the statistics recomputed over all of them.
        record_file = self.output_file.with_suffix("") / f"{normalized_sampling_interval:.4g}.json"
        if not record_file.exists():
            UpCrossProbEvaluator(
                output_file=record_file,
                normalized_sampling_intervals=(normalized_sampling_interval,),
                monte_carlo=self.monte_carlo,
                num_mc_chunks=self.num_mc_chunks,
                device=self.device,
            )()
        with open(record_file) as fp:
            return json.load(fp)

    def _get_exp1_records(self, normalized_sampling_interval: float) -> pd.DataFrame:
        # NOTE: The records of exp 1 at the point (the extrapolated reference with its standard error and the value of
        # the formula), in the order of the records of the evaluator (the product of its grids of the groups).
        records = load_records(self.data_dir / Axis.NORMALIZED_SAMPLING_INTERVAL)
        records = records[
            records[Axis.NORMALIZED_SAMPLING_INTERVAL]
            == float(f"{normalized_sampling_interval:.4g}")
        ]
        records = records.set_index(
            [
                "normalized_initial_mean",
                "normalized_initial_std",
                "normalized_linear_coeff",
                "normalized_quadratic_coeff",
            ]
        )
        evaluator = UpCrossProbEvaluator(output_file=self.output_file)
        keys = [
            (mean, std, linear, quadratic)
            for mean in evaluator.initial_means
            for std in evaluator.initial_stds
            for linear in evaluator.linear_coeffs
            for quadratic in evaluator.quadratic_coeffs
        ]
        return records.loc[keys]

    def __call__(self) -> None:
        strides = np.array(self.monte_carlo.sub_sample_strides)
        substeps = self.monte_carlo.num_sub_samples // strides
        # NOTE: The weights of the extrapolations from three consecutive levels, on all the levels (rows, from the
        # finest triple; the last is that of exp 1).
        weights = _get_extrapolation_weights(tuple(strides[:3].tolist())).numpy()
        weights = np.stack(
            [
                np.pad(weights, (start, len(strides) - 3 - start))
                for start in range(len(strides) - 2)
            ]
        )

        results = []
        for normalized_sampling_interval in self.normalized_sampling_intervals:
            records = self._get_records(normalized_sampling_interval)
            exp1_records = self._get_exp1_records(normalized_sampling_interval)
            stderrs = exp1_records["reference_prob_stderr"].to_numpy()
            resolved_flags = get_resolved_flags(exp1_records, self.max_relative_stderr).to_numpy()
            # NOTE: The errors drawn by the figure of exp 1: those of the formula against the reference, floored at
            # twice the standard error of the reference (`UpCrossProbPlotter`).
            drawn_errors = np.maximum(
                np.abs(
                    exp1_records["predictive_prob"] - exp1_records["reference_prob"]
                ).to_numpy(),
                2.0 * stderrs,
            )
            # NOTE: The detected probabilities on the levels (columns, from the finest) and the extrapolations.
            probs = np.array([record["metrics"]["empirical_up_cross_probs"] for record in records])
            num_mc_samples = np.array(
                [record["config"]["monte_carlo"]["num_mc_samples"] for record in records]
            )
            extrapolations = np.stack(
                [get_extrapolation_stats(probs, w, num_mc_samples)[0] for w in weights], axis=1
            )
            fine, mid, prod = extrapolations[:, 0], extrapolations[:, 1], extrapolations[:, -1]
            # NOTE: The difference between the extrapolation of exp 1 and the finest one on the same paths, with
            # its standard error, over the resolved combinations of exp 1 whose reference has a standard error
            # (all the paths cross at every level otherwise, and nothing is extrapolated).
            residuals, residual_stderrs = get_extrapolation_stats(
                probs, weights[-1] - weights[0], num_mc_samples
            )
            flags = resolved_flags & (stderrs > 0.0)
            ratios = residuals[flags] / stderrs[flags]
            noise_ratios = residual_stderrs[flags] / stderrs[flags]
            error_ratios = np.abs(residuals[flags]) / drawn_errors[flags]
            # NOTE: The order of the remainder: 2^(3/2) between consecutive extrapolations if it is O(n^(-3/2)).
            step_ratios = np.abs(prod - mid) / np.abs(mid - fine)
            # NOTE: The order of the leading term: the slope of the detected probabilities against the substeps
            # (-1/2 for the crossings missed inside a substep).
            slopes = np.array(
                [
                    np.polyfit(np.log(substeps), np.log(np.abs(p - f)), 1)[0]
                    if np.all(np.abs(p - f) > 0.0)
                    else np.nan
                    for p, f in zip(probs, fine, strict=True)
                ]
            )
            result = dict(
                normalized_sampling_interval=normalized_sampling_interval,
                num_combinations=int(flags.sum()),
                residual_over_stderr=dict(
                    median=float(np.median(np.abs(ratios))),
                    q95=float(np.quantile(np.abs(ratios), 0.95)),
                    max=float(np.abs(ratios).max()),
                    num_above_third=int((np.abs(ratios) > 1.0 / 3.0).sum()),
                    mean=float(ratios.mean()),
                    mean_stderr=float(np.sqrt(np.mean(noise_ratios**2.0) / flags.sum())),
                ),
                noise_over_stderr=dict(
                    median=float(np.median(noise_ratios)),
                    max=float(noise_ratios.max()),
                ),
                residual_over_drawn_error=dict(
                    median=float(np.median(error_ratios)),
                    max=float(error_ratios.max()),
                ),
                step_ratio=dict(
                    median=float(np.nanmedian(step_ratios[flags])),
                    q25=float(np.nanquantile(step_ratios[flags], 0.25)),
                    q75=float(np.nanquantile(step_ratios[flags], 0.75)),
                ),
                leading_slope=dict(
                    median=float(np.nanmedian(slopes[flags])),
                    q25=float(np.nanquantile(slopes[flags], 0.25)),
                    q75=float(np.nanquantile(slopes[flags], 0.75)),
                ),
            )
            results.append(result)
            loguru.logger.info(
                f"extrapolation: kappa dt {normalized_sampling_interval:g}, "
                f"{result['num_combinations']} combinations: |residual| / stderr median "
                f"{result['residual_over_stderr']['median']:.3f}, 95% {result['residual_over_stderr']['q95']:.3f}, "
                f"max {result['residual_over_stderr']['max']:.3f}, "
                f"above 1/3: {result['residual_over_stderr']['num_above_third']}, "
                f"mean {result['residual_over_stderr']['mean']:+.3f} +- "
                f"{result['residual_over_stderr']['mean_stderr']:.3f}; "
                f"noise / stderr median {result['noise_over_stderr']['median']:.3f}; "
                f"|residual| / drawn error max {result['residual_over_drawn_error']['max']:.4f}; "
                f"step ratio median {result['step_ratio']['median']:.2f} (2.83 for n^-3/2); "
                f"leading slope median {result['leading_slope']['median']:.2f} (-0.5)"
            )

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as fp:
            json.dump(dict(extrapolation=results), fp, indent=4)
        loguru.logger.success(f"Saved the checks to <{self.output_file}>.")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        ExtrapolationChecker,
        config=(tyro.conf.AvoidSubcommands,),
    )()
