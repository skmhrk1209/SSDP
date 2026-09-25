import dataclasses
import enum
import json
from pathlib import Path

import loguru
import numpy as np
import pandas as pd
import tyro

from tools.analysis.evaluate_approx_error import _get_extrapolation_weights
from tools.analysis.plot_approx_error import configure_style, plt, save_figure

GROUP_NAMES = [
    "normalized_initial_mean",
    "normalized_initial_std",
    "normalized_linear_coeff",
    "normalized_quadratic_coeff",
]
# NOTE: The significant digits of the values of the sweeps (`evaluate_up_cross_prob.sh`).
NUM_AXIS_DIGITS = 4


class Axis(enum.StrEnum):
    # NOTE: The length of the interval on the horizontal axis, each swept by `evaluate_up_cross_prob.sh` and recorded
    # under its name by `evaluate_up_cross_prob.py`: the normalized quadratic variation Omega_i / sigma_st^2 (the
    # variable in which the interpolation residual of Proposition 3.1 is O(Omega_i^2)), or the normalized sampling
    # interval kappa dt (the same information, Omega_i / sigma_st^2 = exp(2 kappa dt) - 1).
    NORMALIZED_QUADRATIC_VARIATION = enum.auto()
    NORMALIZED_SAMPLING_INTERVAL = enum.auto()

    @property
    def symbol(self) -> str:
        return dict(
            [
                (Axis.NORMALIZED_QUADRATIC_VARIATION, r"\widetilde{\Omega}"),
                (Axis.NORMALIZED_SAMPLING_INTERVAL, r"\widetilde{\Delta t}"),
            ]
        )[self]

    @property
    def label(self) -> str:
        name = dict(
            [
                (Axis.NORMALIZED_QUADRATIC_VARIATION, "Normalized Quadratic Variation"),
                (Axis.NORMALIZED_SAMPLING_INTERVAL, "Normalized Sampling Interval"),
            ]
        )[self]
        return rf"{name} ${self.symbol}$"


class Error(enum.StrEnum):
    # NOTE: The absolute error of the probability of the up-crossings, and the error relative to the probability of
    # the reference. Only the paths starting within the noise width sqrt(Omega_i) of the boundary can cross, and
    # the residual of Proposition 3.1 shifts them by O(Omega_i^2), i.e., by O(Omega_i^1.5) of that width: the relative
    # error is O(Omega_i^1.5). Where the density of S(0) is flat over that width, the probability itself is
    # O(Omega_i^0.5) and the absolute error O(Omega_i^2), the order of the residual (an estimate from the residual,
    # not a statement of the paper, which states the order of the residual). The guides show these powers of
    # the horizontal axis (the same powers of kappa dt, to which Omega_i / sigma_st^2 is proportional to first order).
    ABSOLUTE = enum.auto()
    RELATIVE = enum.auto()

    @property
    def label(self) -> str:
        return dict(
            [
                (Error.ABSOLUTE, "Absolute Error"),
                (Error.RELATIVE, "Relative Error"),
            ]
        )[self]

    @property
    def file_id(self) -> str:
        return dict([(Error.ABSOLUTE, "abs"), (Error.RELATIVE, "rel")])[self]

    @property
    def order(self) -> float:
        return dict([(Error.ABSOLUTE, 2.0), (Error.RELATIVE, 1.5)])[self]


def load_records(input_dir: Path) -> pd.DataFrame:
    # NOTE: The records of `evaluate_up_cross_prob.py`, with the random seeds pooled by the numbers of their paths.
    records = []
    for input_file in sorted(input_dir.glob("*.json")):
        with open(input_file) as fp:
            records.extend(json.load(fp))
    records = pd.json_normalize(records)
    records.columns = [column.split(".")[-1] for column in records.columns]
    # NOTE: The values of the axes are those of the sweeps (`evaluate_up_cross_prob.sh`); the axis that
    # the evaluator derives from the swept one carries rounding errors otherwise.
    for axis in Axis:
        records[axis] = records[axis].map(lambda value: float(f"{value:.{NUM_AXIS_DIGITS}g}"))
    [sub_sample_strides] = {tuple(strides) for strides in records["sub_sample_strides"]}

    # NOTE: The probabilities of the zero-crossings detected on the substeps thinned out by each stride (columns),
    # pooled over the random seeds. Both axes of `Axis` are kept (they are in one-to-one correspondence).
    names = [*Axis, *GROUP_NAMES]
    counts = pd.DataFrame(np.stack(records["empirical_up_cross_probs"])).mul(
        records["num_mc_samples"], axis=0
    )
    counts = pd.concat([records[[*names, "num_mc_samples"]], counts], axis=1).groupby(names).sum()
    probs = counts.drop(columns="num_mc_samples").div(counts["num_mc_samples"], axis=0).to_numpy()

    # NOTE: The extrapolation to infinitely many substeps (`_get_extrapolation_weights`). A zero-crossing detected on
    # thinned-out substeps is also detected on finer ones (the strides are multiples of each other), so that
    # the extrapolation is a linear combination of the indicators of exclusive events, which gives its variance.
    weights = _get_extrapolation_weights(sub_sample_strides).numpy()
    event_probs = probs - np.pad(probs[:, 1:], ((0, 0), (0, 1)))
    event_coeffs = np.cumsum(weights)

    records = (
        records.groupby(names)[["predictive_up_cross_prob"]]
        .first()
        .rename(columns=dict(predictive_up_cross_prob="predictive_prob"))
    )
    records["reference_prob"] = event_probs @ event_coeffs
    records["reference_prob_stderr"] = np.sqrt(
        np.clip(event_probs @ event_coeffs**2.0 - records["reference_prob"] ** 2.0, 0.0, None)
        / counts["num_mc_samples"]
    )
    return records.reset_index()


def get_resolved_flags(records: pd.DataFrame, max_relative_stderr: float) -> pd.Series:
    # NOTE: A point is resolved where the reference has a positive probability (a zero probability has a zero
    # standard error) whose standard error is at most the given fraction of it, i.e., at least
    # 1 / max_relative_stderr^2 crossings were observed. An infinite fraction resolves every point.
    return (
        (records["reference_prob"] > 0.0)
        & (records["reference_prob_stderr"] <= max_relative_stderr * records["reference_prob"])
    ) | np.isinf(max_relative_stderr)


def pivot_curves(records: pd.DataFrame, values: pd.Series, axis: Axis) -> pd.DataFrame:
    # NOTE: One column per combination of the dimensionless groups, against the horizontal axis.
    curves = values.to_frame("value").join(records[[axis, *GROUP_NAMES]])
    return curves.pivot(index=axis, columns=GROUP_NAMES, values="value")


@dataclasses.dataclass
class UpCrossProbPlotter:
    # NOTE: Error of the probability of the up-crossings of Eq. (26) on a single interval against the Monte Carlo
    # reference of `evaluate_up_cross_prob.py`, as a function of the length of the interval: one figure per error
    # (`Error`) and per axis (`Axis`, read from `<input_dir>/<axis>/` and saved in `<output_dir>/axis-<axis>/`), with
    # the vertical range of an error common to its axes.
    input_dir: Path
    output_dir: Path
    axes: tuple[Axis, ...] = tuple(Axis)
    # NOTE: The color of the up-crossings in the figures of the paper.
    color: str = "dodgerblue"
    # NOTE: The combinations drawn are those resolved (`get_resolved_flags`) at every point of the sweep: a criterion
    # on the reference alone, never on the error, which would keep its upward fluctuations.
    max_relative_stderr: float = 0.01
    # NOTE: The guide is fitted to the medians at the points of the axis below this value, i.e., in the decade
    # where the expansion of the interpolation residual holds (`summarize_approx_error.py` fits the slope over the same
    # points).
    guide_fit_limit: float = 1.0
    # NOTE: The quantiles over the combinations: the band of the quartiles and the wider band, whose ends set
    # the vertical range.
    quantiles: tuple[float, float, float, float, float] = (0.05, 0.25, 0.5, 0.75, 0.95)

    # TMLR layout dimensions (from tmlr.sty)
    text_width: float = 6.5

    width_ratio: float = 0.5
    aspect_ratio: float = 0.8
    font_size: float = 8.0

    def _get_figure_size(self) -> tuple[float, float]:
        width = self.text_width * self.width_ratio
        height = width * self.aspect_ratio
        return (width, height)

    def _get_curves(self, axis: Axis) -> dict[Error, tuple[pd.DataFrame, pd.DataFrame]]:
        # NOTE: The errors of the drawn combinations against the axis, and the sampling errors of the reference (two
        # standard errors) as their floors. The discretization left after the extrapolation is O(n^(-3/2)), i.e.,
        # smaller than what the extrapolation removes, and is not included.
        records = load_records(self.input_dir / axis)
        resolved_flags = get_resolved_flags(records, self.max_relative_stderr)
        population = (pivot_curves(records, resolved_flags.astype(float), axis) == 1.0).all()
        loguru.logger.info(
            f"{axis}: {population.sum()} of {len(population)} combinations are drawn."
        )

        errors = np.abs(records["predictive_prob"] - records["reference_prob"])
        floors = 2.0 * records["reference_prob_stderr"]
        scales = dict(
            [
                (Error.ABSOLUTE, 1.0),
                (Error.RELATIVE, records["reference_prob"].where(records["reference_prob"] > 0.0)),
            ]
        )
        curves = {}
        for error, scale in scales.items():
            error_curves = pivot_curves(records, errors / scale, axis)
            floor_curves = pivot_curves(records, floors / scale, axis)
            columns = [column for column in error_curves.columns if population[column]]
            curves[error] = (error_curves[columns], floor_curves[columns])
        return curves

    def _plot(
        self,
        axis: Axis,
        error: Error,
        curves: pd.DataFrame,
        floors: pd.DataFrame,
        value_range: tuple[float, float],
        output_file: Path,
    ) -> None:
        # NOTE: One faint line per combination (columns) against the length of the interval (rows), with the median
        # and the quantiles over the combinations, a guide proportional to the power of the axis of the error, and
        # the sampling error of the reference (its median, filled from the bottom) below which the values are
        # dominated by its noise.
        fig, ax = plt.subplots(figsize=self._get_figure_size())

        quantiles = curves.quantile(self.quantiles, axis=1).T
        floor = floors.median(axis=1)
        min_value, max_value = value_range

        ax.plot(
            curves.index,
            curves.clip(lower=min_value),
            color=self.color,
            linewidth=0.5,
            alpha=0.125,
        )
        (median_line,) = ax.plot(
            curves.index,
            quantiles[0.5],
            color=self.color,
            linewidth=1.5,
            marker="o",
            markersize=5.0,
            markerfacecolor="white",
        )
        quartile_band = ax.fill_between(
            curves.index,
            quantiles[0.25].clip(lower=min_value),
            quantiles[0.75],
            color=self.color,
            alpha=0.25,
            linewidth=0.0,
        )
        ax.fill_between(
            curves.index,
            quantiles[0.05].clip(lower=min_value),
            quantiles[0.95],
            color=self.color,
            alpha=0.125,
            linewidth=0.0,
        )
        # NOTE: The guide has the slope of the order of the error, and the height fitted in log by least squares to
        # the medians at the points of the axis below `guide_fit_limit`.
        guide_values = curves.index.to_numpy()
        fit_flags = guide_values < self.guide_fit_limit
        log_height = np.mean(
            np.log(quantiles[0.5].to_numpy()[fit_flags])
            - error.order * np.log(guide_values[fit_flags])
        )
        (guide_line,) = ax.plot(
            guide_values,
            np.exp(log_height) * guide_values**error.order,
            color="black",
            linestyle="--",
        )
        floor_band = ax.fill_between(
            curves.index,
            min_value,
            floor.clip(lower=min_value),
            color="gray",
            alpha=0.25,
            linewidth=0.0,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(curves.index.min(), curves.index.max())
        ax.set_ylim(min_value, max_value)
        ax.set_xlabel(axis.label)
        ax.set_ylabel(error.label)
        ax.legend(
            handles=[(quartile_band, median_line), guide_line, floor_band],
            labels=[
                "Proposition 3.1",
                rf"$\propto {axis.symbol}^{{{error.order:g}}}$",
                r"MCSE ($\times 2$)",
            ],
            loc="upper left",
        )

        save_figure(fig, output_file)

    def __call__(self) -> None:
        configure_style(self.font_size)

        curves = {axis: self._get_curves(axis) for axis in self.axes}

        # NOTE: The vertical range of an error, common to its axes: the powers of ten enclosing the smallest lower
        # quantile and the largest upper quantile over the axes.
        for error in Error:
            quantiles = [
                curves[axis][error][0].quantile([self.quantiles[0], self.quantiles[-1]], axis=1)
                for axis in self.axes
            ]
            value_range = (
                10.0 ** np.floor(np.log10(min(q.iloc[0].min() for q in quantiles))),
                10.0 ** np.ceil(np.log10(max(q.iloc[1].max() for q in quantiles))),
            )
            loguru.logger.info(
                f"{error}: vertical range {value_range[0]:.3g} to {value_range[1]:.3g}."
            )
            for axis in self.axes:
                self._plot(
                    axis=axis,
                    error=error,
                    curves=curves[axis][error][0],
                    floors=curves[axis][error][1],
                    value_range=value_range,
                    output_file=(
                        self.output_dir
                        / f"axis-{axis}"
                        / f"analytic_up_cross_prob_{error.file_id}_error_plot_vs_MC.pdf"
                    ),
                )


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        UpCrossProbPlotter,
        config=(tyro.conf.AvoidSubcommands,),
    )()
