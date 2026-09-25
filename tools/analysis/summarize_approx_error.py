import dataclasses
from pathlib import Path

import loguru
import numpy as np
import pandas as pd
import tyro

from tools.analysis.evaluate_approx_error import Variant
from tools.analysis.plot_approx_error import (
    Metric,
    get_distances,
    load_sweep,
    load_trajectory,
)
from tools.analysis.plot_training_trajectory import TrainingTrajectoryPlotter
from tools.analysis.plot_up_cross_prob import (
    GROUP_NAMES,
    Axis,
    UpCrossProbPlotter,
    get_resolved_flags,
    load_records,
    pivot_curves,
)


@dataclasses.dataclass
class ApproxErrorSummarizer:
    # NOTE: The numbers behind the figures of the three experiments (the tables of the handover), from the data
    # laid out by the launchers (`evaluate_up_cross_prob.sh`, `evaluate_approx_error.sh` and
    # `evaluate_training_trajectory.sh`), followed by the checks on the same data. The thresholds are those of
    # the plotters, so that the numbers are those of the figures.
    data_dir: Path
    output_file: Path

    scene_ids: tuple[str, ...] = ("k1_w0.05", "k1_w0.025", "k3_w0.05", "k3_w0.025")
    phases: tuple[str, ...] = ("0.0", "0.5")
    variants: tuple[Variant, ...] = (Variant.NA, Variant.NA_UP, Variant.BF, Variant.BF_UP)
    # NOTE: <label> <renderer without the approximation> <renderer with it>, as `plot_approx_error.sh`.
    pairs: tuple[tuple[str, Variant, Variant], ...] = (
        ("A   NA - BF", Variant.BF, Variant.NA),
        ("A   NA_UP - BF_UP", Variant.BF_UP, Variant.NA_UP),
        ("B   BF - BF_UP", Variant.BF_UP, Variant.BF),
        ("B   NA - NA_UP", Variant.NA_UP, Variant.NA),
        ("A+B NA - BF_UP", Variant.BF_UP, Variant.NA),
    )

    max_relative_stderr: float = UpCrossProbPlotter.max_relative_stderr
    guide_fit_limit: float = UpCrossProbPlotter.guide_fit_limit
    settled_error: float = TrainingTrajectoryPlotter.settled_error

    def _fit_slope(self, curves: pd.DataFrame) -> float:
        # NOTE: The slope of the median in log-log over the points of the axis below the fit limit of the guide of
        # the plots.
        small = curves.index[curves.index < self.guide_fit_limit]
        return np.polyfit(np.log(small), np.log(curves.loc[small].median(axis=1)), 1)[0]

    def _summarize_sweep(self, axis: Axis) -> tuple[list[str], pd.DataFrame, pd.Series, pd.Series]:
        # NOTE: exp 1 on the axis of a sweep: the population of `plot_up_cross_prob.sh` (the combinations resolved
        # at every point of the sweep).
        records = load_records(self.data_dir / "up_cross_prob" / axis)
        errors = np.abs(records["predictive_prob"] - records["reference_prob"])
        floors = 2.0 * records["reference_prob_stderr"]
        resolved_flags = get_resolved_flags(records, self.max_relative_stderr)
        population = (pivot_curves(records, resolved_flags.astype(float), axis) == 1.0).all()
        curves = pivot_curves(records, errors, axis)
        floor_curves = pivot_curves(records, floors, axis)
        rel_curves = pivot_curves(records, errors / records["reference_prob"], axis)
        columns = [column for column in curves.columns if population[column]]
        curves, floor_curves, rel_curves = (
            curves[columns],
            floor_curves[columns],
            rel_curves[columns],
        )
        worst = records.loc[errors.idxmax()]
        lines = [
            f"===== exp 1 on {axis}, resolved to {self.max_relative_stderr:g} at every point: {curves.shape[1]} combinations",
            f"{axis} | median abs error | 25% | 75% | 95% | median MCSE x 2 | points whose error is below their MCSE x 2",
        ]
        for x in curves.index:
            e, f = curves.loc[x], floor_curves.loc[x]
            lines.append(
                f"{x:8.4g} | {e.median():.2e} | {e.quantile(0.25):.2e} | {e.quantile(0.75):.2e} | {e.quantile(0.95):.2e} | {f.median():.2e} | {int((e < f).sum())}"
            )
        lines += [
            f"slope of the median absolute error over {axis} < {self.guide_fit_limit:g} ({int((curves.index < self.guide_fit_limit).sum())} points): {self._fit_slope(curves):.2f}",
            "median relative error: " + " ".join(f"{v:.2e}" for v in rel_curves.median(axis=1)),
            f"largest absolute error: {errors.max():.3f} at Omega {worst['normalized_quadratic_variation']:.4g} (kappa dt {worst['normalized_sampling_interval']:.3g}), "
            + ", ".join(f"{n.replace('normalized_', '')} {worst[n]:g}" for n in GROUP_NAMES),
        ]
        return lines, records, errors, population

    def _summarize_maps(self, phase: str) -> list[str]:
        # NOTE: exp 2 / exp 3: the distances of the four renderers to Monte Carlo on the (kappa, tau) grid.
        maps = {
            scene_id: load_sweep(self.data_dir / f"maps/phase-{phase}/{scene_id}.json")
            for scene_id in self.scene_ids
        }

        def table(scene_id: str, variant: Variant, name: str) -> pd.Series:
            # NOTE: One value per grid point (tau, kappa), from the pivot of the plots.
            return get_distances(maps[scene_id], variant, name).stack()

        lines = [
            f"===== phase {phase}: grid points {len(table(self.scene_ids[0], Variant.NA, 'hit_prob_1'))}"
        ]
        for metric in Metric:
            lines.append(
                f"--- {metric}: largest distance of each renderer (at kappa, tau; hit prob renderer vs MC there) | median over the grid"
            )
            for scene_id in self.scene_ids:
                for variant in self.variants:
                    d = table(scene_id, variant, metric)
                    tau, kappa = d.idxmax()
                    lines.append(
                        f"{scene_id:10s} {variant:6s} {d.max():.3f} (kappa {kappa:.3g}, tau {tau:.3g}; {table(scene_id, variant, 'hit_prob_1')[d.idxmax()]:.2f} vs {table(scene_id, variant, 'hit_prob_2')[d.idxmax()]:.2f}) | {d.median():.4f}"
                    )
            lines.append(
                f"--- {metric}: with - without the approximation over the grid: median / 75% / 95% / max | min"
            )
            for name, without, with_ in self.pairs:
                for scene_id in self.scene_ids:
                    d = table(scene_id, with_, metric) - table(scene_id, without, metric)
                    lines.append(
                        f"{name:18s} {scene_id:10s} {d.median():+.4f} / {d.quantile(0.75):+.4f} / {d.quantile(0.95):+.4f} / {d.max():+.3f} | {d.min():+.4f}"
                    )
        lines.append(
            "--- hit probability: minimum over the grid of each renderer, and of the MC reference"
        )
        for scene_id in self.scene_ids:
            lines.append(
                f"{scene_id:10s} "
                + " ".join(
                    f"{variant} {table(scene_id, variant, 'hit_prob_1').min():.3f}"
                    for variant in self.variants
                )
                + f" | MC {table(scene_id, Variant.NA, 'hit_prob_2').min():.3f}"
            )
        lines.append(
            "--- Monte Carlo errors of the distances (over the grid and the renderers): standard error max / median | bias max / median"
        )
        for metric in Metric:
            for scene_id in self.scene_ids:
                stderrs = pd.concat(
                    [table(scene_id, variant, f"{metric}_stderr") for variant in self.variants]
                )
                biases = pd.concat(
                    [table(scene_id, variant, f"{metric}_bias") for variant in self.variants]
                )
                lines.append(
                    f"{metric:28s} {scene_id:10s} {stderrs.max():.1e} / {stderrs.median():.1e} | {biases.max():.1e} / {biases.median():.1e}"
                )
        return lines

    def _summarize_learned(self) -> list[str]:
        # NOTE: The error of the renderer used in training along the checkpoints of the learned fields.
        trajectories = {
            (scene_id, variant): load_trajectory(
                self.data_dir / f"trajectories/{scene_id}_{variant}.json"
            )
            for scene_id in self.scene_ids
            for variant in self.variants
        }
        lines = ["===== learned fields: kappa and tau over all the checkpoints (min, max at step)"]
        for (scene_id, variant), t in trajectories.items():
            kappa, tau = t["metrics.ou_drift"], t["metrics.ou_diffusion_root"]
            lines.append(
                f"{scene_id:10s} {variant:6s} kappa [{kappa.min():.3g}, {kappa.max():.3g} @ {int(kappa.idxmax())}] tau [{tau.min():.3g} @ {int(tau.idxmin())}, {tau.max():.3g}]"
            )
        for metric in Metric:
            lines.append(
                f"===== learned fields, {metric}: peak of the median (step) | max q75 | first step from which the median stays < {self.settled_error:g} (the criterion of the inset of the plots) | median, q75 at the last step"
            )
            for (scene_id, variant), t in trajectories.items():
                k = t[f"metrics.{metric}.q50"]
                above = k.index[k >= self.settled_error]
                settle = (
                    int(k.index[k.index > above.max()].min())
                    if len(above) and above.max() < k.index.max()
                    else None
                )
                lines.append(
                    f"{scene_id:10s} {variant:6s} {k.max():.3f} @ {int(k.idxmax()):4d} | {t[f'metrics.{metric}.q75'].max():.3f} | {settle} | {k.iloc[-1]:.1e}, {t[f'metrics.{metric}.q75'].iloc[-1]:.1e}"
                )
        lines.append(
            "===== learned fields: Monte Carlo errors of the distances of a ray at the peak step of the median distance: distance q50 | standard error q50, q75 | bias q50"
        )
        for metric in Metric:
            for (scene_id, variant), t in trajectories.items():
                step = t[f"metrics.{metric}.q50"].idxmax()
                lines.append(
                    f"{metric:28s} {scene_id:10s} {variant:6s} @ {int(step):4d}: {t.loc[step, f'metrics.{metric}.q50']:.1e} | {t.loc[step, f'metrics.{metric}_stderr.q50']:.1e}, {t.loc[step, f'metrics.{metric}_stderr.q75']:.1e} | {t.loc[step, f'metrics.{metric}_bias.q50']:.1e}"
                )
        return lines

    def _summarize_checks(
        self, records: pd.DataFrame, errors: pd.Series, population: pd.Series
    ) -> list[str]:
        # NOTE: The checks on the data of the sweeps of exp 1 (the checks that run their own computations are
        # `check_quadrature.py` and `check_invariance.py`).
        lines = ["===== checks"]
        # NOTE: The test of the orders: the slope per width of the start, for the resolved combinations of
        # the Omega sweep. The absolute error is O(Omega^2) where the density of S(0) is flat over the noise width
        # (sigma large), O(Omega^1.5) where the start lies within the noise width (mu_0 and sigma small), and falls
        # faster where the boundary is far.
        axis = Axis.NORMALIZED_QUADRATIC_VARIATION
        curves = pivot_curves(records, errors, axis)
        lines.append(
            f"--- exp 1, resolved, slope of the median absolute error over Omega < {self.guide_fit_limit:g} per (sigma, mu_0)"
        )
        for sigma in sorted(records["normalized_initial_std"].unique()):
            for mu0 in sorted(records["normalized_initial_mean"].unique()):
                columns = [
                    column
                    for column in curves.columns
                    if population[column] and column[1] == sigma and column[0] == mu0
                ]
                if len(columns) >= 2:
                    lines.append(
                        f"sigma {sigma:4g}, mu_0 {mu0:4g}: n = {len(columns):2d}, slope {self._fit_slope(curves[columns]):.2f}"
                    )
        # NOTE: The check of the paths by the exact down-crossing probability (Eq. 23), over both sweeps.
        raw = pd.concat(
            [
                pd.json_normalize(pd.read_json(path).to_dict("records"))
                for axis in Axis
                for path in sorted((self.data_dir / "up_cross_prob" / axis).glob("*.json"))
            ]
        )
        raw = raw[raw["metrics.empirical_down_cross_prob_stderr"] > 0]
        z = (
            raw["metrics.predictive_down_cross_prob"] - raw["metrics.empirical_down_cross_prob"]
        ) / raw["metrics.empirical_down_cross_prob_stderr"]
        lines.append(
            f"--- down-crossing check: n {len(z)}, mean z {z.mean():.2f}, std z {z.std(ddof=0):.2f}, |z| < 2: {np.mean(np.abs(z) < 2):.3f}"
        )
        return lines

    def __call__(self) -> None:
        lines = []
        sweeps = {axis: self._summarize_sweep(axis) for axis in Axis}
        for sweep_lines, *_ in sweeps.values():
            lines += sweep_lines
        for phase in self.phases:
            lines += self._summarize_maps(phase)
        lines += self._summarize_learned()
        lines += self._summarize_checks(*sweeps[Axis.NORMALIZED_QUADRATIC_VARIATION][1:])

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as fp:
            fp.write("\n".join(lines) + "\n")

        loguru.logger.success(f"Saved the summary to <{self.output_file}>.")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        ApproxErrorSummarizer,
        config=(tyro.conf.AvoidSubcommands,),
    )()
