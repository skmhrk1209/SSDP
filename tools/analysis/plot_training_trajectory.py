import dataclasses
from pathlib import Path

import pandas as pd
import tyro
from matplotlib import ticker

from tools.analysis.evaluate_approx_error import Variant
from tools.analysis.plot_approx_error import (
    VARIANT_COLORS,
    Metric,
    configure_style,
    get_variant_id,
    get_variant_name,
    load_trajectory,
    plt,
    save_figure,
    snap_range,
)


def get_max_error(trajectory_files: tuple[Path, ...], metric: Metric) -> float:
    # NOTE: The top of the vertical axis common to all the plots of a metric: the largest upper quartile over
    # the given runs and their checkpoints.
    return max(load_trajectory(f)[f"metrics.{metric}.q75"].max() for f in trajectory_files)


@dataclasses.dataclass
class TrainingTrajectoryPlotter:
    # NOTE: `<trajectory_dir>/<scene_id>_<variant>.json` from `evaluate_training_trajectory.py`.
    trajectory_dir: Path
    # NOTE: For each scene, the plot is saved in the corresponding directory, next to the maps of
    # `plot_approx_error.py` and with the same size.
    scene_ids: tuple[str, ...]
    output_dirs: tuple[Path, ...]
    # NOTE: The vertical range (`get_max_error` over all the runs of the metric, computed by `plot_approx_error.sh`),
    # common to all the figures of the metric.
    max_error: float
    # NOTE: The renderer without the approximation of interest, and the one with it.
    variants: tuple[Variant, Variant] = (Variant.BF_UP, Variant.NA)
    metric: Metric = Metric.CONDITIONAL_CRAMER_DISTANCE

    # TMLR layout dimensions (from tmlr.sty)
    text_width: float = 6.5

    width_ratio: float = 0.5
    aspect_ratio: float = 0.8
    font_size: float = 8.0

    # NOTE: The steps until the errors of all the renderers have settled below this, and one more checkpoint, are
    # magnified in an inset, whose vertical range ends at the largest 75% point in it.
    settled_error: float = 1.0e-3
    inset_bounds: tuple[float, float, float, float] = (0.25, 0.2, 0.7, 0.55)

    def _get_figure_size(self) -> tuple[float, float]:
        width = self.text_width * self.width_ratio
        height = width * self.aspect_ratio
        return (width, height)

    def _load_trajectories(
        self, scene_id: str, variants: tuple[Variant, ...]
    ) -> dict[Variant, pd.DataFrame]:
        return {
            variant: load_trajectory(self.trajectory_dir / f"{scene_id}_{variant}.json")
            for variant in variants
        }

    def _save_errors(self, scene_id: str, output_file: Path) -> None:
        # NOTE: The median and the 25-75% range over the rays of the error of the renderer used in training.
        fig, ax = plt.subplots(figsize=self._get_figure_size())
        inset_ax = ax.inset_axes(self.inset_bounds)
        trajectories = self._load_trajectories(scene_id, self.variants)

        # NOTE: The inset ends one checkpoint after the first from which the medians of all the renderers stay
        # settled.
        medians = pd.concat(
            [t[f"metrics.{self.metric}.q50"] for t in trajectories.values()], axis=1
        )
        unsettled_steps = medians.index[(medians >= self.settled_error).any(axis=1)]
        last_unsettled_step = (
            unsettled_steps.max() if len(unsettled_steps) else medians.index.min()
        )
        min_step = int(medians.index.min())
        inset_step = int(medians.index[medians.index > last_unsettled_step].min()) + min_step
        max_inset_error = max(
            t.loc[:inset_step, f"metrics.{self.metric}.q75"].max() for t in trajectories.values()
        )

        handles = []
        for variant, trajectory in trajectories.items():
            for target_ax in (ax, inset_ax):
                band = target_ax.fill_between(
                    trajectory.index,
                    trajectory[f"metrics.{self.metric}.q25"],
                    trajectory[f"metrics.{self.metric}.q75"],
                    color=VARIANT_COLORS[variant],
                    alpha=0.25,
                    linewidth=0.0,
                )
                (line,) = target_ax.plot(
                    trajectory.index,
                    trajectory[f"metrics.{self.metric}.q50"],
                    color=VARIANT_COLORS[variant],
                    linewidth=1.5,
                )
            handles.append((band, line))

        max_step = int(round(medians.index.max(), -2))
        ax.set_xlim(0, max_step)
        inset_ax.set_xlim(min_step, inset_step)
        inset_ax.xaxis.set_major_locator(ticker.MultipleLocator(min_step))
        # NOTE: The vertical ranges end at the ticks.
        ax.set_ylim(*snap_range(ax.yaxis, (0.0, self.max_error)))
        inset_ax.set_ylim(*snap_range(inset_ax.yaxis, (0.0, max_inset_error)))
        ax.set_xlabel("Training Step")
        ax.set_ylabel(self.metric.label)
        ax.legend(
            handles=handles,
            labels=[get_variant_name(variant) for variant in self.variants],
            loc="upper right",
        )

        save_figure(fig, output_file)

    def __call__(self) -> None:
        configure_style(self.font_size)

        variant_ids = [get_variant_id(variant) for variant in self.variants]
        for scene_id, output_dir in zip(self.scene_ids, self.output_dirs, strict=True):
            self._save_errors(
                scene_id=scene_id,
                output_file=(
                    output_dir
                    / f"learned_first_passage_pmf_{self.metric.file_id}_distance_plot_{'_vs_'.join(variant_ids)}_vs_MC.pdf"
                ),
            )


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        TrainingTrajectoryPlotter,
        config=(tyro.conf.AvoidSubcommands,),
    )()
