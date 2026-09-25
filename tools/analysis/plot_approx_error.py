import dataclasses
import enum
import json
from pathlib import Path

import loguru
import matplotlib
import numpy as np
import pandas as pd
import tyro
from matplotlib.axis import Axis
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator

from tools.analysis.evaluate_approx_error import Variant

# NOTE: The texts are typeset by LaTeX with the fonts of the paper (tmlr.sty) and its marks.
matplotlib.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402

LATEX_PREAMBLE = "".join(
    [
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{amsmath}",
        r"\usepackage{pifont}",
        r"\newcommand{\cmark}{\ding{51}}",
        r"\newcommand{\xmark}{\ding{55}}",
    ]
)

TRAJECTORY_COLOR = "black"


class Metric(enum.StrEnum):
    # NOTE: The distance between the first-passage distributions conditioned on a first passage inside the ray,
    # and the squared distance between the probabilities of that first passage (`_compute_metrics`).
    CONDITIONAL_CRAMER_DISTANCE = enum.auto()
    SQUARED_DISTANCE = enum.auto()

    @property
    def label(self) -> str:
        return dict(
            [
                (Metric.CONDITIONAL_CRAMER_DISTANCE, r"Cram\'er Distance"),
                (Metric.SQUARED_DISTANCE, "Squared Distance"),
            ]
        )[self]

    @property
    def file_id(self) -> str:
        return dict(
            [
                (Metric.CONDITIONAL_CRAMER_DISTANCE, "cramer"),
                (Metric.SQUARED_DISTANCE, "squared"),
            ]
        )[self]


VARIANT_COLORS = {
    Variant.NA: "tomato",
    Variant.BF_UP: "dodgerblue",
    Variant.BF: "mediumseagreen",
    Variant.NA_UP: "hotpink",
}


def get_variant_name(variant: Variant) -> str:
    # NOTE: Whether the renderer models the survival event and the up-crossing event.
    survival_mark = r"\xmark" if variant.is_survival_approx else r"\cmark"
    up_cross_mark = r"\xmark" if variant.is_up_cross_approx else r"\cmark"
    return rf"SSDP-Facto ($\mathcal{{A}}$ {survival_mark}, $\mathcal{{B}}^{{\uparrow}}$ {up_cross_mark})"


def get_variant_id(variant: Variant) -> str:
    return variant.upper().replace("_", "-")


# NOTE: The maps are drawn on tau (the square root of the diffusion) and kappa of the OU process, uniformly in log.
AXIS_NAMES = ("ou_diffusion_root", "ou_drift")
AXIS_LABELS = (r"OU Diffusion $\tau$", r"OU Drift $\kappa$")


def configure_style(font_size: float) -> None:
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": LATEX_PREAMBLE,
            "font.family": "serif",
            "font.size": font_size,
            "figure.constrained_layout.use": True,
            "legend.frameon": False,
        }
    )


def snap_range(
    axis: Axis, value_range: tuple[float, float], tolerance: float = 1.0e-9
) -> tuple[float, float]:
    # NOTE: The range extended outward to multiples of the tick step that the automatic locator of the axis chooses
    # for it, so that the ticks end at the ends of the range (iterated in case the locator changes its step for
    # the extended range; the tolerance keeps a range that is already a multiple, up to rounding, in place).
    for _ in range(10):
        axis.set_view_interval(*value_range, ignore=True)
        (step, *_) = np.diff(axis.get_majorticklocs())
        snapped_range = (
            float(np.floor(value_range[0] / step + tolerance) * step),
            float(np.ceil(value_range[1] / step - tolerance) * step),
        )
        if np.allclose(snapped_range, value_range, rtol=tolerance, atol=0.0):
            break
        value_range = snapped_range
    return value_range


def fit_square_axes(fig: plt.Figure, ax: plt.Axes, tolerance: float = 2.0e-3) -> None:
    # NOTE: The height of the figure set so that the axes is square: its width is what the layout gives it, and
    # the decorations around it do not depend on the size of the figure. Used instead of a box aspect so that
    # a color bar attached to the axes, which the layout gives the height of the axes, matches it.
    for _ in range(10):
        fig.canvas.draw()
        position = ax.get_position()
        width, height = fig.get_size_inches()
        axes_width, axes_height = position.width * width, position.height * height
        if abs(axes_width - axes_height) < tolerance:
            break
        fig.set_size_inches(width, height + axes_width - axes_height)


def save_figure(fig: plt.Figure, output_file: Path, dpi: int = 600) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)

    loguru.logger.success(f"Saved the figure to <{output_file}>.")


def load_sweep(sweep_file: Path) -> pd.DataFrame:
    with open(sweep_file) as fp:
        sweep = pd.json_normalize(json.load(fp))
    sweep["config.ou_diffusion_root"] = np.sqrt(sweep["config.ou_diffusion"])
    return sweep


def get_distances(sweep: pd.DataFrame, variant: Variant, metric: str) -> pd.DataFrame:
    # NOTE: Distances of a renderer to Monte Carlo on the grid of the horizontal (rows) and vertical (columns) axes.
    horizontal_name, vertical_name = AXIS_NAMES
    return sweep[sweep["variant"] == variant].pivot(
        index=f"config.{horizontal_name}",
        columns=f"config.{vertical_name}",
        values=f"metrics.{metric}",
    )


def get_max_distance(sweep_files: tuple[Path, ...], metric: str, quantile: float) -> float:
    # NOTE: The top of the color scale common to all the maps of a metric: the largest quantile (of the given
    # level) of a map of the distance of a renderer to Monte Carlo, over the given sweeps and all the renderers.
    return max(
        get_distances(load_sweep(sweep_file), variant, metric).stack().quantile(quantile)
        for sweep_file in sweep_files
        for variant in Variant
        if variant is not Variant.MC
    )


def load_trajectory(trajectory_file: Path) -> pd.DataFrame:
    with open(trajectory_file) as fp:
        trajectory = pd.json_normalize(json.load(fp))
    trajectory["metrics.ou_diffusion_root"] = np.sqrt(trajectory["metrics.ou_diffusion"])
    return trajectory.set_index("config.step").sort_index()


def interpolate_distances(
    distances: pd.DataFrame,
    num_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # NOTE: The grid is interpolated bilinearly (in the log coordinates in which it is uniform) on this many points
    # per axis for drawing the map: the gouraud shading of `pcolormesh` is linear on the triangles of each cell and
    # shows their diagonals on the raw grid; at this number the cells are far below the pixels of the rasterized
    # map and the picture no longer depends on it.
    lookup = RegularGridInterpolator(
        points=(np.log10(distances.index.values), np.log10(distances.columns.values)),
        values=distances.values,
    )
    horizontal_values = np.logspace(
        *np.log10([distances.index.min(), distances.index.max()]), num_points
    )
    vertical_values = np.logspace(
        *np.log10([distances.columns.min(), distances.columns.max()]), num_points
    )
    horizontal_values, vertical_values = np.meshgrid(horizontal_values, vertical_values)
    values = lookup(np.stack([np.log10(horizontal_values), np.log10(vertical_values)], axis=-1))
    return horizontal_values, vertical_values, values


def draw_trajectory(
    ax: plt.Axes,
    trajectory: pd.DataFrame,
    distances: pd.DataFrame,
    annotated_step: int,
    font_size: float,
) -> None:
    # NOTE: The learned parameters of all the checkpoints, clipped onto the map. The arrows and the labels of the steps
    # stop at the annotated step; the rest is a plain line.
    horizontal_name, vertical_name = AXIS_NAMES
    horizontal_values = trajectory[f"metrics.{horizontal_name}"].clip(
        distances.index.min(), distances.index.max()
    )
    vertical_values = trajectory[f"metrics.{vertical_name}"].clip(
        distances.columns.min(), distances.columns.max()
    )
    color = TRAJECTORY_COLOR

    ax.plot(horizontal_values, vertical_values, color=color, linewidth=1.0, zorder=5)
    annotated_steps = trajectory.index[trajectory.index <= annotated_step]
    for step, next_step in zip(annotated_steps[:-1], annotated_steps[1:], strict=True):
        ax.annotate(
            text="",
            xy=(horizontal_values.loc[next_step], vertical_values.loc[next_step]),
            xytext=(horizontal_values.loc[step], vertical_values.loc[step]),
            arrowprops=dict(
                arrowstyle="-|>", color=color, linewidth=1.0, shrinkA=0.0, shrinkB=0.0
            ),
            zorder=6,
        )
    for step in annotated_steps:
        ax.annotate(
            text=f"{step}",
            xy=(horizontal_values.loc[step], vertical_values.loc[step]),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=font_size - 2.0,
            bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.5),
            zorder=8,
        )
    # NOTE: The first and the last checkpoints.
    ax.scatter(
        horizontal_values.iloc[[0, -1]],
        vertical_values.iloc[[0, -1]],
        s=15,
        facecolor="white",
        edgecolor=color,
        linewidth=1.0,
        zorder=7,
        clip_on=False,
    )


@dataclasses.dataclass
class ApproxErrorPlotter:
    # NOTE: `<sweep_dir>/<scene_id>.json` from `evaluate_approx_error.py`.
    sweep_dir: Path
    # NOTE: For each scene, the maps of the distances of the two renderers to Monte Carlo and that of their difference
    # are saved in the corresponding directory.
    scene_ids: tuple[str, ...]
    output_dirs: tuple[Path, ...]
    # NOTE: The color scale saturates at this value (`get_max_distance` over all the sweeps of the metric, computed by
    # `plot_approx_error.sh`); the difference at plus and minus this value.
    max_distance: float
    # NOTE: The renderer without the approximation of interest, and the one with it.
    variants: tuple[Variant, Variant] = (Variant.BF_UP, Variant.NA)
    metric: Metric = Metric.CONDITIONAL_CRAMER_DISTANCE
    # NOTE: The runs whose learned parameters are drawn on the maps, from
    # `<trajectory_dir>/<scene_id>_<variant>.json` of `evaluate_training_trajectory.py`.
    trajectory_dir: Path | None = None
    annotated_step: int = 500

    # TMLR layout dimensions (from tmlr.sty)
    text_width: float = 6.5

    # NOTE: Half of the text width, with the height to width ratio 1:1.25.
    width_ratio: float = 0.5
    aspect_ratio: float = 0.8
    font_size: float = 8.0

    def _get_figure_size(self) -> tuple[float, float]:
        width = self.text_width * self.width_ratio
        height = width * self.aspect_ratio
        return (width, height)

    def _get_distance_cmap(self, variant: Variant) -> Colormap:
        # NOTE: The distance of a renderer in its color.
        return LinearSegmentedColormap.from_list(
            name="distance", colors=["white", VARIANT_COLORS[variant]]
        )

    def _get_difference_cmap(self) -> Colormap:
        # NOTE: The difference of the distances diverges toward the color of the worse renderer.
        variant_1, variant_2 = self.variants
        return LinearSegmentedColormap.from_list(
            name="difference",
            colors=[VARIANT_COLORS[variant_1], "white", VARIANT_COLORS[variant_2]],
        )

    def _get_color_scale(self) -> float:
        # NOTE: The end of the color scale: the difference maps, whose scale is symmetric about zero, get the range
        # extended to the ticks that their color bar chooses for it (`snap_range`), and the distance maps share
        # the end (so that the scales of the two kinds of maps coincide up to the sign). The color bar is laid out as
        # in `_save_map`, on a map with the extent of the grid.
        distances = pd.DataFrame(np.zeros((2, 2)), index=[1.0e-3, 1.0e1], columns=[1.0e-2, 1.0e2])
        fig, ax = plt.subplots(figsize=self._get_figure_size())
        colorbar = self._draw_distances(
            ax=ax,
            distances=distances,
            value_range=(-self.max_distance, self.max_distance),
            cmap=self._get_difference_cmap(),
            extend="both",
            label=f"Difference in {self.metric.label}",
        )
        fit_square_axes(fig, ax)
        _, max_distance = snap_range(colorbar.ax.yaxis, (-self.max_distance, self.max_distance))
        plt.close(fig)
        return max_distance

    def _draw_distances(
        self,
        ax: plt.Axes,
        distances: pd.DataFrame,
        value_range: tuple[float, float],
        cmap: Colormap,
        extend: str,
        label: str,
    ) -> Colorbar:
        horizontal_values, vertical_values, values = interpolate_distances(distances)
        mesh = ax.pcolormesh(
            horizontal_values,
            vertical_values,
            values,
            cmap=cmap,
            vmin=value_range[0],
            vmax=value_range[1],
            shading="gouraud",
            rasterized=True,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(distances.index.min(), distances.index.max())
        ax.set_ylim(distances.columns.min(), distances.columns.max())
        ax.set_xlabel(AXIS_LABELS[0])
        ax.set_ylabel(AXIS_LABELS[1])

        return ax.figure.colorbar(mesh, ax=ax, extend=extend, label=label)

    def _save_map(
        self,
        values: pd.DataFrame,
        value_range: tuple[float, float],
        cmap: Colormap,
        extend: str,
        label: str,
        trajectory_file: Path | None,
        output_file: Path,
    ) -> None:
        # NOTE: A map of the values with the parameters learned by a run drawn on it.
        fig, ax = plt.subplots(figsize=self._get_figure_size())
        self._draw_distances(
            ax=ax,
            distances=values,
            value_range=value_range,
            cmap=cmap,
            extend=extend,
            label=label,
        )
        if trajectory_file is not None:
            draw_trajectory(
                ax=ax,
                trajectory=load_trajectory(trajectory_file),
                distances=values,
                annotated_step=self.annotated_step,
                font_size=self.font_size,
            )
        fit_square_axes(fig, ax)
        save_figure(fig, output_file)

    def _get_trajectory_file(self, scene_id: str, variant: Variant) -> Path | None:
        return (
            None
            if self.trajectory_dir is None
            else self.trajectory_dir / f"{scene_id}_{variant}.json"
        )

    def __call__(self) -> None:
        configure_style(self.font_size)

        variant_ids = [get_variant_id(variant) for variant in self.variants]
        max_distance = self._get_color_scale()

        for scene_id, output_dir in zip(self.scene_ids, self.output_dirs, strict=True):
            sweep = load_sweep(self.sweep_dir / f"{scene_id}.json")
            distances = {
                variant: get_distances(sweep, variant, self.metric) for variant in self.variants
            }

            # NOTE: The distance of each renderer to Monte Carlo, with the parameters learned by its own run.
            for variant, variant_id in zip(self.variants, variant_ids, strict=True):
                self._save_map(
                    values=distances[variant],
                    value_range=(0.0, max_distance),
                    cmap=self._get_distance_cmap(variant),
                    extend="max",
                    label=self.metric.label,
                    trajectory_file=self._get_trajectory_file(scene_id, variant),
                    output_file=(
                        output_dir
                        / f"analytic_first_passage_pmf_{self.metric.file_id}_distance_map_{variant_id}_vs_MC.pdf"
                    ),
                )

            # NOTE: The difference (with the approximation minus without), with the parameters learned by the run
            # with the approximation.
            variant_1, variant_2 = self.variants
            self._save_map(
                values=distances[variant_2] - distances[variant_1],
                value_range=(-max_distance, max_distance),
                cmap=self._get_difference_cmap(),
                extend="both",
                label=f"Difference in {self.metric.label}",
                trajectory_file=self._get_trajectory_file(scene_id, variant_2),
                output_file=(
                    output_dir
                    / f"analytic_first_passage_pmf_{self.metric.file_id}_difference_map_{'_vs_'.join(variant_ids)}.pdf"
                ),
            )


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        ApproxErrorPlotter,
        config=(tyro.conf.AvoidSubcommands,),
    )()
