import dataclasses
from pathlib import Path

import numpy as np
import tyro
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from tools.analysis.evaluate_approx_error import Variant
from tools.analysis.plot_approx_error import (
    VARIANT_COLORS,
    configure_style,
    fit_square_axes,
    get_variant_id,
    get_variant_name,
    plt,
    save_figure,
    snap_range,
)


def get_sdf_range(slice_files: tuple[Path, ...], quantiles: tuple[float, float]) -> float:
    # NOTE: The end of the color scale common to all the slices: the larger magnitude of the given lower and upper
    # quantiles of the learned SDF over the given slices (the scale is symmetric about zero).
    sdf_values = np.concatenate([np.load(slice_file)["sdf"].ravel() for slice_file in slice_files])
    lower_value, upper_value = np.quantile(sdf_values, quantiles)
    return max(-lower_value, upper_value)


@dataclasses.dataclass
class SDFErrorPlotter:
    # NOTE: The slice of the learned SDF of `evaluate_sdf_error.py` (`<input_dir>/<scene_id>_<variant>.npz`)
    # around the slab: the SDF in the color of the renderer inside the learned surface and in gray outside (white at
    # zero), with the zero level sets of the learned SDF (solid) and of the exact one (dashed).
    input_dir: Path
    output_dir: Path
    scene_id: str
    # NOTE: The color scale saturates at plus and minus this value (`get_sdf_range` over all the slices, computed by
    # `plot_sdf_error.sh`), extended to the ticks of the color bar.
    sdf_range: float
    variant: Variant = Variant.NA
    # NOTE: The ticks of the axes, given by hand for this figure only: the window of the slice is a choice, not
    # a multiple of the tick steps that matplotlib chooses for it.
    tick_steps: tuple[float, float] = (0.025, 0.25)

    # TMLR layout dimensions (from tmlr.sty)
    text_width: float = 6.5

    width_ratio: float = 0.5
    aspect_ratio: float = 0.8
    font_size: float = 8.0

    def _get_figure_size(self) -> tuple[float, float]:
        width = self.text_width * self.width_ratio
        return width, width * self.aspect_ratio

    def __call__(self) -> None:
        configure_style(self.font_size)

        slices = np.load(self.input_dir / f"{self.scene_id}_{self.variant}.npz")
        x, y, sdf, exact_sdf = (slices[name] for name in ("x", "y", "sdf", "exact_sdf"))

        fig, ax = plt.subplots(figsize=self._get_figure_size())
        mesh = ax.pcolormesh(
            x,
            y,
            sdf.T,
            cmap=LinearSegmentedColormap.from_list(
                name="sdf", colors=[VARIANT_COLORS[self.variant], "white", "gray"]
            ),
            vmin=-self.sdf_range,
            vmax=self.sdf_range,
            shading="nearest",
            rasterized=True,
        )
        for values, linestyle in ((sdf, "-"), (exact_sdf, "--")):
            ax.contour(
                x, y, values.T, levels=[0.0], colors="black", linestyles=linestyle, linewidths=0.8
            )
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        for axis, tick_step in zip((ax.xaxis, ax.yaxis), self.tick_steps, strict=True):
            axis.set_major_locator(ticker.MultipleLocator(tick_step))
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.legend(
            handles=[
                Line2D([], [], color="black", linestyle="-"),
                Line2D([], [], color="black", linestyle="--"),
            ],
            labels=[get_variant_name(self.variant), "Ground Truth"],
            # NOTE: Above the axes (which the slab fills from end to end), in one column so that the layout does
            # not narrow the axes to the width of the legend.
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            frameon=False,
        )
        colorbar = fig.colorbar(mesh, ax=ax, extend="both", label="Learned Signed Distance")
        fit_square_axes(fig, ax)
        # NOTE: The color scale ends at the ticks of the color bar (whose length is set above).
        mesh.set_clim(*snap_range(colorbar.ax.yaxis, (-self.sdf_range, self.sdf_range)))
        save_figure(fig, self.output_dir / f"learned_sdf_slice_{get_variant_id(self.variant)}.pdf")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        SDFErrorPlotter,
        config=(tyro.conf.AvoidSubcommands,),
    )()
