import dataclasses
import itertools
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tyro
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axisartist.axislines import AxesZero


@dataclasses.dataclass
class CrossingFigureCreator:
    output_dir: Path

    # TMLR layout dimensions (from tmlr.sty)
    text_width: float = 6.5

    crossing_width_ratio: float = 0.5
    crossing_aspect_ratio: float = 1.0 / 3.0
    crossing_font_size: float = 8.0

    reflection_width_ratio: float = 0.5
    reflection_aspect_ratio: float = 1.0 / 3.0
    reflection_font_size: float = 8.0

    down_cross_color: str = "tomato"
    down_cross_width: float = 1.0

    up_cross_color: str = "dodgerblue"
    up_cross_width: float = 1.0

    survival_color: str = "mediumseagreen"
    survival_width: float = 1.0

    marker_size: float = 10.0

    ou_drift: float = 50.0
    ou_diffusion: float = 0.5
    initial_variance: float = 0.1

    start_time: float = 0.25
    end_time: float = 0.75

    resolution: int = 1000

    seed: int = 0

    def _get_figure_size(self, width_ratio: float, aspect_ratio: float) -> tuple[float, float]:
        width = self.text_width * width_ratio
        height = width * aspect_ratio
        return (width, height)

    def _get_markov_chain(
        self,
        times: npt.NDArray,
        marginal_means: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        intervals = np.diff(times)
        markov_scales = np.exp(-self.ou_drift * intervals)
        markov_vars = self.ou_diffusion / (2.0 * self.ou_drift)
        markov_vars = markov_vars * -np.expm1(-2.0 * self.ou_drift * intervals)
        markov_shifts = marginal_means[..., 1:] - markov_scales * marginal_means[..., :-1]
        return markov_scales, markov_shifts, markov_vars

    def _get_ssdp_samples(
        self,
        times: npt.NDArray,
        marginal_means: npt.NDArray,
        num_samples: int = 1,
    ) -> npt.NDArray:
        markov_scales, markov_shifts, markov_vars = self._get_markov_chain(times, marginal_means)
        ssdp_samples = itertools.accumulate(
            iterable=zip(markov_scales, markov_shifts, markov_vars, strict=True),
            func=lambda prev_samples, markov_params: np.random.normal(
                loc=markov_params[0] * prev_samples + markov_params[1],
                scale=np.sqrt(markov_params[2]),
                size=(num_samples,),
            ),
            initial=np.random.normal(
                loc=marginal_means[0],
                scale=np.sqrt(self.initial_variance),
                size=(num_samples,),
            ),
        )
        ssdp_samples = np.stack(list(ssdp_samples), axis=-1)
        return ssdp_samples

    def _visualize_crossing(
        self,
        axes: mpl.axes.Axes,
        times: npt.NDArray,
        down_ssdp_samples: npt.NDArray,
        up_ssdp_samples: npt.NDArray,
    ) -> mpl.axes.Axes:
        down_cross_index = np.argmax(down_ssdp_samples <= 0.0)
        axes.plot(
            times[times < self.start_time],
            down_ssdp_samples[times < self.start_time],
            color=self.survival_color,
            linewidth=self.survival_width,
            label=r"$\mathcal{A}_{i}$",
        )
        axes.plot(
            times[times >= self.start_time],
            down_ssdp_samples[times >= self.start_time],
            color=self.down_cross_color,
            linewidth=self.down_cross_width,
            label=r"$\mathcal{B}_{i}^{\downarrow}$",
        )
        axes.plot(
            times[down_cross_index],
            down_ssdp_samples[down_cross_index],
            marker="o",
            markersize=self.marker_size,
            markeredgewidth=self.down_cross_width,
            markerfacecolor="none",
            markeredgecolor="black",
        )
        axes.plot(
            self.end_time,
            np.interp(self.end_time, times, down_ssdp_samples),
            marker="o",
            markersize=self.marker_size,
            markeredgewidth=self.down_cross_width,
            markerfacecolor="none",
            markeredgecolor="black",
        )

        up_cross_index = np.argmax(up_ssdp_samples <= 0.0)
        axes.plot(
            times[times >= self.start_time],
            up_ssdp_samples[times >= self.start_time],
            color=self.up_cross_color,
            linewidth=self.up_cross_width,
            label=r"$\mathcal{B}_{i}^{\uparrow}$",
        )
        axes.plot(
            times[up_cross_index],
            up_ssdp_samples[up_cross_index],
            marker="o",
            markersize=self.marker_size,
            markeredgewidth=self.up_cross_width,
            markerfacecolor="none",
            markeredgecolor="black",
        )
        axes.plot(
            self.end_time,
            np.interp(self.end_time, times, up_ssdp_samples),
            marker="o",
            markersize=self.marker_size,
            markeredgewidth=self.up_cross_width,
            markerfacecolor="none",
            markeredgecolor="black",
        )

        axes.axvline(
            x=self.start_time,
            color="black",
            alpha=0.5,
            linewidth=1.0,
            linestyle="--",
        )
        axes.axvline(
            x=self.end_time,
            color="black",
            alpha=0.5,
            linewidth=1.0,
            linestyle="--",
        )

        axes.text(
            x=self.start_time - 0.01,
            y=0.0 - 0.05,
            s="$t_{i}$",
            ha="right",
            va="top",
            color="black",
        )
        axes.text(
            x=self.end_time - 0.01,
            y=0.0 - 0.05,
            s="$t_{i+1}$",
            ha="right",
            va="top",
            color="black",
        )

        axes.text(
            x=self.start_time - 0.1 - 0.01,
            y=1.0,
            s="$S(t)$",
            ha="right",
            va="center",
            color="black",
        )
        axes.text(
            x=self.end_time + 0.1,
            y=0.0 - 0.05,
            s="$t$",
            ha="center",
            va="top",
            color="black",
        )

        axes.set_xlim(self.start_time - 0.1, self.end_time + 0.1)
        axes.set_ylim(-0.75, 1.0)

        axes.set_yticks([0.0])

        axes.legend(
            loc="upper center",
            frameon=False,
            ncols=3,
            columnspacing=1.0,
            bbox_to_anchor=(0.5, 1.1),
        )

        for direction in ["xzero", "left"]:
            axes.axis[direction].set_visible(True)
            axes.axis[direction].set_axisline_style("-|>")
            axes.axis[direction].major_ticks.set_visible(False)
            axes.axis[direction].minor_ticks.set_visible(False)
            if direction == "left":
                axes.axis[direction].major_ticklabels.set_visible(True)
                axes.axis[direction].minor_ticklabels.set_visible(False)
            else:
                axes.axis[direction].major_ticklabels.set_visible(False)
                axes.axis[direction].minor_ticklabels.set_visible(False)

        for direction in ["yzero", "right", "bottom", "top"]:
            axes.axis[direction].set_visible(False)

        return axes

    def _visualize_reflection(
        self,
        axes: mpl.axes.Axes,
        times: npt.NDArray,
        up_ssdp_samples: npt.NDArray,
    ) -> mpl.axes.Axes:
        up_cross_index = np.argmax(up_ssdp_samples <= 0.0)
        axes.plot(
            times[:up_cross_index],
            up_ssdp_samples[:up_cross_index],
            color=self.up_cross_color,
            linewidth=self.up_cross_width,
            label=r"Brownian bridge $B(\omega)$",
        )
        axes.plot(
            times[up_cross_index:],
            up_ssdp_samples[up_cross_index:],
            color=self.up_cross_color,
            linewidth=self.up_cross_width,
            alpha=0.25,
        )
        axes.plot(
            times[up_cross_index:],
            -up_ssdp_samples[up_cross_index:],
            color=self.down_cross_color,
            linewidth=self.down_cross_width,
            label=r"Reflected path $-B(\omega)$",
        )
        axes.plot(
            times[up_cross_index],
            up_ssdp_samples[up_cross_index],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=self.marker_size,
            markeredgewidth=self.up_cross_width,
        )
        axes.plot(
            self.end_time,
            np.interp(self.end_time, times, up_ssdp_samples),
            marker="o",
            markerfacecolor="none",
            markeredgecolor=(0.0, 0.0, 0.0, 0.25),
            markersize=self.marker_size,
            markeredgewidth=self.up_cross_width,
        )
        axes.plot(
            self.end_time,
            np.interp(self.end_time, times, -up_ssdp_samples),
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=self.marker_size,
            markeredgewidth=self.down_cross_width,
        )

        axes.axvline(
            x=self.start_time,
            color="black",
            alpha=0.5,
            linewidth=1.0,
            linestyle="--",
        )
        axes.axvline(
            x=self.end_time,
            color="black",
            alpha=0.5,
            linewidth=1.0,
            linestyle="--",
        )

        arrow = FancyArrowPatch(
            posA=(0.5, -0.5),
            posB=(0.5, 0.5),
            arrowstyle="->",
            mutation_scale=15.0,
            color="black",
            linewidth=1.0,
            connectionstyle="arc3,rad=0.5",
        )
        axes.add_patch(arrow)

        zero_level = np.interp(self.start_time, times, up_ssdp_samples)
        arrow = FancyArrowPatch(
            posA=(self.start_time, zero_level),
            posB=(self.end_time + 0.1, zero_level),
            arrowstyle="-|>",
            mutation_scale=10.0,
            color="black",
            linewidth=1.0,
        )
        axes.add_patch(arrow)

        axes.text(
            x=self.end_time - 0.02,
            y=np.interp(self.end_time, times, up_ssdp_samples) + 0.1,
            s="$y$",
            ha="right",
            va="center",
            color="black",
        )
        axes.text(
            x=self.end_time - 0.02,
            y=np.interp(self.end_time, times, -up_ssdp_samples) - 0.1,
            s=r"$2 \ell - y$",
            ha="right",
            va="center",
            color="black",
        )

        axes.text(
            x=self.start_time - 0.01,
            y=1.25,
            s=r"$B(\omega)$",
            ha="right",
            va="center",
            color="black",
        )
        axes.text(
            x=self.end_time + 0.1 - 0.04,
            y=zero_level,
            s=r"$\omega$",
            ha="center",
            va="bottom",
            color="black",
        )

        axes.set_xlim(self.start_time, self.end_time + 0.1)
        axes.set_ylim(-0.75, 1.25)

        axes.set_yticks([0.0, zero_level])
        axes.set_yticklabels([r"$\ell$", "0"])

        axes.legend(
            loc="upper center",
            frameon=False,
            bbox_to_anchor=(0.4, 1.2),
            fontsize=self.reflection_font_size - 2.0,
        )

        for direction in ["xzero", "left"]:
            axes.axis[direction].set_visible(True)
            if direction == "xzero":
                axes.axis[direction].line.set_linestyle("--")
            else:
                axes.axis[direction].set_axisline_style("-|>")
            axes.axis[direction].major_ticks.set_visible(False)
            axes.axis[direction].minor_ticks.set_visible(False)
            if direction == "left":
                axes.axis[direction].major_ticklabels.set_visible(True)
                axes.axis[direction].minor_ticklabels.set_visible(False)
            else:
                axes.axis[direction].major_ticklabels.set_visible(False)
                axes.axis[direction].minor_ticklabels.set_visible(False)

        for direction in ["yzero", "right", "bottom", "top"]:
            axes.axis[direction].set_visible(False)

        return axes

    def __call__(self) -> None:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["DejaVu Serif"],
                "mathtext.fontset": "cm",
            }
        )

        np.random.seed(self.seed)

        times = np.linspace(0.0, 1.0, self.resolution)

        down_marginal_means = -2.0 * times + 1.0
        up_marginal_means = np.where(
            times < 0.25,
            -2.0 * times + 1.0,
            np.cos(times * 2.0 * np.pi) * 1.0 + 0.5,
        )

        [down_ssdp_samples] = self._get_ssdp_samples(times, down_marginal_means)
        [up_ssdp_samples] = self._get_ssdp_samples(times, up_marginal_means)

        plt.rcParams.update({"font.size": self.crossing_font_size})

        figure_size = self._get_figure_size(
            width_ratio=self.crossing_width_ratio,
            aspect_ratio=self.crossing_aspect_ratio,
        )
        figure = plt.figure(figsize=figure_size)

        axes = figure.add_subplot(111, axes_class=AxesZero)
        self._visualize_crossing(
            axes=axes,
            times=times,
            down_ssdp_samples=down_ssdp_samples,
            up_ssdp_samples=up_ssdp_samples,
        )

        plt.savefig(
            self.output_dir / "crossing.pdf",
            bbox_inches="tight",
            pad_inches=0.0,
        )

        up_marginal_means *= 2.0 * (1.0 - times) + 0.5 * times
        [up_ssdp_samples] = self._get_ssdp_samples(times, up_marginal_means)

        plt.rcParams.update({"font.size": self.reflection_font_size})

        figure_size = self._get_figure_size(
            width_ratio=self.reflection_width_ratio,
            aspect_ratio=self.reflection_aspect_ratio,
        )
        figure = plt.figure(figsize=figure_size)

        axes = figure.add_subplot(111, axes_class=AxesZero)
        self._visualize_reflection(
            axes=axes,
            times=times,
            up_ssdp_samples=up_ssdp_samples,
        )

        plt.savefig(
            self.output_dir / "reflection.pdf",
            bbox_inches="tight",
            pad_inches=0.0,
        )


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        CrossingFigureCreator,
        config=(tyro.conf.AvoidSubcommands,),
    )()
