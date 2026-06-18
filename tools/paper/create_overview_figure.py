import dataclasses
import itertools
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy
import tyro
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import FancyArrowPatch, Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _get_figure_size(
    base_width: float,
    width_ratio: float,
    aspect_ratio: float,
) -> tuple[float, float]:
    width = base_width * width_ratio
    height = width * aspect_ratio
    return (width, height)


@dataclasses.dataclass
class OverviewFigureCreator:
    output_dir: Path

    # TMLR layout dimensions (from tmlr.sty)
    text_width: float = 6.5

    ssdp_width_ratio: float = 0.4
    ssdp_aspect_ratio: float = 1.0 / 4.0
    ssdp_font_size: float = 8.0

    ssdp_mean_color: str = "dodgerblue"
    ssdp_mean_width: float = 1.0

    ssdp_sample_color: str = "dodgerblue"
    ssdp_sample_alpha: float = 0.2
    ssdp_sample_width: float = 0.2

    prior_width_ratio: float = 0.4
    prior_aspect_ratio: float = 1.0 / 4.0
    prior_font_size: float = 8.0

    prior_color: str = "tomato"
    prior_alpha: float = 0.25
    prior_width: float = 1.0

    joint_width_ratio: float = 0.4
    joint_aspect_ratio: float = 1.0
    joint_font_size: float = 8.0

    joint_color: str = "dodgerblue"
    joint_alpha: float = 0.75

    marginal_color: str = "mediumseagreen"
    marginal_alpha: float = 0.25
    marginal_width: float = 2.0

    observation_color: str = "red"
    observation_alpha: float = 1.0
    observation_width: float = 1.0

    ou_drifts: float = 1.0
    ou_diffusions: float = 0.1
    initial_variance: float = 0.1

    observation: float = 0.6

    resolution: int = 100
    num_samples: int = 10000

    seed: int = 0

    def _get_markov_chain(
        self,
        times: npt.NDArray,
        marginal_means: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        intervals = np.diff(times)
        markov_scales = np.exp(-self.ou_drifts * intervals)
        markov_vars = self.ou_diffusions / (2.0 * self.ou_drifts)
        markov_vars = markov_vars * -np.expm1(-2.0 * self.ou_drifts * intervals)
        markov_shifts = marginal_means[..., 1:] - markov_scales * marginal_means[..., :-1]
        return markov_scales, markov_shifts, markov_vars

    def _get_ssdp_samples(
        self,
        times: npt.NDArray,
        marginal_means: npt.NDArray,
    ) -> npt.NDArray:
        markov_scales, markov_shifts, markov_vars = self._get_markov_chain(times, marginal_means)
        ssdp_samples = itertools.accumulate(
            iterable=zip(markov_scales, markov_shifts, markov_vars, strict=True),
            func=lambda prev_samples, markov_params: np.random.normal(
                loc=markov_params[0] * prev_samples + markov_params[1],
                scale=np.sqrt(markov_params[2]),
                size=(self.num_samples,),
            ),
            initial=np.random.normal(
                loc=marginal_means[0],
                scale=np.sqrt(self.initial_variance),
                size=(self.num_samples,),
            ),
        )
        ssdp_samples = np.stack(list(ssdp_samples), axis=-1)
        return ssdp_samples

    def _get_first_passage_distributon(self, ssdp_samples: npt.NDArray) -> npt.NDArray:
        foreground_mask = np.any(ssdp_samples <= 0.0, axis=-1)
        first_passage_indices = np.argmax(ssdp_samples <= 0.0, axis=-1)
        first_passage_indices = first_passage_indices[foreground_mask]
        first_passage_distributon = np.eye(ssdp_samples.shape[-1])[first_passage_indices]
        first_passage_distributon = np.sum(first_passage_distributon, axis=0)
        first_passage_distributon /= np.sum(first_passage_distributon)
        return first_passage_distributon

    def _visualize_ssdp_samples(
        self,
        axes: mpl.axes.Axes,
        times: npt.NDArray,
        marginal_means: npt.NDArray,
        ssdp_samples: npt.NDArray,
        max_num_samples: int = 100,
    ) -> mpl.axes.Axes:
        axes.plot(
            times,
            marginal_means,
            color=self.ssdp_mean_color,
            linewidth=self.ssdp_mean_width,
            label=r"$S_{\boldsymbol{r}}(t)$",
        )

        for ssdp_samples in ssdp_samples[:max_num_samples, ...]:
            axes.plot(
                times,
                ssdp_samples,
                color=self.ssdp_sample_color,
                alpha=self.ssdp_sample_alpha,
                linewidth=self.ssdp_sample_width,
            )

        arrow = FancyArrowPatch(
            posA=(0.0, 0.0),
            posB=(1.0, 0.0),
            arrowstyle="->",
            mutation_scale=10.0,
            color="black",
            alpha=1.0,
            linewidth=0.5,
            zorder=2,
        )
        axes.add_patch(arrow)

        axes.plot(
            np.linspace(0.0, 1.0, 11)[:-1],
            np.linspace(0.0, 0.0, 11)[:-1],
            marker="o",
            markersize=2.0,
            linestyle="none",
            markerfacecolor="gold",
            markeredgecolor="black",
            markeredgewidth=0.5,
        )

        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(-1.1, 1.1)

        axes.axis("off")

        return axes

    def _visualize_prior_distribution(
        self,
        axes: mpl.axes.Axes,
        times: npt.NDArray,
        priors: npt.NDArray,
    ) -> mpl.axes.Axes:
        priors /= np.max(priors)

        axes.fill_between(
            x=times,
            y1=0.0,
            y2=priors,
            color=self.prior_color,
            alpha=self.prior_alpha,
        )
        axes.plot(
            times,
            priors,
            color=self.prior_color,
            linewidth=self.prior_width,
            label=r"$f_{H_{\boldsymbol{r}}}(t)$",
        )

        arrow = FancyArrowPatch(
            posA=(0.0, 0.0),
            posB=(1.0, 0.0),
            arrowstyle="->",
            mutation_scale=10.0,
            color="black",
            alpha=1.0,
            linewidth=0.5,
            zorder=2,
        )
        axes.add_patch(arrow)

        axes.plot(
            np.linspace(0.0, 1.0, 11)[:-1],
            np.linspace(0.0, 0.0, 11)[:-1],
            marker="o",
            markersize=2.0,
            linestyle="none",
            markerfacecolor="gold",
            markeredgecolor="black",
            markeredgewidth=0.5,
        )

        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(-0.1, 1.1)

        axes.axis("off")

        return axes

    def _get_likelihood(self, times: npt.NDArray, colors: npt.NDArray) -> npt.NDArray:
        means = np.cos(2.0 * np.pi * times + np.pi) * 0.25 + 0.25
        return scipy.stats.norm.pdf(colors, means, 0.1)

    def _visualize_joint_distribution(
        self,
        axes: mpl.axes.Axes,
        times: npt.NDArray,
        colors: npt.NDArray,
        priors: npt.NDArray,
        likelihoods: npt.NDArray,
    ) -> mpl.axes.Axes:
        joints = priors * likelihoods
        marginals = np.sum(joints, axis=-1)

        priors /= np.max(priors)
        joints /= np.max(joints)
        marginals /= np.max(marginals)

        axes.plot(
            xs=times,
            ys=0.0,
            zs=priors,
            color=self.prior_color,
            linewidth=self.prior_width,
            label=r"$f_{H_{\boldsymbol{r}}}(t)$",
        )
        axes.add_collection3d(
            Poly3DCollection(
                verts=[
                    [
                        (0.0, 0.0, 0.0),
                        *[(time, 0.0, prior) for time, prior in zip(times, priors, strict=True)],
                        (1.0, 0.0, 0.0),
                    ]
                ],
                facecolor=self.prior_color,
                alpha=self.prior_alpha,
            )
        )

        axes.plot(
            xs=0.0,
            ys=colors,
            zs=marginals,
            color=self.marginal_color,
            linewidth=self.marginal_width,
            label=r"$f_{C_{\boldsymbol{r}}}(c)$",
        )
        axes.add_collection3d(
            Poly3DCollection(
                verts=[
                    [
                        (0.0, 0.0, 0.0),
                        *[
                            (0.0, color, marginal)
                            for color, marginal in zip(colors, marginals, strict=True)
                        ],
                        (0.0, 1.0, 0.0),
                    ]
                ],
                facecolor=self.marginal_color,
                alpha=self.marginal_alpha,
            )
        )

        likelihood = np.interp(self.observation, colors, marginals)

        axes.plot(
            xs=[0.0, 0.0],
            ys=[self.observation, self.observation],
            zs=[0.0, likelihood],
            color=self.observation_color,
            linewidth=self.observation_width,
            linestyle="--",
            # label="Observation",
        )
        axes.plot(
            xs=0.0,
            ys=self.observation,
            zs=likelihood,
            marker="*",
            markersize=7.5,
            color=self.observation_color,
            alpha=self.observation_alpha,
        )
        axes.text(
            x=0.0,
            y=self.observation,
            z=likelihood + 0.15,
            s=r"$f_{C_{\boldsymbol{r}}}(c_{\boldsymbol{r}})$",
            color=self.observation_color,
        )

        times, colors = np.meshgrid(times, colors)

        joint_face_cmap = LinearSegmentedColormap.from_list(
            name="custom",
            colors=["white", self.joint_color],
        )
        axes.plot_surface(
            X=times,
            Y=colors,
            Z=joints,
            cmap=joint_face_cmap,
            alpha=self.joint_alpha,
            rcount=self.resolution,
            ccount=self.resolution,
            norm=PowerNorm(gamma=0.25),
        )
        axes.plot_wireframe(
            X=times,
            Y=colors,
            Z=joints,
            color="black",
            alpha=0.5,
            linewidth=0.5,
            rcount=self.resolution // 10,
            ccount=self.resolution // 10,
        )

        handles, _ = axes.get_legend_handles_labels()
        proxy = Patch(
            color=self.joint_color,
            label=r"$f_{H_{\boldsymbol{r}},C_{\boldsymbol{r}}}(t, c)$",
        )
        handles.insert(2, proxy)

        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(0.0, 1.0)
        axes.set_zlim(0.0, 1.0)

        axes.set_xlabel(r"Time $t$", labelpad=-15)
        axes.set_ylabel(r"Color $c$", labelpad=-15)
        axes.set_zlabel(r"Density", labelpad=-15)

        for axis in [axes.xaxis, axes.yaxis, axes.zaxis]:
            axis.set_ticks(np.linspace(0.0, 1.0, 11))
            axis.set_ticklabels([])
            axis.pane.set_facecolor("white")
            axis.pane.set_edgecolor("black")
            axis._axinfo["grid"]["color"] = (0.0, 0.0, 0.0, 0.1)
            axis._axinfo["tick"]["inward_factor"] = 0
            axis._axinfo["tick"]["outward_factor"] = 0

        axes.set_box_aspect([1.0, 1.0, 0.5])
        axes.view_init(elev=30.0, azim=45.0)

        axes.legend(
            loc="lower center",
            frameon=False,
            handles=handles,
            ncols=4,
            columnspacing=1.0,
            bbox_to_anchor=(0.5, -0.05),
        )

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
        colors = np.linspace(0.0, 1.0, self.resolution)

        marginal_means = np.cos(np.pi * times)
        ssdp_samples = self._get_ssdp_samples(times, marginal_means)

        priors = self._get_first_passage_distributon(ssdp_samples)
        likelihoods = self._get_likelihood(times, colors[..., None])

        plt.rcParams.update({"font.size": self.ssdp_font_size})

        figure_size = _get_figure_size(
            base_width=self.text_width,
            width_ratio=self.ssdp_width_ratio,
            aspect_ratio=self.ssdp_aspect_ratio,
        )
        figure = plt.figure(figsize=figure_size)

        axes = figure.add_subplot(111)
        self._visualize_ssdp_samples(
            axes=axes,
            times=times,
            marginal_means=marginal_means,
            ssdp_samples=ssdp_samples,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            self.output_dir / "ssdp_samples.svg",
            bbox_inches="tight",
            pad_inches=0.0,
            transparent=True,
        )

        plt.rcParams.update({"font.size": self.prior_font_size})

        figure_size = _get_figure_size(
            base_width=self.text_width,
            width_ratio=self.prior_width_ratio,
            aspect_ratio=self.prior_aspect_ratio,
        )
        figure = plt.figure(figsize=figure_size)

        axes = figure.add_subplot(111)
        self._visualize_prior_distribution(
            axes=axes,
            times=times,
            priors=priors,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            self.output_dir / "prior_distribution.svg",
            bbox_inches="tight",
            pad_inches=0.0,
            transparent=True,
        )

        plt.rcParams.update({"font.size": self.joint_font_size})

        figure_size = _get_figure_size(
            base_width=self.text_width,
            width_ratio=self.joint_width_ratio,
            aspect_ratio=self.joint_aspect_ratio,
        )
        figure = plt.figure(figsize=figure_size)

        axes = figure.add_subplot(111, projection="3d")
        self._visualize_joint_distribution(
            axes=axes,
            times=times,
            colors=colors,
            priors=priors,
            likelihoods=likelihoods,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            self.output_dir / "joint_distribution.svg",
            bbox_inches="tight",
            pad_inches=0.0,
            transparent=True,
        )


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        OverviewFigureCreator,
        config=(tyro.conf.AvoidSubcommands,),
    )()
