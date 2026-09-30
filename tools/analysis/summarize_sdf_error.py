import dataclasses
import json
from pathlib import Path

import loguru
import tyro

from tools.analysis.evaluate_approx_error import Variant


@dataclasses.dataclass
class SDFErrorSummarizer:
    # NOTE: The table of the reconstructed geometry (`evaluate_sdf_error.py`): the thickness along the axis rays and
    # the learned SDF on the exact faces, pooled over all the slabs of the scene, per scene and renderer.
    data_dir: Path
    output_file: Path

    scene_ids: tuple[str, ...] = ("k1_w0.05", "k3_w0.05", "k1_w0.025", "k3_w0.025")
    variants: tuple[Variant, ...] = (Variant.NA, Variant.NA_UP, Variant.BF, Variant.BF_UP)

    def __call__(self) -> None:
        lines = [
            "===== reconstructed slabs: thickness along the axis rays of all the slabs (mean, median [25%, 75%], vanished rays) | learned SDF on the exact faces of all the slabs: |SDF| mean, median / 75% / 95%, signed mean, signed median"
        ]
        for scene_id in self.scene_ids:
            for variant in self.variants:
                with open(self.data_dir / f"{scene_id}_{variant}.json") as fp:
                    record = json.load(fp)
                metrics = record["metrics"]
                # NOTE: The pooled thickness of the runs evaluated before it was recorded is that of their single slab.
                slabs = metrics.get(
                    "all_slabs", metrics["slabs"][0] if len(metrics["slabs"]) == 1 else None
                )
                faces = metrics["faces"]["all_faces"]
                thickness = f"mean {slabs['thickness']['mean']:.4f}, {slabs['thickness']['q50']:.4f} [{slabs['thickness']['q25']:.4f}, {slabs['thickness']['q75']:.4f}] ({slabs['num_vanished_rays']}/{slabs['num_rays']})"
                lines.append(
                    f"{scene_id:10s} {variant:6s} true {slabs['true_thickness']:.4f}: {thickness} | "
                    f"mean {faces['absolute_sdf']['mean']:.4f}, {faces['absolute_sdf']['q50']:.4f} / {faces['absolute_sdf']['q75']:.4f} / {faces['absolute_sdf']['q95']:.4f}, "
                    f"signed mean {faces['signed_sdf']['mean']:+.4f}, median {faces['signed_sdf']['q50']:+.4f}"
                )

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as fp:
            fp.write("\n".join(lines) + "\n")

        loguru.logger.success(f"Saved the summary to <{self.output_file}>.")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        SDFErrorSummarizer,
        config=(tyro.conf.AvoidSubcommands,),
    )()
