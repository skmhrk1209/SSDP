import dataclasses
import json
from pathlib import Path

import loguru
import tyro

from tools.analysis.evaluate_approx_error import Variant


@dataclasses.dataclass
class SDFErrorSummarizer:
    # NOTE: The table of the reconstructed geometry (`evaluate_sdf_error.py`): the thickness of the slabs along
    # the axis rays and the learned SDF on the exact faces, per scene and renderer.
    data_dir: Path
    output_file: Path

    scene_ids: tuple[str, ...] = ("k1_w0.05", "k1_w0.025")
    variants: tuple[Variant, ...] = (Variant.NA, Variant.NA_UP, Variant.BF, Variant.BF_UP)

    def __call__(self) -> None:
        lines = [
            "===== reconstructed slab: thickness along the axis rays (median [25%, 75%], vanished rays) | learned SDF on the exact faces: |SDF| median / 75% / 95%, signed median"
        ]
        for scene_id in self.scene_ids:
            for variant in self.variants:
                with open(self.data_dir / f"{scene_id}_{variant}.json") as fp:
                    record = json.load(fp)
                slabs, faces = record["metrics"]["slabs"], record["metrics"]["faces"]["all_faces"]
                thickness = " ".join(
                    f"{s['thickness']['q50']:.4f} [{s['thickness']['q25']:.4f}, {s['thickness']['q75']:.4f}] ({s['num_vanished_rays']}/{s['num_rays']})"
                    for s in slabs
                )
                lines.append(
                    f"{scene_id:10s} {variant:6s} true {slabs[0]['true_thickness']:.4f}: {thickness} | "
                    f"{faces['absolute_sdf']['q50']:.4f} / {faces['absolute_sdf']['q75']:.4f} / {faces['absolute_sdf']['q95']:.4f}, "
                    f"signed {faces['signed_sdf']['q50']:+.4f}"
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
