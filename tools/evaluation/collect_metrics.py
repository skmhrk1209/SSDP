import dataclasses
import enum
import json
import re
from pathlib import Path

import loguru
import pandas as pd
import tyro
from pandas.core.groupby import SeriesGroupBy


@dataclasses.dataclass
class MetricsCollector:
    class OutputFormat(enum.StrEnum):
        CSV = enum.auto()
        JSON = enum.auto()
        LATEX = enum.auto()

    class SelectionOp(enum.StrEnum):
        MIN = enum.auto()
        MAX = enum.auto()
        MEDIAN = enum.auto()

    class MedianMode(enum.StrEnum):
        LOWER = enum.auto()
        HIGHER = enum.auto()

    input_dir: Path
    output_dir: Path | None = None

    model_pattern: str = (
        r"(neus|oav|ssdp)"
        r"(?:-(facto|official))?"
        r"(?:-(base|half|double|quarter|quadruple))?"
    )
    scene_pattern: str = (
        r"(?:neus|oav|ssdp)"
        r"(?:-(?:facto|official))?"
        r"(?:-(?:base|half|double|quarter|quadruple))?"
        r"-(.+)"
    )

    criteria: str = "unmasked_average_distance"
    selection_op: SelectionOp = SelectionOp.MEDIAN
    median_mode: MedianMode = MedianMode.LOWER

    def _search_pattern(self, pattern: str | re.Pattern[str], path: Path) -> str | None:
        for part in reversed(path.parts):
            match = re.search(pattern, part)
            if match:
                return "-".join(filter(None, match.groups()))

    def _collect_metrics(self) -> pd.DataFrame:
        model_pattern = re.compile(self.model_pattern)
        scene_pattern = re.compile(self.scene_pattern)

        records = []

        for metric_dir in self.input_dir.glob("**/evaluation"):
            model_name = self._search_pattern(model_pattern, metric_dir)
            if not model_name:
                continue

            scene_name = self._search_pattern(scene_pattern, metric_dir)
            if not scene_name:
                continue

            record = dict(
                model_name=model_name,
                scene_name=scene_name,
                metric_dir=str(metric_dir),
            )

            for metric_file in metric_dir.glob("**/metrics.json"):
                with open(metric_file) as fp:
                    metrics = json.load(fp)
                prefix = metric_file.parent.name
                record |= {f"{prefix}_{key}": value for key, value in metrics.items()}

            records.append(record)

        loguru.logger.info(f"Found {len(records)} evaluation records.")

        return pd.DataFrame(records)

    def _idxmedian(self, groups: SeriesGroupBy, mode: MedianMode) -> pd.Series:
        values = groups.quantile(0.5, interpolation=mode.value)
        indices = groups.apply(lambda group: (group - values.loc[group.name]).abs().idxmin())
        return indices

    def _select_model(self, metrics: pd.DataFrame) -> pd.DataFrame:
        isna = metrics[self.criteria].isna()
        if isna.any():
            for _, row in metrics[isna].iterrows():
                loguru.logger.warning(
                    f"Missing criteria <{self.criteria}> for <{row['metric_dir']}>."
                )
            metrics = metrics[~isna]

        groups = metrics.groupby(["model_name", "scene_name"])
        groups = groups[self.criteria]

        if self.selection_op is self.SelectionOp.MIN:
            indices = groups.idxmin()
        elif self.selection_op is self.SelectionOp.MAX:
            indices = groups.idxmax()
        elif self.selection_op is self.SelectionOp.MEDIAN:
            indices = self._idxmedian(groups, self.median_mode)

        metrics = metrics.loc[indices].reset_index(drop=True)

        return metrics

    def __call__(self) -> None:
        metrics = self._collect_metrics()
        metrics = self._select_model(metrics)

        statistics = metrics.melt(
            id_vars=["model_name", "scene_name", "metric_dir"],
            var_name="metric_name",
            value_name="metric_value",
        )
        statistics = statistics.groupby(["model_name", "metric_name"])
        statistics = statistics["metric_value"].describe()
        statistics = statistics.reset_index(drop=False)

        output_dir = self.output_dir or self.input_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = output_dir / "metrics.json"
        statistics_file = output_dir / "statistics.json"

        metrics.to_json(metrics_file, orient="records", indent=4)
        statistics.to_json(statistics_file, orient="records", indent=4)

        loguru.logger.success(
            f"Saved the evaluation metrics and statistics to "
            f"<{metrics_file}> and <{statistics_file}>."
        )


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        MetricsCollector,
        config=(tyro.conf.AvoidSubcommands,),
    )()
