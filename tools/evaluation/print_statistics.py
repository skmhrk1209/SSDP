import dataclasses
import enum
import json
from pathlib import Path

import pandas as pd
import tyro
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text


@dataclasses.dataclass
class StatisticsPrinter:
    class OutputFormat(enum.StrEnum):
        TERM = enum.auto()
        LATEX = enum.auto()

    input_files: tuple[Path, ...]
    metric_names: dict[str, str] | None = dataclasses.field(
        default_factory=lambda: dict(
            unmasked_average_distance="CD",
            unmasked_average_ratio_25="F1@2.5",
            unmasked_average_ratio_50="F1@5.0",
            masked_average_distance="CD (M)",
            masked_average_ratio_25="F1@2.5 (M)",
            masked_average_ratio_50="F1@5.0 (M)",
            depth_crps="CRPS",
            depth_mae="MAE",
            depth_ece="ECE",
            depth_sharpness="Sharp.",
        ),
    )
    higher_is_better: tuple[str, ...] = ("ratio",)
    output_format: OutputFormat = OutputFormat.TERM
    model_pattern: str | None = None
    statistic_key: str = "mean"
    best_color: str = "bright_red"
    precision: int = 2
    strict: bool = False

    def _is_higher_better(self, metric_name: str) -> bool:
        return any(substr in metric_name for substr in self.higher_is_better)

    def _get_metric_names(self, statistics: pd.DataFrame) -> list[str]:
        metric_names = list(statistics.columns)
        if self.metric_names:
            metric_names = [
                metric_name for metric_name in self.metric_names if metric_name in metric_names
            ]
        return metric_names

    def _get_best_model_names(self, statistics: pd.DataFrame) -> dict[str, str]:
        best_model_names = {}
        for metric_name in self._get_metric_names(statistics):
            metrics = statistics[metric_name].dropna()
            if metrics.empty:
                continue
            best_model_name = (
                metrics.idxmax() if self._is_higher_better(metric_name) else metrics.idxmin()
            )
            best_model_names[metric_name] = best_model_name
        return best_model_names

    def _latex_escape(self, text: str) -> str:
        for old, new in (
            ("&", r"\&"),
            ("%", r"\%"),
            ("$", r"\$"),
            ("#", r"\#"),
            ("_", r"\_"),
            ("{", r"\{"),
            ("}", r"\}"),
        ):
            text = text.replace(old, new)
        return text

    def __call__(self) -> None:
        statistics = []
        for input_file in self.input_files:
            with open(input_file) as fp:
                statistics.extend(json.load(fp))

        statistics = pd.DataFrame(statistics)

        if self.model_pattern:
            statistics = statistics[
                statistics["model_name"].str.contains(
                    pat=self.model_pattern,
                    regex=True,
                    na=False,
                )
            ]

        counts = statistics.groupby("model_name")["count"]
        nonzero_counts = statistics[statistics["count"] > 0].groupby("model_name")["count"]
        num_uniques = nonzero_counts.nunique(dropna=True)
        if self.strict and not num_uniques[num_uniques > 1].empty:
            raise ValueError(
                "Found inconsistent non-zero `count` values across metrics. "
                "Expected exactly one shared count per model."
            )
        counts = nonzero_counts.first().combine_first(counts.first())

        statistics = statistics.pivot_table(
            index="model_name",
            columns="metric_name",
            values=self.statistic_key,
        )
        statistics = statistics.join(counts.rename("count"))
        statistics = statistics.sort_index()

        if self.output_format is self.OutputFormat.TERM:
            self._print_terminal(statistics)
        elif self.output_format is self.OutputFormat.LATEX:
            self._print_latex(statistics)

    def _print_terminal(self, statistics: pd.DataFrame) -> None:
        metric_names = self._get_metric_names(statistics)
        best_model_names = self._get_best_model_names(statistics)

        table = Table(
            title=",\n".join(f"📊 {input_file}" for input_file in self.input_files),
            box=box.HORIZONTALS,
        )
        table.add_column("🤖 Model", justify="left")
        table.add_column("#", justify="right")

        for metric_name in metric_names:
            direction = "\u2191" if self._is_higher_better(metric_name) else "\u2193"
            metric_name = self.metric_names[metric_name] if self.metric_names else metric_name
            table.add_column(f"{metric_name} ({direction})", justify="right")

        for model_name, row in statistics.iterrows():
            count = row["count"]
            count_text = "\u2014" if pd.isna(count) else str(int(count))
            metric_texts = []
            for metric_name in metric_names:
                metric = row[metric_name]
                style = "default"
                if pd.isna(metric):
                    metric_text = "\u2014"
                    style = "dim"
                else:
                    metric_text = f"{metric:.{self.precision}f}"
                    if model_name == best_model_names.get(metric_name):
                        style = f"bold {self.best_color}"
                metric_texts.append(Text(metric_text, style=style))
            table.add_row(model_name, count_text, *metric_texts)

        console = Console()
        console.print()
        console.print(table)
        console.print()

    def _print_latex(self, statistics: pd.DataFrame) -> None:
        metric_names = self._get_metric_names(statistics)
        best_model_names = self._get_best_model_names(statistics)

        headers = ["Model"]
        for metric_name in metric_names:
            direction = r"$\uparrow$" if self._is_higher_better(metric_name) else r"$\downarrow$"
            metric_name = self.metric_names[metric_name] if self.metric_names else metric_name
            headers.append(f"{self._latex_escape(metric_name)} ({direction})")

        table = [
            rf"\begin{{tabular}}{{l{'r' * len(metric_names)}}}",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
        ]

        for model_name, row in statistics.iterrows():
            metric_texts = [self._latex_escape(model_name)]
            for metric_name in metric_names:
                metric = row[metric_name]
                if pd.isna(metric):
                    metric_text = "--"
                else:
                    metric_text = f"{metric:.{self.precision}f}"
                    if model_name == best_model_names.get(metric_name):
                        metric_text = rf"\textbf{{{metric_text}}}"
                metric_texts.append(metric_text)
            table.append(" & ".join(metric_texts) + r" \\")

        table.extend([r"\bottomrule", r"\end{tabular}"])
        table = "\n".join(table)

        print(table)


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_blue")
    tyro.cli(
        StatisticsPrinter,
        config=(tyro.conf.AvoidSubcommands,),
    )()
