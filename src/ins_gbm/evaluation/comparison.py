from __future__ import annotations

from typing import TYPE_CHECKING, Union

import polars as pl

if TYPE_CHECKING:
    from ins_gbm.evaluation.report import EvaluationReport
    from ins_gbm.evaluation.cv_report import CVResult


def compare_reports(
    reports: dict[str, "Union[EvaluationReport, CVResult]"],
) -> pl.DataFrame:
    """Compare two or more EvaluationReport or CVResult objects side by side.

    Returns a DataFrame with one row per metric, one column per report key,
    and a 'preferred' column indicating which report wins on each metric.
    CV values are formatted as 'mean +/- std'; single test-set values as 'mean'.
    """
    from ins_gbm.evaluation.report import EvaluationReport
    from ins_gbm.evaluation.cv_report import CVResult
    from ins_gbm.evaluation.metrics import METRIC_DIRECTIONS

    for name, report in reports.items():
        if isinstance(report, EvaluationReport) and report._comparison_models is not None:
            raise ValueError(
                f"Report {name!r} is a comparison-mode EvaluationReport and cannot be "
                "used with compare_reports(). Pass individual single-model reports or "
                "CVResult objects instead."
            )

    report_data: dict[str, dict[str, tuple[float, float | None]]] = {}
    all_metrics: set[str] = set()

    for name, report in reports.items():
        if isinstance(report, CVResult):
            gbm_rows = report.summary.filter(pl.col("model") == "gbm")
            d: dict[str, tuple[float, float | None]] = {}
            for row in gbm_rows.iter_rows(named=True):
                d[row["metric"]] = (row["mean"], row["std"])
            report_data[name] = d
        else:
            d = {}
            for row in report.metrics().iter_rows(named=True):
                d[row["metric"]] = (row["value"], None)
            report_data[name] = d
        all_metrics.update(report_data[name].keys())

    names = list(reports.keys())
    rows = []

    for metric in sorted(all_metrics):
        row: dict = {"metric": metric}
        values: dict[str, float | None] = {}

        for name in names:
            if metric in report_data[name]:
                mean, std = report_data[name][metric]
                row[name] = f"{mean:.4f} +/- {std:.4f}" if std is not None else f"{mean:.4f}"
                values[name] = mean
            else:
                row[name] = None
                values[name] = None

        direction = METRIC_DIRECTIONS.get(metric, "lower")
        valid = {n: v for n, v in values.items() if v is not None}

        if not valid:
            row["preferred"] = None
        elif len(valid) == 1:
            row["preferred"] = next(iter(valid))
        else:
            best_val = max(valid.values()) if direction == "higher" else min(valid.values())
            best_names = [n for n, v in valid.items() if abs(v - best_val) < 1e-6]
            row["preferred"] = "tie" if len(best_names) > 1 else best_names[0]

        rows.append(row)

    return pl.DataFrame(rows)
