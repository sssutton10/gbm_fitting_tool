from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from ins_gbm.data.model_data import ModelData
    from ins_gbm.evaluation.cv_report import CVResult
    from ins_gbm.evaluation.report import EvaluationReport


def compare_cv_double_lift(
    reference: CVResult,
    candidate: CVResult,
    *,
    data: ModelData | None = None,
    n_bins: int = 10,
    deviation: str = "absolute",
) -> pl.DataFrame:
    """Score aligned OOF predictions; positive scores favor the candidate.

    If both results were loaded from disk, supply the original training data in
    its original row order. No training data is stored in the CV artifact.

    Args:
        reference ('CVResult'): Reference cross-validation result.
        candidate ('CVResult'): Candidate cross-validation result.
        data (Optional['ModelData']): Model data to fit, transform, predict, or evaluate.
            Optional.
        n_bins (int): Number of bins used to summarize predictions. Defaults to 10.
        deviation (str): Deviation measure: "absolute" or "relative". Defaults to 'absolute'.
    """
    import numpy as np

    from ins_gbm.evaluation.cv_report import GBM_MODEL_LABEL, _cv_data_signature
    from ins_gbm.evaluation.metrics import (
        _double_lift_metric_inputs,
        double_lift_score,
        double_lift_table,
    )

    for label, result in (("reference", reference), ("candidate", candidate)):
        if (
            result.predictions is None
            or result.row_folds is None
            or result.data_signature is None
            or result.objective is None
            or GBM_MODEL_LABEL not in result.predictions.columns
        ):
            raise ValueError(
                f"{label} CVResult lacks aligned OOF predictions or provenance"
            )
    if reference.data_signature != candidate.data_signature:
        raise ValueError(
            "CV results have different evaluation rows, weights, or fold assignments"
        )
    if (
        reference.predictions.height != candidate.predictions.height
        or not reference.row_folds.equals(candidate.row_folds)
    ):
        raise ValueError("CV results have different row or fold alignment")

    if data is not None:
        data.validate(require_multiple_folds=False)
        actual, exposure, weight, objective = (
            data.target,
            data.exposure,
            data.weight,
            data.objective,
        )
    else:
        source = candidate if candidate.actual is not None else reference
        actual, exposure, weight, objective = (
            source.actual,
            source.exposure,
            source.weight,
            source.objective,
        )
    if actual is None:
        raise ValueError(
            "Evaluation data is needed for double lift; "
            "pass data=original_training_data"
        )
    if (
        objective != candidate.objective
        or _cv_data_signature(
            objective,
            actual,
            exposure,
            weight,
            candidate.row_folds,
        )
        != candidate.data_signature
    ):
        raise ValueError("Evaluation data does not match the CV results or row order")

    prediction_a = reference.predictions[GBM_MODEL_LABEL]
    prediction_b = candidate.predictions[GBM_MODEL_LABEL]

    def score(indices: np.ndarray) -> float:
        """Score the selected aligned evaluation rows.

        Args:
            indices (np.ndarray): Row indices to retain.
        """
        take = indices.tolist()
        a, p_a, p_b, w = _double_lift_metric_inputs(
            objective,
            actual.gather(take),
            prediction_a.gather(take),
            prediction_b.gather(take),
            exposure.gather(take) if exposure is not None else None,
            weight.gather(take) if weight is not None else None,
        )
        table = double_lift_table(
            a,
            p_a,
            p_b,
            weights=w,
            n_bins=min(n_bins, len(indices)),
        )
        return double_lift_score(table, deviation=deviation)

    n_rows = candidate.predictions.height
    rows = [
        {
            "scope": "overall",
            "fold": None,
            "n_rows": n_rows,
            "score": score(np.arange(n_rows)),
        }
    ]
    folds = candidate.row_folds.to_numpy()
    for fold in candidate.row_folds.unique(maintain_order=True).to_list():
        indices = np.flatnonzero(folds == fold)
        rows.append(
            {
                "scope": "fold",
                "fold": str(fold),
                "n_rows": len(indices),
                "score": score(indices) if len(indices) >= 2 else None,
            }
        )
    return pl.DataFrame(rows)


def compare_reports(
    reports: dict[str, EvaluationReport | CVResult],
    *,
    data: ModelData | None = None,
) -> pl.DataFrame:
    """Compare two or more EvaluationReport or CVResult objects side by side.

    Returns a DataFrame with one row per metric, one column per report key,
    and a 'preferred' column indicating which report wins on each metric.
    CV values are formatted as 'mean +/- std'; single test-set values as 'mean'.
    For two aligned CV results, the pooled double-lift score appears under the
    second report name; positive scores favor that report. Supply *data* when
    both CV results were loaded from disk. Use compare_cv_double_lift() for
    per-fold scores.

    Args:
        reports (dict[str, 'Union[EvaluationReport, CVResult]']): Mapping from display names to
            evaluation or CV reports.
        data (Optional['ModelData']): Model data to fit, transform, predict, or evaluate.
            Optional.
    """
    from ins_gbm.evaluation.cv_report import CVResult
    from ins_gbm.evaluation.report import EvaluationReport

    for name, report in reports.items():
        if isinstance(report, EvaluationReport) and report.is_comparison_mode:
            raise ValueError(
                f"Report {name!r} is a comparison-mode EvaluationReport and cannot be "
                "used with compare_reports(). Pass individual single-model reports or "
                "CVResult objects instead."
            )

    report_data = {name: _report_metrics(report) for name, report in reports.items()}
    all_metrics = set().union(*(metrics.keys() for metrics in report_data.values()))
    names = list(reports.keys())
    rows = []
    for metric in sorted(all_metrics):
        row: dict = {"metric": metric}
        values: dict[str, float | None] = {}
        for name in names:
            if metric in report_data[name]:
                mean, std = report_data[name][metric]
                row[name] = (
                    f"{mean:.4f} +/- {std:.4f}" if std is not None else f"{mean:.4f}"
                )
                values[name] = mean
            else:
                row[name] = None
                values[name] = None
        row["preferred"] = _preferred_report(metric, values)
        rows.append(row)
    if len(names) == 2 and all(isinstance(reports[name], CVResult) for name in names):
        reference, candidate = (reports[name] for name in names)
        has_evaluation_data = (
            data is not None
            or candidate.actual is not None
            or reference.actual is not None
        )
        if _aligned_cv_pair(reference, candidate) and has_evaluation_data:
            score = compare_cv_double_lift(
                reference,
                candidate,
                data=data,
            )["score"][0]
            rows.append(
                {
                    "metric": "double_lift_score",
                    names[0]: None,
                    names[1]: f"{score:+.4f}",
                    "preferred": (
                        "tie"
                        if abs(score) < 1e-6
                        else names[1]
                        if score > 0
                        else names[0]
                    ),
                }
            )
            rows.sort(key=lambda row: row["metric"])

    return pl.DataFrame(rows)


def _report_metrics(report) -> dict[str, tuple[float, float | None]]:
    """Read only the fitted model's standalone metrics from a report.

    Args:
        report (object): Evaluation or cross-validation report to summarize.
    """
    from ins_gbm.evaluation.cv_report import GBM_MODEL_LABEL, CVResult

    if isinstance(report, CVResult):
        gbm_rows = report.summary.filter(pl.col("model") == GBM_MODEL_LABEL)
        gbm_rows = gbm_rows.filter(pl.col("metric") != "double_lift_score")
        return {
            row["metric"]: (row["mean"], row["std"])
            for row in gbm_rows.iter_rows(named=True)
        }
    return {
        row["metric"]: (row["value"], None)
        for row in report._single_metrics().iter_rows(named=True)
    }


def _preferred_report(metric: str, values: dict[str, float | None]) -> str | None:
    """Choose the report with the best value for a metric.

    Args:
        metric (str): Metric name used for scoring or selection.
        values (dict[str, float | None]): Metric values keyed by report name.
    """
    from ins_gbm.evaluation.metrics import METRIC_DIRECTIONS

    valid = {name: value for name, value in values.items() if value is not None}
    if not valid:
        return None
    if len(valid) == 1:
        return next(iter(valid))
    direction = METRIC_DIRECTIONS.get(metric, "lower")
    best = max(valid.values()) if direction == "higher" else min(valid.values())
    winners = [name for name, value in valid.items() if abs(value - best) < 1e-6]
    return "tie" if len(winners) > 1 else winners[0]


def _aligned_cv_pair(reference: CVResult, candidate: CVResult) -> bool:
    """Check provenance required for a pooled double-lift score.

    Args:
        reference ('CVResult'): Reference cross-validation result.
        candidate ('CVResult'): Candidate cross-validation result.
    """
    return (
        reference.data_signature is not None
        and reference.data_signature == candidate.data_signature
        and reference.predictions is not None
        and candidate.predictions is not None
        and reference.objective is not None
        and candidate.objective is not None
        and reference.row_folds is not None
        and candidate.row_folds is not None
        and reference.row_folds.equals(candidate.row_folds)
    )
