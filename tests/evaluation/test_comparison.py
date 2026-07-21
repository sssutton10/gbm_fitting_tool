import polars as pl

from ins_gbm.data.loader import load_model_data
from ins_gbm.evaluation.comparison import compare_reports
from ins_gbm.evaluation.report import EvaluationReport
from ins_gbm.models.lightgbm import LightGBMModel


def test_compare_reports_accepts_explicit_reports(poisson_parquet):
    data = load_model_data(path=str(poisson_parquet), target="claim_count", exposure="exposure", feature_cols=["x1", "x3"], objective="poisson")
    fitted = LightGBMModel(objective="poisson").fit(data, params={"n_estimators": 10, "verbose": -1})
    report = EvaluationReport(fitted, data, data)
    comparison = compare_reports({"one": report, "two": report})
    assert {"metric", "one", "two", "preferred"}.issubset(comparison.columns)


def test_compare_reports_uses_fitted_model_metrics_not_benchmark_predictions(poisson_parquet):
    data = load_model_data(path=str(poisson_parquet), target="claim_count", exposure="exposure", feature_cols=["x1", "x3"], objective="poisson")
    fitted = LightGBMModel(objective="poisson").fit(data, params={"n_estimators": 10, "verbose": -1})
    report = EvaluationReport(
        fitted,
        data,
        data,
        comparison_predictions={"benchmark": pl.Series([1.0] * data.n_rows)},
    )

    comparison = compare_reports({"model": report})
    model_metrics = {
        row["metric"]: f'{row["value"]:.4f}'
        for row in report._single_metrics().iter_rows(named=True)
    }
    benchmark_metrics = {
        row["metric"]: f'{row["value"]:.4f}'
        for row in report.metrics().filter(pl.col("model") == "benchmark").iter_rows(named=True)
    }
    comparison_metrics = dict(zip(comparison["metric"].to_list(), comparison["model"].to_list()))

    assert comparison_metrics == model_metrics
    assert any(model_metrics[metric] != benchmark_metrics[metric] for metric in model_metrics)
