import os

from ins_gbm.data.loader import load_model_data
from ins_gbm.evaluation.report import EvaluationReport
from ins_gbm.models.lightgbm import LightGBMModel


def _fitted(path):
    data = load_model_data(path=str(path), target="claim_count", exposure="exposure", feature_cols=["x1", "x3"], objective="poisson")
    return data, LightGBMModel(objective="poisson").fit(data, params={"n_estimators": 10, "verbose": -1})


def test_report_metrics_and_export(poisson_parquet, tmp_path):
    data, fitted = _fitted(poisson_parquet)
    report = EvaluationReport(fitted_model=fitted, evaluation_data=data, train_data=data)
    assert "metric" in report.metrics().columns
    report.export(str(tmp_path))
    assert os.path.exists(tmp_path / "metrics.csv")


def test_compare_does_not_accept_redundant_evaluation_argument(poisson_parquet):
    data, fitted = _fitted(poisson_parquet)
    report = EvaluationReport.compare({"a": (fitted, data, data), "b": (fitted, data, data)})
    assert report.is_comparison_mode
    assert set(report.metrics()["model"].unique()) == {"a", "b"}
    assert "double_lift_score" in report.metrics()["metric"]
    assert report.double_lift_score() == 0.0
    assert report.plot_double_lift() is not None


def test_named_model_comparison_exports_double_lift(poisson_parquet, tmp_path):
    data, fitted = _fitted(poisson_parquet)
    report = EvaluationReport.compare({
        "a": (fitted, data, data),
        "b": (fitted, data, data),
    })
    report.export(str(tmp_path))
    assert os.path.exists(tmp_path / "double_lift_a_vs_b.png")
