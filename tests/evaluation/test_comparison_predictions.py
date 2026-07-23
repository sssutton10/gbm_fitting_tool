import polars as pl

from ins_gbm.data.loader import load_model_data
from ins_gbm.evaluation.report import EvaluationReport
from ins_gbm.models.lightgbm import LightGBMModel


def test_report_comparison_predictions(poisson_parquet):
    data = load_model_data(path=str(poisson_parquet), target="claim_count", exposure="exposure", feature_cols=["x1", "x3"], objective="poisson")
    fitted = LightGBMModel(objective="poisson").fit(data, params={"n_estimators": 10, "verbose": -1})
    report = EvaluationReport(fitted, data, data, comparison_predictions={"legacy": pl.Series([1.0] * data.n_rows)})
    assert set(report.metrics()["model"].unique()) == {"GBM", "legacy"}
    assert "double_lift_score" in report.metrics()["metric"]
    assert report.plot_double_lift("legacy") is not None
    assert isinstance(report.double_lift_score("legacy"), float)
