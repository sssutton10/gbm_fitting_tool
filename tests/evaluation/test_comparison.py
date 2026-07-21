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
