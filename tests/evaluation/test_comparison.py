import polars as pl
import pytest

from ins_gbm.data.loader import load_model_data
from ins_gbm.data.model_data import ModelData
from ins_gbm.data.schema import infer_schema
from ins_gbm.data.splitter import TrainTestSplit
from ins_gbm.evaluation.comparison import compare_reports
from ins_gbm.evaluation.cv_report import CrossValidationReport
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import ModelPipeline, ModelRecipe


def _run_pipeline(poisson_parquet, seed: int):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    return ModelPipeline(
        data=data,
        split=TrainTestSplit(train_ratio=0.7, seed=seed),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()


def _run_cv(poisson_raw, seed: int):
    schema = infer_schema(poisson_raw, ["x1", "x3"])
    data = ModelData(
        features=poisson_raw.select(["x1", "x3"]),
        target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"],
        weight=None,
        feature_names=["x1", "x3"],
        schema=schema,
        objective="poisson",
    ).validate()
    return CrossValidationReport(
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        data=data,
        n_folds=3,
        seed=seed,
    ).run()


def test_compare_two_evaluation_reports(poisson_parquet):
    r1 = _run_pipeline(poisson_parquet, seed=0).report
    r2 = _run_pipeline(poisson_parquet, seed=1).report
    df = compare_reports({"model_a": r1, "model_b": r2})
    assert "metric" in df.columns
    assert "model_a" in df.columns
    assert "model_b" in df.columns
    assert "preferred" in df.columns


def test_compare_two_cv_results(poisson_raw):
    r1 = _run_cv(poisson_raw, seed=0)
    r2 = _run_cv(poisson_raw, seed=1)
    df = compare_reports({"cv_a": r1, "cv_b": r2})
    assert set(df.columns) == {"metric", "cv_a", "cv_b", "preferred"}
    # CV values should show +/- notation
    gini_row = df.filter(pl.col("metric") == "gini")
    assert "+/-" in gini_row["cv_a"][0]


def test_compare_mixed_inputs(poisson_parquet, poisson_raw):
    eval_report = _run_pipeline(poisson_parquet, seed=0).report
    cv_result = _run_cv(poisson_raw, seed=0)
    df = compare_reports({"single_test": eval_report, "cv": cv_result})
    assert df.height > 0
    # single_test values should NOT have +/-
    gini_row = df.filter(pl.col("metric") == "gini")
    assert "+/-" not in gini_row["single_test"][0]
    assert "+/-" in gini_row["cv"][0]


def test_preferred_uses_metric_direction(poisson_parquet):
    r1 = _run_pipeline(poisson_parquet, seed=0).report
    r2 = _run_pipeline(poisson_parquet, seed=1).report
    df = compare_reports({"a": r1, "b": r2})
    # For each metric, preferred must be one of the report names or "tie"
    valid = {"a", "b", "tie"}
    for val in df["preferred"].to_list():
        assert val in valid


def test_preferred_is_tie_for_identical_inputs(poisson_parquet):
    r1 = _run_pipeline(poisson_parquet, seed=42).report
    df = compare_reports({"x": r1, "y": r1})
    assert (df["preferred"] == "tie").all()


def test_comparison_mode_report_raises(poisson_parquet):
    from ins_gbm.evaluation.report import EvaluationReport
    r1 = _run_pipeline(poisson_parquet, seed=0)
    r2 = _run_pipeline(poisson_parquet, seed=1)
    comparison_report = EvaluationReport.compare(
        models={
            "a": (r1.fitted_model, r1.train_data, r1.test_data),
            "b": (r2.fitted_model, r2.train_data, r2.test_data),
        },
        test_data=r1.test_data,
    )
    with pytest.raises(ValueError, match="comparison-mode"):
        compare_reports({"bad": comparison_report})


def test_compare_reports_exported_from_evaluation_package():
    from ins_gbm.evaluation import compare_reports as fn
    assert callable(fn)
