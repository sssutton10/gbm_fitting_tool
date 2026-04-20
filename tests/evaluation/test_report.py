"""Tests for EvaluationReport."""
import os
import matplotlib
matplotlib.use("Agg")
import polars as pl
import pytest
from ins_gbm.data.loader import load_model_data
from ins_gbm.data.splitter import TrainTestSplit
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.evaluation.report import EvaluationReport


def _fit_poisson(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    train, test = TrainTestSplit(seed=42).split(data)
    fitted = LightGBMModel(objective="poisson").fit(train, params={"n_estimators": 20, "verbose": -1})
    return fitted, train, test


def test_report_metrics_returns_dataframe(poisson_parquet):
    fitted, train, test = _fit_poisson(poisson_parquet)
    report = EvaluationReport(fitted_model=fitted, test_data=test, train_data=train)
    metrics = report.metrics()
    assert isinstance(metrics, pl.DataFrame)
    assert "metric" in metrics.columns
    assert "value" in metrics.columns
    metric_names = metrics["metric"].to_list()
    assert "poisson_deviance" in metric_names
    assert "gini" in metric_names
    assert "rmse" in metric_names


def test_report_export_creates_files(tmp_path, poisson_parquet):
    fitted, train, test = _fit_poisson(poisson_parquet)
    report = EvaluationReport(fitted_model=fitted, test_data=test, train_data=train)
    report.export(str(tmp_path))
    assert os.path.exists(str(tmp_path / "metrics.csv"))
    assert os.path.exists(str(tmp_path / "lift.png"))
    assert os.path.exists(str(tmp_path / "ave.png"))
    assert os.path.exists(str(tmp_path / "calibration.png"))
    assert os.path.exists(str(tmp_path / "feature_importance.png"))


def test_report_comparison_mode(poisson_parquet):
    fitted_a, train, test = _fit_poisson(poisson_parquet)
    fitted_b, _, _ = _fit_poisson(poisson_parquet)
    comparison = EvaluationReport.compare(
        models={"lgb_a": (fitted_a, train, test), "lgb_b": (fitted_b, train, test)},
        test_data=test,
    )
    metrics = comparison.metrics()
    assert "model" in metrics.columns
    assert set(metrics["model"].to_list()) >= {"lgb_a", "lgb_b"}


def test_report_comparison_export(tmp_path, poisson_parquet):
    fitted_a, train, test = _fit_poisson(poisson_parquet)
    fitted_b, _, _ = _fit_poisson(poisson_parquet)
    comparison = EvaluationReport.compare(
        models={"lgb_a": (fitted_a, train, test), "lgb_b": (fitted_b, train, test)},
        test_data=test,
    )
    comparison.export(str(tmp_path))
    assert os.path.exists(str(tmp_path / "metrics.csv"))
    assert os.path.exists(str(tmp_path / "double_lift_lgb_a_vs_lgb_b.png"))
