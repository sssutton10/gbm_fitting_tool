"""Tests for comparison_predictions in EvaluationReport."""
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
    fitted = LightGBMModel(objective="poisson").fit(train, params={"n_estimators": 10, "verbose": -1})
    return fitted, train, test


def _fake_preds(n: int, seed: int = 0) -> pl.Series:
    import numpy as np
    rng = np.random.default_rng(seed)
    return pl.Series(rng.uniform(0.01, 1.0, n))


# ── No comparison_predictions: existing API unchanged ────────────────────────

def test_no_comparison_predictions_metrics_format(poisson_parquet):
    fitted, train, test = _fit_poisson(poisson_parquet)
    report = EvaluationReport(fitted_model=fitted, test_data=test, train_data=train)
    metrics = report.metrics()
    assert isinstance(metrics, pl.DataFrame)
    assert "metric" in metrics.columns
    assert "value" in metrics.columns
    assert "model" not in metrics.columns  # backward compat — no model column


# ── With comparison_predictions: long-format with model column ────────────────

def test_comparison_predictions_metrics_has_model_column(poisson_parquet):
    fitted, train, test = _fit_poisson(poisson_parquet)
    prod_preds = _fake_preds(test.n_rows)
    report = EvaluationReport(
        fitted_model=fitted, test_data=test, train_data=train,
        comparison_predictions={"production": prod_preds},
    )
    metrics = report.metrics()
    assert "model" in metrics.columns
    assert "metric" in metrics.columns
    assert "value" in metrics.columns


def test_comparison_predictions_metrics_has_gbm_and_comparison_rows(poisson_parquet):
    fitted, train, test = _fit_poisson(poisson_parquet)
    prod_preds = _fake_preds(test.n_rows)
    report = EvaluationReport(
        fitted_model=fitted, test_data=test, train_data=train,
        comparison_predictions={"production": prod_preds},
    )
    models_in_output = set(report.metrics()["model"].to_list())
    assert "GBM" in models_in_output
    assert "production" in models_in_output


def test_comparison_predictions_multiple_series(poisson_parquet):
    fitted, train, test = _fit_poisson(poisson_parquet)
    report = EvaluationReport(
        fitted_model=fitted, test_data=test, train_data=train,
        comparison_predictions={
            "prod_v1": _fake_preds(test.n_rows, seed=1),
            "prod_v2": _fake_preds(test.n_rows, seed=2),
        },
    )
    models_in_output = set(report.metrics()["model"].to_list())
    assert {"GBM", "prod_v1", "prod_v2"}.issubset(models_in_output)


# ── plot_double_lift ──────────────────────────────────────────────────────────

def test_plot_double_lift_returns_figure(poisson_parquet):
    import matplotlib.figure
    fitted, train, test = _fit_poisson(poisson_parquet)
    prod_preds = _fake_preds(test.n_rows)
    report = EvaluationReport(
        fitted_model=fitted, test_data=test, train_data=train,
        comparison_predictions={"production": prod_preds},
    )
    fig = report.plot_double_lift("production")
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close("all")


def test_plot_double_lift_unknown_name_raises(poisson_parquet):
    fitted, train, test = _fit_poisson(poisson_parquet)
    report = EvaluationReport(
        fitted_model=fitted, test_data=test, train_data=train,
        comparison_predictions={"production": _fake_preds(test.n_rows)},
    )
    with pytest.raises(KeyError):
        report.plot_double_lift("nonexistent")


def test_plot_double_lift_no_comparison_predictions_raises(poisson_parquet):
    fitted, train, test = _fit_poisson(poisson_parquet)
    report = EvaluationReport(fitted_model=fitted, test_data=test, train_data=train)
    with pytest.raises((KeyError, AttributeError, TypeError)):
        report.plot_double_lift("production")


# ── export writes double_lift PNGs ───────────────────────────────────────────

def test_export_writes_double_lift_png(tmp_path, poisson_parquet):
    fitted, train, test = _fit_poisson(poisson_parquet)
    prod_preds = _fake_preds(test.n_rows)
    report = EvaluationReport(
        fitted_model=fitted, test_data=test, train_data=train,
        comparison_predictions={"production": prod_preds},
    )
    report.export(str(tmp_path))
    assert os.path.exists(str(tmp_path / "double_lift_GBM_vs_production.png"))


def test_export_still_writes_standard_files(tmp_path, poisson_parquet):
    fitted, train, test = _fit_poisson(poisson_parquet)
    report = EvaluationReport(
        fitted_model=fitted, test_data=test, train_data=train,
        comparison_predictions={"prod": _fake_preds(test.n_rows)},
    )
    report.export(str(tmp_path))
    for fname in ("metrics.csv", "lift.png", "ave.png", "calibration.png", "feature_importance.png"):
        assert os.path.exists(str(tmp_path / fname)), f"missing: {fname}"
