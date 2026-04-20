"""Smoke tests: verify plots produce valid figures and PNG exports."""
import os
import numpy as np
import polars as pl
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

from ins_gbm.evaluation.plots import (
    plot_lift,
    plot_double_lift,
    plot_ave,
    plot_calibration,
    plot_feature_importance,
    plot_loss_ratio,
)


@pytest.fixture
def sample_data(rng):
    n = 200
    actual = pl.Series(rng.poisson(1.0, n).astype(float))
    predicted = pl.Series(np.maximum(actual.to_numpy() + rng.normal(0, 0.3, n), 0.01))
    exposure = pl.Series(rng.uniform(0.5, 2.0, n))
    return actual, predicted, exposure


def test_plot_lift_returns_figure(sample_data):
    actual, predicted, exposure = sample_data
    fig = plot_lift(actual, predicted, weights=exposure)
    assert fig is not None
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)


def test_plot_lift_exports_png(tmp_path, sample_data):
    actual, predicted, exposure = sample_data
    out = str(tmp_path / "lift.png")
    plot_lift(actual, predicted, weights=exposure, output_path=out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_plot_double_lift_returns_figure(sample_data):
    actual, pred1, exposure = sample_data
    pred2 = pl.Series(pred1.to_numpy() * 1.1)
    fig = plot_double_lift(actual, pred1, pred2, weights=exposure)
    assert fig is not None


def test_plot_ave_returns_figure(sample_data):
    actual, predicted, exposure = sample_data
    fig = plot_ave(actual, predicted, weights=exposure)
    assert fig is not None


def test_plot_calibration_returns_figure(sample_data):
    actual, predicted, exposure = sample_data
    fig = plot_calibration(actual, predicted, weights=exposure)
    assert fig is not None


def test_plot_feature_importance_returns_figure():
    imp = pl.DataFrame({
        "feature": ["x1", "x2", "x3"],
        "importance": [0.5, 0.3, 0.2],
    })
    fig = plot_feature_importance(imp)
    assert fig is not None


def test_plot_loss_ratio_returns_figure(sample_data):
    actual, predicted, exposure = sample_data
    # loss = actual * some amount, premium = exposure * some rate
    loss = pl.Series(actual.to_numpy() * 1000.0)
    premium = pl.Series(exposure.to_numpy() * 500.0)
    fig = plot_loss_ratio(actual, predicted, loss=loss, premium=premium)
    assert fig is not None


def test_all_plots_export_cleanly(tmp_path, sample_data):
    actual, predicted, exposure = sample_data
    plots = {
        "lift": lambda p: plot_lift(actual, predicted, weights=exposure, output_path=p),
        "ave": lambda p: plot_ave(actual, predicted, weights=exposure, output_path=p),
        "calibration": lambda p: plot_calibration(actual, predicted, output_path=p),
    }
    for name, fn in plots.items():
        path = str(tmp_path / f"{name}.png")
        fn(path)
        assert os.path.exists(path), f"{name}.png was not created"
        assert os.path.getsize(path) > 0, f"{name}.png is empty"
