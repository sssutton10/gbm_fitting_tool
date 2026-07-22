import numpy as np
import polars as pl
import pytest
from ins_gbm.evaluation.metrics import (
    poisson_deviance,
    gamma_deviance,
    normalized_gini,
    rmse,
    mae,
    METRIC_DIRECTIONS,
    compute_metrics,
)


# ── Poisson deviance ───────────────────────────────────────────────────────────

def test_poisson_deviance_perfect_predictions():
    actual = pl.Series([1.0, 2.0, 3.0])
    pred = pl.Series([1.0, 2.0, 3.0])
    assert poisson_deviance(actual, pred) == pytest.approx(0.0, abs=1e-10)


def test_poisson_deviance_manual():
    # d_i = 2*(y*log(y/mu) - (y - mu))
    # y=2, mu=1: 2*(2*log(2) - 1) = 2*(1.3863 - 1) = 0.7726
    actual = pl.Series([2.0])
    pred = pl.Series([1.0])
    expected = 2.0 * (2.0 * np.log(2.0) - (2.0 - 1.0))
    assert poisson_deviance(actual, pred) == pytest.approx(expected, rel=1e-6)


def test_poisson_deviance_zero_actual_is_valid():
    # 0 * log(0/mu) = 0 by convention
    actual = pl.Series([0.0, 1.0])
    pred = pl.Series([1.0, 1.0])
    result = poisson_deviance(actual, pred)
    assert np.isfinite(result)


def test_poisson_deviance_weighted():
    actual = pl.Series([1.0, 2.0])
    pred = pl.Series([1.0, 2.0])
    weights = pl.Series([2.0, 3.0])
    # perfect predictions → deviance = 0 regardless of weights
    assert poisson_deviance(actual, pred, weights=weights) == pytest.approx(0.0, abs=1e-10)


def test_poisson_deviance_rejects_nonpositive_predictions():
    with pytest.raises(ValueError, match="positive"):
        poisson_deviance(pl.Series([1.0]), pl.Series([0.0]))


# ── Gamma deviance ─────────────────────────────────────────────────────────────

def test_gamma_deviance_perfect_predictions():
    actual = pl.Series([100.0, 200.0, 300.0])
    pred = pl.Series([100.0, 200.0, 300.0])
    assert gamma_deviance(actual, pred) == pytest.approx(0.0, abs=1e-10)


def test_gamma_deviance_manual():
    # d_i = 2*(-log(y/mu) + (y-mu)/mu)
    # y=2, mu=1: 2*(-log(2) + (2-1)/1) = 2*(-0.6931 + 1) = 0.6137
    actual = pl.Series([2.0])
    pred = pl.Series([1.0])
    expected = 2.0 * (-np.log(2.0 / 1.0) + (2.0 - 1.0) / 1.0)
    assert gamma_deviance(actual, pred) == pytest.approx(expected, rel=1e-6)


def test_gamma_deviance_rejects_nonpositive_actual():
    with pytest.raises(ValueError, match="positive"):
        gamma_deviance(pl.Series([0.0, 1.0]), pl.Series([1.0, 1.0]))


def test_gamma_deviance_rejects_nonpositive_predictions():
    with pytest.raises(ValueError, match="positive"):
        gamma_deviance(pl.Series([1.0]), pl.Series([0.0]))


def test_gamma_deviance_weighted():
    actual = pl.Series([100.0, 200.0])
    pred = pl.Series([100.0, 200.0])
    weights = pl.Series([1.0, 5.0])
    assert gamma_deviance(actual, pred, weights=weights) == pytest.approx(0.0, abs=1e-10)


# ── Normalized Gini ────────────────────────────────────────────────────────────

def test_gini_perfect_model():
    actual = pl.Series([1.0, 2.0, 3.0, 4.0])
    pred = pl.Series([1.0, 2.0, 3.0, 4.0])
    assert normalized_gini(actual, pred) == pytest.approx(1.0, abs=1e-6)


def test_gini_random_model_near_zero():
    rng = np.random.default_rng(0)
    actual = pl.Series(rng.poisson(1.0, 500).astype(float))
    pred = pl.Series(rng.uniform(0, 1, 500))  # random — should be near 0
    result = normalized_gini(actual, pred)
    assert abs(result) < 0.15


def test_gini_weighted():
    actual = pl.Series([1.0, 2.0, 3.0, 4.0])
    pred = pl.Series([1.0, 2.0, 3.0, 4.0])
    weights = pl.Series([1.0, 1.0, 1.0, 1.0])
    assert normalized_gini(actual, pred, weights=weights) == pytest.approx(1.0, abs=1e-6)


# ── RMSE and MAE ───────────────────────────────────────────────────────────────

def test_rmse_zero():
    s = pl.Series([1.0, 2.0, 3.0])
    assert rmse(s, s) == pytest.approx(0.0)


def test_rmse_manual():
    actual = pl.Series([1.0, 3.0])
    pred = pl.Series([2.0, 2.0])
    # errors: [-1, 1], MSE = 1, RMSE = 1
    assert rmse(actual, pred) == pytest.approx(1.0)


def test_mae_zero():
    s = pl.Series([1.0, 2.0])
    assert mae(s, s) == pytest.approx(0.0)


def test_mae_manual():
    actual = pl.Series([1.0, 3.0])
    pred = pl.Series([2.0, 2.0])
    assert mae(actual, pred) == pytest.approx(1.0)


# ── compute_metrics and METRIC_DIRECTIONS ──────────────────────────────────

def test_compute_metrics_poisson_returns_four_metrics():
    actual = pl.Series([1.0, 2.0, 3.0])
    predicted = pl.Series([1.1, 1.9, 3.2])
    exposure = pl.Series([1.0, 1.0, 1.0])
    result = compute_metrics(
        objective="poisson",
        actual=actual,
        predicted=predicted,
        exposure=exposure,
    )
    assert set(result["metric"].to_list()) == {"poisson_deviance", "gini", "rmse", "mae"}
    assert result["value"].dtype == pl.Float64


def test_compute_metrics_gamma_returns_four_metrics():
    actual = pl.Series([100.0, 200.0, 300.0])
    predicted = pl.Series([110.0, 190.0, 320.0])
    weight = pl.Series([1.0, 1.0, 1.0])
    result = compute_metrics(
        objective="gamma",
        actual=actual,
        predicted=predicted,
        weight=weight,
    )
    assert set(result["metric"].to_list()) == {"gamma_deviance", "gini", "rmse", "mae"}


def test_compute_metrics_matches_individual_functions():
    actual = pl.Series([1.0, 0.0, 2.0])
    predicted = pl.Series([0.9, 0.1, 2.1])
    exposure = pl.Series([1.5, 0.5, 1.0])
    result = compute_metrics(
        objective="poisson",
        actual=actual,
        predicted=predicted,
        exposure=exposure,
    )
    expected_deviance = poisson_deviance(
        actual / exposure,
        predicted / exposure,
        weights=exposure,
    )
    row = result.filter(pl.col("metric") == "poisson_deviance")["value"][0]
    assert abs(row - expected_deviance) < 1e-10


def test_compute_metrics_poisson_combines_exposure_and_model_weight():
    actual = pl.Series([1.0, 4.0, 3.0])
    predicted = pl.Series([2.0, 2.0, 4.5])
    exposure = pl.Series([1.0, 2.0, 3.0])
    weight = pl.Series([3.0, 5.0, 2.0])

    result = compute_metrics(
        objective="poisson",
        actual=actual,
        predicted=predicted,
        exposure=exposure,
        weight=weight,
    )

    actual_rate = actual / exposure
    predicted_rate = predicted / exposure
    effective_weight = exposure * weight
    expected_deviance = poisson_deviance(
        actual_rate,
        predicted_rate,
        weights=effective_weight,
    )
    expected_gini = normalized_gini(
        actual_rate,
        predicted_rate,
        weights=effective_weight,
    )

    deviance = result.filter(pl.col("metric") == "poisson_deviance")["value"][0]
    gini = result.filter(pl.col("metric") == "gini")["value"][0]
    assert deviance == pytest.approx(expected_deviance)
    assert gini == pytest.approx(expected_gini)


def test_compute_metrics_no_weights():
    actual = pl.Series([1.0, 2.0, 3.0])
    predicted = pl.Series([1.1, 1.9, 3.2])
    result = compute_metrics(
        objective="poisson",
        actual=actual,
        predicted=predicted,
    )
    assert set(result["metric"].to_list()) == {"poisson_deviance", "gini", "rmse", "mae"}


def test_metric_directions_has_all_keys():
    assert METRIC_DIRECTIONS["gini"] == "higher"
    assert METRIC_DIRECTIONS["poisson_deviance"] == "lower"
    assert METRIC_DIRECTIONS["gamma_deviance"] == "lower"
    assert METRIC_DIRECTIONS["rmse"] == "lower"
    assert METRIC_DIRECTIONS["mae"] == "lower"
