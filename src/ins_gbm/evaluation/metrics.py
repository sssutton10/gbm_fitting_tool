"""Actuarial evaluation metrics for Poisson frequency and Gamma severity models."""
from typing import Optional

import numpy as np
import polars as pl


def _to_numpy(s: pl.Series) -> np.ndarray:
    return s.to_numpy().astype(np.float64)


def poisson_deviance(
    actual: pl.Series,
    predicted: pl.Series,
    weights: Optional[pl.Series] = None,
) -> float:
    """Mean Poisson deviance, optionally exposure-weighted.

    d_i = 2 * (y_i * log(y_i / mu_i) - (y_i - mu_i))
    Convention: 0 * log(0) = 0.
    """
    y = _to_numpy(actual)
    mu = _to_numpy(predicted)

    if np.any(mu <= 0):
        raise ValueError("predicted values must be positive for Poisson deviance")

    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(y > 0, y * np.log(y / mu), 0.0)

    d = 2.0 * (log_term - (y - mu))

    if weights is not None:
        w = _to_numpy(weights)
        return float(np.sum(w * d) / np.sum(w))
    return float(np.mean(d))


def gamma_deviance(
    actual: pl.Series,
    predicted: pl.Series,
    weights: Optional[pl.Series] = None,
) -> float:
    """Mean Gamma deviance, optionally weight-adjusted.

    d_i = 2 * (-log(y_i / mu_i) + (y_i - mu_i) / mu_i)
    """
    y = _to_numpy(actual)
    mu = _to_numpy(predicted)

    if np.any(y <= 0):
        raise ValueError("actual values must be positive for Gamma deviance")
    if np.any(mu <= 0):
        raise ValueError("predicted values must be positive for Gamma deviance")

    d = 2.0 * (-np.log(y / mu) + (y - mu) / mu)

    if weights is not None:
        w = _to_numpy(weights)
        return float(np.sum(w * d) / np.sum(w))
    return float(np.mean(d))


def normalized_gini(
    actual: pl.Series,
    predicted: pl.Series,
    weights: Optional[pl.Series] = None,
) -> float:
    """Normalized Gini coefficient.

    Sort by predicted descending, compute Lorenz curve, compare to perfect model.
    Returns value in [-1, 1]; 1.0 = perfect ranking, 0.0 = random.
    """
    y = _to_numpy(actual)
    p = _to_numpy(predicted)
    w = _to_numpy(weights) if weights is not None else np.ones(len(y))

    def _gini(order: np.ndarray) -> float:
        sorted_y = y[order]
        sorted_w = w[order]
        cum_actual = np.cumsum(sorted_y * sorted_w) / max(np.sum(sorted_y * sorted_w), 1e-15)
        cum_weight = np.cumsum(sorted_w) / max(np.sum(sorted_w), 1e-15)
        # Prepend (0, 0) for trapezoidal integration
        cum_actual = np.concatenate([[0.0], cum_actual])
        cum_weight = np.concatenate([[0.0], cum_weight])
        return float(1.0 - 2.0 * np.trapezoid(cum_actual, cum_weight))

    gini_model = _gini(np.argsort(-p))
    gini_perfect = _gini(np.argsort(-y))

    if abs(gini_perfect) < 1e-15:
        return 0.0
    return gini_model / gini_perfect


def rmse(
    actual: pl.Series,
    predicted: pl.Series,
    weights: Optional[pl.Series] = None,
) -> float:
    """Root mean squared error, optionally weighted."""
    y = _to_numpy(actual)
    p = _to_numpy(predicted)
    residuals = (y - p) ** 2
    if weights is not None:
        w = _to_numpy(weights)
        return float(np.sqrt(np.sum(w * residuals) / np.sum(w)))
    return float(np.sqrt(np.mean(residuals)))


def mae(
    actual: pl.Series,
    predicted: pl.Series,
    weights: Optional[pl.Series] = None,
) -> float:
    """Mean absolute error, optionally weighted."""
    y = _to_numpy(actual)
    p = _to_numpy(predicted)
    residuals = np.abs(y - p)
    if weights is not None:
        w = _to_numpy(weights)
        return float(np.sum(w * residuals) / np.sum(w))
    return float(np.mean(residuals))
