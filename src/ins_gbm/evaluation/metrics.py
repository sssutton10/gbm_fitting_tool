"""Actuarial evaluation metrics for Poisson frequency and Gamma severity models."""
from typing import Literal, Optional

import numpy as np
import polars as pl

from ins_gbm.data.model_data import Objective


def _to_numpy(s: pl.Series) -> np.ndarray:
    return s.to_numpy().astype(np.float64)


def poisson_deviance(
    actual: pl.Series,
    predicted: pl.Series,
    weights: Optional[pl.Series] = None,
) -> float:
    """Mean Poisson deviance, optionally observation-weighted.

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


def _poisson_rate_metric_inputs(
    actual: pl.Series,
    predicted: pl.Series,
    exposure: Optional[pl.Series],
    weight: Optional[pl.Series],
) -> tuple[pl.Series, pl.Series, Optional[pl.Series]]:
    """Return rate-scale inputs and their effective observation weights.

    Frequency data enters the modeling API as claim counts and expected claim
    counts.  Dividing both by exposure and weighting by exposure gives the
    standard actuarial rate formulation without applying exposure twice.  A
    separate model weight multiplies exposure when supplied.
    """
    if exposure is None:
        return actual, predicted, weight

    effective_weight = exposure if weight is None else exposure * weight
    return actual / exposure, predicted / exposure, effective_weight


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


Direction = Literal["higher", "lower"]
METRIC_DIRECTIONS: dict[str, Direction] = {
    "gini": "higher",
    "poisson_deviance": "lower",
    "gamma_deviance": "lower",
    "rmse": "lower",
    "mae": "lower",
}


def compute_metrics(
    *,
    objective: Objective,
    actual: pl.Series,
    predicted: pl.Series,
    exposure: Optional[pl.Series] = None,
    weight: Optional[pl.Series] = None,
) -> pl.DataFrame:
    """Compute all standard metrics for a given objective.

    Returns a DataFrame with columns ['metric', 'value'].

    Parameters
    ----------
    objective : Objective
        Either "poisson" or "gamma".
    actual : pl.Series
        Actual target values.
    predicted : pl.Series
        Model predictions.
    exposure : Optional[pl.Series]
        Exposure for Poisson models. Count actuals and predictions are divided
        by exposure for deviance and Gini, then weighted by exposure.
    weight : Optional[pl.Series]
        Optional observation/model weights. For Poisson models these multiply
        exposure; for Gamma models they are used directly.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ['metric', 'value'] containing four metrics:
        - objective-specific deviance (poisson_deviance or gamma_deviance)
        - gini (normalized Gini coefficient)
        - rmse (root mean squared error)
        - mae (mean absolute error)
    """
    rows: list[dict] = []
    if objective == "poisson":
        metric_actual, metric_predicted, metric_weight = _poisson_rate_metric_inputs(
            actual, predicted, exposure, weight
        )
        rows.append({
            "metric": "poisson_deviance",
            "value": poisson_deviance(
                metric_actual, metric_predicted, weights=metric_weight
            ),
        })
        gini_actual = metric_actual
        gini_predicted = metric_predicted
        gini_weights = metric_weight
    else:
        rows.append({
            "metric": "gamma_deviance",
            "value": gamma_deviance(actual, predicted, weights=weight),
        })
        gini_actual = actual
        gini_predicted = predicted
        gini_weights = weight
    rows.append({
        "metric": "gini",
        "value": normalized_gini(
            gini_actual, gini_predicted, weights=gini_weights
        )
    })
    rows.append({"metric": "rmse", "value": rmse(actual, predicted)})
    rows.append({"metric": "mae", "value": mae(actual, predicted)})
    return pl.DataFrame(rows)
