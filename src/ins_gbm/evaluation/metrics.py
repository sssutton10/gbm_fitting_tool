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


def _double_lift_metric_inputs(
    objective: Objective,
    actual: pl.Series,
    predicted_a: pl.Series,
    predicted_b: pl.Series,
    exposure: Optional[pl.Series],
    weight: Optional[pl.Series],
) -> tuple[pl.Series, pl.Series, pl.Series, Optional[pl.Series]]:
    """Return consistently scaled comparison inputs for double-lift metrics."""
    if objective == "poisson" and exposure is not None:
        effective_weight = exposure if weight is None else exposure * weight
        return (
            actual / exposure,
            predicted_a / exposure,
            predicted_b / exposure,
            effective_weight,
        )
    return actual, predicted_a, predicted_b, weight


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


def _double_lift_bucket_ids(
    scores: np.ndarray,
    weights: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Assign stable, approximately equal-weight double-lift buckets."""
    if (
        not isinstance(n_bins, int)
        or isinstance(n_bins, bool)
        or n_bins < 2
        or n_bins > len(scores)
    ):
        raise ValueError(
            f"n_bins must be an integer from 2 through {len(scores)}"
        )

    order = np.argsort(scores, kind="mergesort")
    cumulative_midpoints = np.cumsum(weights[order]) - weights[order] / 2.0
    raw_ids = np.floor(
        cumulative_midpoints / weights.sum() * n_bins
    ).astype(int)
    raw_ids = np.clip(raw_ids, 0, n_bins - 1)

    bucket_ids = np.empty(len(scores), dtype=int)
    bucket_ids[order] = raw_ids + 1

    # Concentrated weights can leave bucket numbers unused. Compact them so
    # chart axes remain consecutive without changing bucket membership.
    used = sorted(set(bucket_ids.tolist()))
    remap = {old: new for new, old in enumerate(used, 1)}
    return np.array([remap[item] for item in bucket_ids], dtype=int)


def double_lift_table(
    actual: pl.Series,
    predicted_a: pl.Series,
    predicted_b: pl.Series,
    weights: Optional[pl.Series] = None,
    n_bins: int = 10,
) -> pl.DataFrame:
    """Summarize two models in buckets ordered by the B/A prediction ratio.

    Buckets are approximately equal-weight. ``model1`` is ``predicted_a`` and
    ``model2`` is ``predicted_b``; this ordering also defines the sign of
    :func:`double_lift_score`.
    """
    y = _to_numpy(actual)
    model1 = _to_numpy(predicted_a)
    model2 = _to_numpy(predicted_b)
    if not (len(y) == len(model1) == len(model2)) or len(y) == 0:
        raise ValueError(
            "actual, predicted_a, and predicted_b must be non-empty and "
            "have matching lengths"
        )
    if not (
        np.all(np.isfinite(y))
        and np.all(np.isfinite(model1))
        and np.all(np.isfinite(model2))
    ):
        raise ValueError("double-lift inputs must contain only finite values")
    if np.any(model1 == 0):
        raise ValueError(
            "predicted_a must not contain zero when computing a double-lift ratio"
        )

    if weights is None:
        w = np.ones(len(y), dtype=np.float64)
    else:
        w = _to_numpy(weights)
        if len(w) != len(y):
            raise ValueError("weights must have the same length as actual")
        if (
            not np.all(np.isfinite(w))
            or np.any(w < 0)
            or w.sum() <= 0
        ):
            raise ValueError(
                "weights must be finite, non-negative, and have a positive total"
            )

    ratio = model2 / model1
    if not np.all(np.isfinite(ratio)):
        raise ValueError(
            "predicted_a and predicted_b must produce finite double-lift ratios"
        )

    bucket_ids = _double_lift_bucket_ids(ratio, w, n_bins)
    rows: list[dict] = []
    for bucket in sorted(set(bucket_ids.tolist())):
        mask = bucket_ids == bucket
        bucket_weight = float(w[mask].sum())
        if bucket_weight <= 0:
            continue
        rows.append({
            "bucket": bucket,
            "actual": float(np.dot(y[mask], w[mask]) / bucket_weight),
            "model1": float(np.dot(model1[mask], w[mask]) / bucket_weight),
            "model2": float(np.dot(model2[mask], w[mask]) / bucket_weight),
            "ratio_mean": float(np.dot(ratio[mask], w[mask]) / bucket_weight),
            "weight": bucket_weight,
        })
    return pl.DataFrame(rows)


def double_lift_score(
    dl_table: pl.DataFrame,
    deviation: Literal["absolute", "relative"] = "absolute",
) -> float:
    """Score model 2 against model 1 from a double-lift table.

    The absolute score is ``sum(|model1 - actual| - |model2 - actual|)``.
    Positive values favor model 2, negative values favor model 1, and zero is
    a tie. Relative deviation applies the same comparison on ratio errors.
    """
    required = {"actual", "model1", "model2"}
    missing = sorted(required - set(dl_table.columns))
    if missing:
        raise ValueError(
            f"dl_table is missing required columns: {missing}"
        )
    if deviation not in {"absolute", "relative"}:
        raise ValueError("deviation must be 'absolute' or 'relative'")

    actual = _to_numpy(dl_table["actual"])
    model1 = _to_numpy(dl_table["model1"])
    model2 = _to_numpy(dl_table["model2"])
    if (
        len(actual) == 0
        or not np.all(np.isfinite(actual))
        or not np.all(np.isfinite(model1))
        or not np.all(np.isfinite(model2))
    ):
        raise ValueError(
            "dl_table actual and model columns must be finite and non-empty"
        )

    if deviation == "absolute":
        return float(
            (np.abs(model1 - actual) - np.abs(model2 - actual)).sum()
        )

    if np.any(np.abs(model1) < 1e-12) or np.any(np.abs(model2) < 1e-12):
        raise ValueError(
            "relative double-lift score requires non-zero model values"
        )
    return float(
        (
            np.abs(actual / model1 - 1.0)
            - np.abs(actual / model2 - 1.0)
        ).sum()
    )


Direction = Literal["higher", "lower"]
METRIC_DIRECTIONS: dict[str, Direction] = {
    "gini": "higher",
    "poisson_deviance": "lower",
    "gamma_deviance": "lower",
    "rmse": "lower",
    "mae": "lower",
    # Negative values favor model 1, which is the report's focal model.
    "double_lift_score": "lower",
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
