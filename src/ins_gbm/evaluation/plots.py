"""Actuarial evaluation plots for insurance GBM models."""
from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl


def _to_numpy(s: pl.Series) -> np.ndarray:
    return s.to_numpy().astype(np.float64)


def _decile_summary(
    actual: np.ndarray,
    predicted: np.ndarray,
    weights: Optional[np.ndarray],
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort by predicted, bin into deciles, return (bin_label, mean_actual, mean_predicted)."""
    order = np.argsort(predicted)
    actual_s = actual[order]
    predicted_s = predicted[order]
    w = weights[order] if weights is not None else np.ones(len(actual))

    bins = np.array_split(np.arange(len(actual_s)), n_bins)
    mean_actual = np.array([np.average(actual_s[b], weights=w[b]) for b in bins])
    mean_pred = np.array([np.average(predicted_s[b], weights=w[b]) for b in bins])
    labels = np.arange(1, n_bins + 1)
    return labels, mean_actual, mean_pred


def plot_lift(
    actual: pl.Series,
    predicted: pl.Series,
    weights: Optional[pl.Series] = None,
    n_bins: int = 10,
    output_path: Optional[str] = None,
    title: str = "Ordered Lift Chart",
) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt

    y = _to_numpy(actual)
    p = _to_numpy(predicted)
    w = _to_numpy(weights) if weights is not None else None

    labels, mean_actual, mean_pred = _decile_summary(y, p, w, n_bins)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(labels, mean_actual, "o-", label="Actual", color="steelblue")
    ax.plot(labels, mean_pred, "s--", label="Predicted", color="darkorange")
    ax.set_xlabel("Decile (sorted by predicted)")
    ax.set_ylabel("Mean value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
    return fig


def plot_double_lift(
    actual: pl.Series,
    predicted_a: pl.Series,
    predicted_b: pl.Series,
    weights: Optional[pl.Series] = None,
    n_bins: int = 10,
    labels: tuple[str, str] = ("Model A", "Model B"),
    output_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt
    from ins_gbm.evaluation.metrics import double_lift_table

    summary = double_lift_table(
        actual,
        predicted_a,
        predicted_b,
        weights=weights,
        n_bins=n_bins,
    )
    bin_labels = summary["bucket"].to_numpy()
    avg_actual = summary["actual"].to_numpy()
    avg_a = summary["model1"].to_numpy()
    avg_b = summary["model2"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bin_labels, avg_actual, "o-", label="Actual", color="black")
    ax.plot(bin_labels, avg_a, "s--", label=labels[0], color="steelblue")
    ax.plot(bin_labels, avg_b, "^--", label=labels[1], color="darkorange")
    ax.set_xlabel("Bucket (sorted by B/A ratio)")
    ax.set_ylabel("Mean value")
    ax.set_title("Double Lift Chart")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=100)
        import matplotlib.pyplot as _plt
        _plt.close(fig)
    return fig


def plot_ave(
    actual: pl.Series,
    predicted: pl.Series,
    weights: Optional[pl.Series] = None,
    n_bins: int = 10,
    output_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Actual vs Expected plot by predicted decile."""
    import matplotlib.pyplot as plt

    y = _to_numpy(actual)
    p = _to_numpy(predicted)
    w = _to_numpy(weights) if weights is not None else None

    labels, mean_actual, mean_pred = _decile_summary(y, p, w, n_bins)
    ave_ratio = mean_actual / np.maximum(mean_pred, 1e-15)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, ave_ratio, color="steelblue", alpha=0.7)
    ax.axhline(1.0, color="red", linestyle="--", label="A/E = 1.0")
    ax.set_xlabel("Decile (sorted by predicted)")
    ax.set_ylabel("Actual / Expected")
    ax.set_title("Actual vs Expected (A/E) by Decile")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
    return fig


def plot_calibration(
    actual: pl.Series,
    predicted: pl.Series,
    weights: Optional[pl.Series] = None,
    n_bins: int = 10,
    output_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Calibration curve: mean predicted vs mean actual per bin."""
    import matplotlib.pyplot as plt

    y = _to_numpy(actual)
    p = _to_numpy(predicted)
    w = _to_numpy(weights) if weights is not None else None

    _, mean_actual, mean_pred = _decile_summary(y, p, w, n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(mean_pred, mean_actual, color="steelblue", zorder=3)
    lim = [min(mean_pred.min(), mean_actual.min()) * 0.9,
           max(mean_pred.max(), mean_actual.max()) * 1.1]
    ax.plot(lim, lim, "r--", label="Perfect calibration")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Mean Predicted")
    ax.set_ylabel("Mean Actual")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
    return fig


def plot_feature_importance(
    importance_df: pl.DataFrame,
    top_n: int = 20,
    output_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Horizontal bar chart of feature importance (top_n features)."""
    import matplotlib.pyplot as plt

    df = importance_df.sort("importance", descending=True).head(top_n)
    features = df["feature"].to_list()
    scores = df["importance"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.4)))
    ax.barh(features[::-1], scores[::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance (top {top_n})")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
    return fig


def plot_loss_ratio(
    actual: pl.Series,
    predicted: pl.Series,
    loss: pl.Series,
    premium: pl.Series,
    n_bins: int = 10,
    output_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Loss ratio by predicted frequency/severity decile."""
    import matplotlib.pyplot as plt

    p = _to_numpy(predicted)
    loss_np = _to_numpy(loss)
    prem_np = _to_numpy(premium)

    order = np.argsort(p)
    loss_s = loss_np[order]
    prem_s = prem_np[order]

    bins = np.array_split(np.arange(len(p)), n_bins)
    bin_labels = np.arange(1, n_bins + 1)
    loss_ratio = np.array([
        loss_s[b].sum() / max(prem_s[b].sum(), 1e-15) for b in bins
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bin_labels, loss_ratio, color="steelblue", alpha=0.7)
    overall_lr = loss_np.sum() / max(prem_np.sum(), 1e-15)
    ax.axhline(overall_lr, color="red", linestyle="--", label=f"Overall LR = {overall_lr:.2%}")
    ax.set_xlabel("Decile (sorted by predicted)")
    ax.set_ylabel("Loss Ratio")
    ax.set_title("Loss Ratio by Predicted Decile")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
    return fig
