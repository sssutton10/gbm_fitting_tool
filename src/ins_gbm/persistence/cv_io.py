"""Portable, data-light artifacts for cross-validation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from ins_gbm.evaluation.cv_report import CVResult

_FORMAT_VERSION = 1
_FOLD_COLUMN = "_cv_fold"


def _json_scalar(value):
    """Convert a scalar to a JSON-compatible value.

    Args:
        value (object): Value to inspect or replace.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"CV metadata value {value!r} is not JSON serializable")


def save_cv_result(result: CVResult, output_dir: str) -> None:
    """Save fold metrics, summary, predictions, and provenance to a directory.

    Args:
        result ('CVResult'): Cross-validation result to save.
        output_dir (str): Directory for saved artifacts.
    """
    if (
        result.predictions is None
        or result.row_folds is None
        or result.objective is None
        or result.data_signature is None
        or result.cv_config is None
    ):
        raise ValueError("CVResult lacks predictions or provenance required for saving")
    n_rows = result.predictions.height
    if len(result.row_folds) != n_rows or "gbm" not in result.predictions.columns:
        raise ValueError("CVResult predictions and row folds must align")
    if _FOLD_COLUMN in result.predictions.columns:
        raise ValueError(f"Prediction column {_FOLD_COLUMN!r} is reserved")

    fold_params = [
        {"fold": fold, "params": params}
        for fold, params in (result.fold_params or {}).items()
    ]
    metadata = {
        "format_version": _FORMAT_VERSION,
        "objective": result.objective,
        "feature_names": result.feature_names,
        "fold_col": result.fold_col,
        "cv_config": result.cv_config,
        "row_count": n_rows,
        "data_signature": result.data_signature,
        "fold_params": fold_params,
    }
    metadata_json = json.dumps(
        metadata, indent=2, default=_json_scalar, allow_nan=False
    )

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    result.fold_metrics.write_parquet(path / "fold_metrics.parquet")
    result.summary.write_parquet(path / "summary.parquet")
    result.predictions.with_columns(
        result.row_folds.rename(_FOLD_COLUMN)
    ).write_parquet(path / "predictions.parquet")
    (path / "metadata.json").write_text(metadata_json, encoding="utf-8")


def load_cv_result(output_dir: str) -> CVResult:
    """Load a saved CV report without loading targets, weights, or features.

    Args:
        output_dir (str): Directory for saved artifacts.
    """
    from ins_gbm.evaluation.cv_report import CVResult

    path = Path(output_dir)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != _FORMAT_VERSION:
        raise ValueError("Unsupported CV report artifact version")
    predictions = pl.read_parquet(path / "predictions.parquet")
    if _FOLD_COLUMN not in predictions.columns or "gbm" not in predictions.columns:
        raise ValueError("CV report prediction artifact lacks folds or gbm predictions")
    if predictions.height != metadata["row_count"]:
        raise ValueError("CV report prediction row count does not match metadata")
    return CVResult(
        fold_metrics=pl.read_parquet(path / "fold_metrics.parquet"),
        summary=pl.read_parquet(path / "summary.parquet"),
        fold_col=metadata["fold_col"],
        predictions=predictions.drop(_FOLD_COLUMN),
        row_folds=predictions[_FOLD_COLUMN].rename("fold"),
        objective=metadata["objective"],
        feature_names=metadata["feature_names"],
        cv_config=metadata["cv_config"],
        data_signature=metadata["data_signature"],
        fold_params={item["fold"]: item["params"] for item in metadata["fold_params"]},
    )
