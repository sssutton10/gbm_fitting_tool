"""Shared dtype policy for model-fitting buffers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

FIT_DTYPE = np.float32
FIT_POLARS_DTYPE = pl.Float32


def frame_to_fit_array(
    frame: pl.DataFrame,
    columns: Sequence[str] | None = None,
) -> np.ndarray:
    """Return a dense, C-compatible float32 model matrix.

    Args:
        frame (pl.DataFrame): Input data frame.
        columns (Sequence[str] | None): Ordered column names to extract. Optional.
    """
    selected = frame.select(columns) if columns is not None else frame
    return selected.cast(FIT_POLARS_DTYPE).to_numpy()


def series_to_fit_array(series: pl.Series) -> np.ndarray:
    """Return a one-dimensional float32 fitting array.

    Args:
        series (pl.Series): Input data series.
    """
    return series.cast(FIT_POLARS_DTYPE).to_numpy()


def replace_value_with_nan(array: np.ndarray, value: float) -> np.ndarray:
    """Replace a sentinel lazily, copying only when the sentinel is present.

    Args:
        array (np.ndarray): Input NumPy array.
        value (float): Value to inspect or replace.
    """
    sentinel_rows = array == value
    if not sentinel_rows.any():
        return array
    result = array.copy()
    result[sentinel_rows] = np.nan
    return result


def cast_float64_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Downcast only Float64 columns, preserving categorical and integer data.

    Args:
        frame (pl.DataFrame): Input data frame.
    """
    float64_columns = [
        name for name, dtype in frame.schema.items() if dtype == pl.Float64
    ]
    if not float64_columns:
        return frame
    return frame.with_columns(pl.col(float64_columns).cast(FIT_POLARS_DTYPE))


def cast_float64_series(series: pl.Series | None) -> pl.Series | None:
    """Downcast a Float64 series while preserving all other dtypes.

    Args:
        series (pl.Series | None): Input data series.
    """
    if series is not None and series.dtype == pl.Float64:
        return series.cast(FIT_POLARS_DTYPE)
    return series
