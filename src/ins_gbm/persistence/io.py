from __future__ import annotations

import dataclasses
import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ins_gbm.pipeline import FittedPipeline

_PIPELINE_FILE = "pipeline.pkl"


def save_pipeline(fitted_pipeline: "FittedPipeline", output_dir: str) -> None:
    """Persist a FittedPipeline to *output_dir*.

    Artifacts written
    -----------------
    - ``pipeline.pkl``     — full pipeline object via cloudpickle (handles closures)
    - ``metadata.json``    — human-readable ReproducibilityMetadata
    - ``metrics.csv``      — evaluation metrics from the test set
    - ``tuning_history.parquet`` — trial history (if tuning was run)
    """
    import cloudpickle

    os.makedirs(output_dir, exist_ok=True)

    # cloudpickle handles locally-defined closures (predict_fn, importance_fn)
    # that standard pickle/joblib cannot serialize.
    with open(os.path.join(output_dir, _PIPELINE_FILE), "wb") as f:
        cloudpickle.dump(fitted_pipeline, f)

    meta = fitted_pipeline.metadata
    meta_dict = dataclasses.asdict(meta)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta_dict, f, indent=2)

    try:
        fitted_pipeline.report.metrics().write_csv(
            os.path.join(output_dir, "metrics.csv")
        )
    except Exception as exc:
        import warnings
        warnings.warn(f"Could not write metrics.csv: {exc}", stacklevel=2)

    if fitted_pipeline.tuning_history is not None:
        fitted_pipeline.tuning_history.write_parquet(
            os.path.join(output_dir, "tuning_history.parquet")
        )


def load_pipeline(output_dir: str) -> "FittedPipeline":
    """Load a FittedPipeline previously saved with :func:`save_pipeline`."""
    import cloudpickle

    path = os.path.join(output_dir, _PIPELINE_FILE)
    try:
        with open(path, "rb") as f:
            return cloudpickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No pipeline artifact found at {path!r}. "
            "Was save_pipeline() called with the same output_dir?"
        )
