from __future__ import annotations

import dataclasses
import json
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ins_gbm.data.model_data import ModelData
    from ins_gbm.pipeline import FittedPipeline

_PIPELINE_FILE = "pipeline.pkl"


def save_pipeline(fitted_pipeline: "FittedPipeline", output_dir: str) -> None:
    """Persist a compact FittedPipeline to *output_dir*.

    Artifacts written
    -----------------
    - ``pipeline.pkl``     — fitted pipeline without raw training rows
    - ``metadata.json``    — human-readable ReproducibilityMetadata
    - ``tuning_history.parquet`` — trial history (if tuning was run)
    """
    import cloudpickle

    os.makedirs(output_dir, exist_ok=True)

    # Preserve the schema needed by predict_raw(), but do not serialize policy-
    # level training rows. dataclasses.replace() is shallow, so this neither
    # copies the dataset nor mutates the caller's in-memory fitted pipeline.
    compact_pipeline = dataclasses.replace(
        fitted_pipeline,
        raw_train_data=None,
        input_schema=fitted_pipeline._input_schema(),
    )

    # cloudpickle handles locally-defined closures (predict_fn, importance_fn)
    # that standard pickle/joblib cannot serialize.
    with open(os.path.join(output_dir, _PIPELINE_FILE), "wb") as f:
        cloudpickle.dump(compact_pipeline, f)

    meta = fitted_pipeline.metadata
    meta_dict = dataclasses.asdict(meta)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta_dict, f, indent=2)

    if fitted_pipeline.tuning_history is not None:
        fitted_pipeline.tuning_history.write_parquet(
            os.path.join(output_dir, "tuning_history.parquet")
        )


def load_pipeline(
    output_dir: str,
    training_data: Optional["ModelData"] = None,
) -> "FittedPipeline":
    """Load a pipeline, optionally reattaching its original training rows.

    A compact pipeline loaded without ``training_data`` supports prediction,
    evaluation, and feature importance. Supply the exact original training rows
    in their original order to enable ``train_data`` and OOF ensemble fitting.
    """
    import cloudpickle

    path = os.path.join(output_dir, _PIPELINE_FILE)
    try:
        with open(path, "rb") as f:
            fitted_pipeline = cloudpickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No pipeline artifact found at {path!r}. "
            "Was save_pipeline() called with the same output_dir?"
        )

    if training_data is None:
        return fitted_pipeline

    expected_objective = fitted_pipeline.fitted_model.objective
    if (
        training_data.objective is not None
        and training_data.objective != expected_objective
    ):
        raise ValueError(
            "training_data objective does not match the fitted pipeline: "
            f"{training_data.objective!r} != {expected_objective!r}"
        )

    attached_data = training_data.select_features(
        fitted_pipeline.input_feature_names
    ).validate()
    return dataclasses.replace(
        fitted_pipeline,
        raw_train_data=attached_data,
        input_schema=fitted_pipeline._input_schema() or attached_data.schema,
    )
