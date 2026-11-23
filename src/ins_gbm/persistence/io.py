from __future__ import annotations

import dataclasses
import json
import os
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ins_gbm.data.model_data import ModelData
    from ins_gbm.pipeline import FittedPipeline

_PIPELINE_FILE = "pipeline.pkl"


def save_pipeline(
    fitted_pipeline: FittedPipeline, output_dir: str, *,
    include_training_data: bool = False,
) -> None:
    """Persist prediction dependencies, or optionally the full training state.

    Artifacts written
    -----------------
    - ``pipeline.pkl``     — fitted pipeline (no training rows by default)
    - ``metadata.json``    — human-readable ReproducibilityMetadata
    - ``tuning_history.parquet`` — trial history (if tuning was run)

    Default saves remove unused raw inputs, one-hot state, preprocessing
    branches, and selection diagnostics. Inputs feeding retained components
    remain required. Custom transforms with unknown dependencies keep their full
    input contract. The live pipeline is not pruned. Compact refits freeze the
    retained feature selection; full saves preserve the original recipe.

    Args:
        fitted_pipeline ('FittedPipeline'): Pipeline fitted on training data.
        output_dir (str): Directory for saved artifacts.
        include_training_data (bool): Preserve original training rows, unused
            feature state, and selection diagnostics. Defaults to False.
    """
    import cloudpickle

    os.makedirs(output_dir, exist_ok=True)

    from ins_gbm.persistence.compact import pipeline_for_save

    compact_pipeline = pipeline_for_save(
        fitted_pipeline, include_training_data=include_training_data,
    )

    # cloudpickle handles locally-defined closures (predict_fn, importance_fn)
    # that standard pickle/joblib cannot serialize.
    with open(os.path.join(output_dir, _PIPELINE_FILE), "wb") as f:
        cloudpickle.dump(compact_pipeline, f)

    meta = compact_pipeline.metadata
    meta_dict = dataclasses.asdict(meta)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta_dict, f, indent=2)

    if fitted_pipeline.tuning_history is not None:
        fitted_pipeline.tuning_history.write_parquet(
            os.path.join(output_dir, "tuning_history.parquet")
        )


def load_pipeline(
    output_dir: str,
    training_data: ModelData | None = None,
) -> FittedPipeline:
    """Load a pipeline, optionally reattaching its original training rows.

    A compact pipeline loaded without ``training_data`` supports prediction,
    evaluation, and feature importance. Supply the exact original training rows
    in their original order to enable ``train_data`` and OOF ensemble fitting.
    Only the saved input features are required when reattaching data. Full saves
    already contain training rows; explicit ``training_data`` replaces them.

    Args:
        output_dir (str): Directory for saved artifacts.
        training_data (Optional['ModelData']): Optional original training data to reattach to the loaded artifact. Optional.
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

    if getattr(fitted_pipeline.metadata, "artifact_version", 1) < 2:
        warnings.warn(
            "This artifact predates prediction-contract version 2; refit it "
            "to obtain corrected prediction scales and offset behavior.",
            UserWarning,
            stacklevel=2,
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


def load_model(
    output_dir: str, training_data: ModelData | None = None
) -> FittedPipeline:
    """Load a fitted model artifact (public convenience alias).

    Args:
        output_dir (str): Directory for saved artifacts.
        training_data (Optional['ModelData']): Optional original training data to reattach to the loaded artifact. Optional.
    """
    return load_pipeline(output_dir, training_data=training_data)
