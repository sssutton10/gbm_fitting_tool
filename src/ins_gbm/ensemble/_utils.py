from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ins_gbm.data.dtypes import FIT_DTYPE
from ins_gbm.data.model_data import ModelData
from ins_gbm.preprocessing.chain import fit_transform_chain

if TYPE_CHECKING:
    from ins_gbm.pipeline import FittedPipeline, ModelRecipe


def _validate_ensemble_pipelines(
    pipelines: list["FittedPipeline"], *, require_training: bool = True
) -> None:
    if not pipelines:
        raise ValueError("at least one fitted pipeline is required")
    objective = pipelines[0].fitted_model.objective
    for pipeline in pipelines[1:]:
        if pipeline.fitted_model.objective != objective:
            raise ValueError("ensemble pipelines must use the same objective")
    if not require_training:
        return
    data = [pipeline._require_raw_train_data() for pipeline in pipelines]
    reference = data[0]
    for current in data[1:]:
        if current.n_rows != reference.n_rows:
            raise ValueError("ensemble training data must have matching row counts")
        for field in ("target", "exposure", "weight", "cv_fold"):
            left, right = getattr(reference, field), getattr(current, field)
            if (left is None) != (right is None):
                raise ValueError(f"ensemble training data must have matching {field}")
            if left is not None and not np.array_equal(
                left.to_numpy(), right.to_numpy(), equal_nan=True
            ):
                raise ValueError(f"ensemble training rows are not aligned for {field}")


def _apply_pipeline_transforms(pipeline: "FittedPipeline", data: ModelData) -> ModelData:
    """Apply a fitted pipeline's encoder, selector, and preprocessors to *data*."""
    current = data.select_features(pipeline.input_feature_names)
    if pipeline.encoder is not None:
        current = current.with_features(pipeline.encoder.transform(current.features))
    if pipeline.selected_features is not None:
        current = current.with_features(
            current.features.select(pipeline.selected_features)
        )
    for prep in pipeline.preprocessors:
        current = current.with_features(prep.transform(current.features))
    return current


def _predict_from_pipeline(pipeline: "FittedPipeline", data: ModelData) -> np.ndarray:
    transformed = _apply_pipeline_transforms(pipeline, data)
    return pipeline.fitted_model.predict(
        transformed, prediction_type="response"
    ).to_numpy().astype(FIT_DTYPE, copy=False)


def _apply_recipe_fold_transforms(
    recipe: "ModelRecipe",
    fold_train: ModelData,
    fold_val: ModelData,
) -> tuple[ModelData, ModelData]:
    """Fit recipe's encoder/selector/preprocessors on fold_train; transform both folds.

    Used in stacking OOF generation and blending OOF mode to prevent leakage
    across fold boundaries.  Returns (transformed_train, transformed_val).
    """
    result = fit_transform_chain(
        fold_train,
        encoder=recipe.encoder,
        selector=recipe.selection,
        preprocessing=recipe.preprocessing,
    )
    return result.data, result.chain.transform(fold_val)


def _apply_pipeline_recipe_fold_transforms(
    pipeline: "FittedPipeline",
    fold_train: ModelData,
    fold_val: ModelData,
) -> tuple[ModelData, ModelData]:
    """Refit recipe transforms while preserving a manual encoded selection."""
    if pipeline.recipe.selection is None and pipeline.selected_features is not None:
        encoded = fit_transform_chain(fold_train, encoder=pipeline.recipe.encoder)
        train, val = encoded.data, encoded.chain.transform(fold_val)
        missing = [name for name in pipeline.selected_features if name not in train.features.columns]
        if missing:
            raise ValueError(
                "Manual encoded features are absent in this fold: "
                f"{missing}. Use stable encoder levels or a raw feature subset."
            )
        train = train.with_features(train.features.select(pipeline.selected_features))
        val = val.with_features(val.features.select(pipeline.selected_features))
        processed = fit_transform_chain(
            train, preprocessing=pipeline.recipe.preprocessing
        )
        return processed.data, processed.chain.transform(val)
    return _apply_recipe_fold_transforms(pipeline.recipe, fold_train, fold_val)
