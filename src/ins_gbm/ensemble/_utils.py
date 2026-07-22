from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData
from ins_gbm.preprocessing.chain import fit_transform_chain

if TYPE_CHECKING:
    from ins_gbm.pipeline import FittedPipeline, ModelRecipe


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
    return pipeline.fitted_model.predict(transformed, prediction_type="response").to_numpy()


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
