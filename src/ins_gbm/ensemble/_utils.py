from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData

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
    from ins_gbm.preprocessing.steps import validate_preprocessing_steps

    validate_preprocessing_steps(recipe.preprocessing)
    current_train = fold_train
    current_val = fold_val

    if recipe.encoder is not None:
        schema = getattr(current_train, "schema", None)
        fitted_enc = recipe.encoder.fit(current_train.features, schema)
        current_train = current_train.with_features(fitted_enc.transform(current_train.features))
        current_val = current_val.with_features(fitted_enc.transform(current_val.features))

    if recipe.selection is not None:
        fitted_sel = recipe.selection.fit(current_train)
        sel_feats = fitted_sel.selected_features()
        current_train = current_train.with_features(current_train.features.select(sel_feats))
        current_val = current_val.with_features(current_val.features.select(sel_feats))

    for prep in recipe.preprocessing:
        # Pass target so supervised reducers (e.g. PLS) can fit; unsupervised
        # reducers accept and ignore it (fit(features, target=None)).
        fitted_prep = prep.fit(current_train.features, current_train.target)
        current_train = current_train.with_features(fitted_prep.transform(current_train.features))
        current_val = current_val.with_features(fitted_prep.transform(current_val.features))

    return current_train, current_val
