from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ins_gbm.data.dtypes import FIT_DTYPE
from ins_gbm.data.folds import CVConfig, resolve_folds
from ins_gbm.data.model_data import ModelData
from ins_gbm.preprocessing.chain import TransformFitResult, fit_transform_chain

if TYPE_CHECKING:
    from ins_gbm.pipeline import FittedPipeline, ModelRecipe


def _resolve_ensemble_folds(
    data: ModelData,
    cv: CVConfig | None,
    cv_folds: int,
    seed: int,
    use_data_folds: bool | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Use the same fold policy as tuning for ensemble OOF predictions."""
    if cv is not None and use_data_folds is not None:
        raise ValueError("Pass either cv or use_data_folds, not both")
    config = cv or CVConfig(
        n_splits=cv_folds,
        seed=seed,
        folds=(
            "auto"
            if use_data_folds is None
            else ("predefined" if use_data_folds else "random")
        ),
    )
    return resolve_folds(data, config)[1]


def _validate_ensemble_pipelines(
    pipelines: list[FittedPipeline], *, require_training: bool = True
) -> None:
    """Check base pipelines have compatible objectives and training data.

    Args:
        pipelines (list['FittedPipeline']): Base fitted pipelines to combine.
        require_training (bool): Whether original training rows must be attached. Defaults to
            True.
    """
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
        for field in ("target", "exposure", "weight", "offset", "cv_fold"):
            left, right = getattr(reference, field), getattr(current, field)
            if (left is None) != (right is None):
                raise ValueError(f"ensemble training data must have matching {field}")
            if left is not None and not np.array_equal(
                left.to_numpy(), right.to_numpy(), equal_nan=True
            ):
                raise ValueError(f"ensemble training rows are not aligned for {field}")


def _apply_pipeline_transforms(pipeline: FittedPipeline, data: ModelData) -> ModelData:
    """Apply a fitted pipeline's encoder, selector, and preprocessors to *data*.

    Args:
        pipeline ('FittedPipeline'): Fitted pipeline used for prediction or transformation.
        data (ModelData): Model data to fit, transform, predict, or evaluate.
    """
    return pipeline._prepare_data(data)


def _predict_from_pipeline(pipeline: FittedPipeline, data: ModelData) -> np.ndarray:
    """Generate predictions through a fitted base pipeline.

    Args:
        pipeline ('FittedPipeline'): Fitted pipeline used for prediction or transformation.
        data (ModelData): Model data to fit, transform, predict, or evaluate.
    """
    transformed = _apply_pipeline_transforms(pipeline, data)
    return (
        pipeline.fitted_model.predict(transformed, prediction_type="response")
        .to_numpy()
        .astype(FIT_DTYPE, copy=False)
    )


def _apply_recipe_fold_transforms(
    recipe: ModelRecipe,
    fold_train: ModelData,
    fold_val: ModelData,
) -> tuple[ModelData, ModelData]:
    """Fit recipe's encoder/selector/preprocessors on fold_train; transform both folds.

    Used in stacking OOF generation and blending OOF mode to prevent leakage
    across fold boundaries.  Returns (transformed_train, transformed_val).

    Args:
        recipe ('ModelRecipe'): Unfitted pipeline recipe.
        fold_train (ModelData): Training rows for the current fold.
        fold_val (ModelData): Validation rows for the current fold.
    """
    result = fit_transform_chain(
        fold_train,
        encoder=recipe.encoder,
        selector=recipe.selection,
        preprocessing=recipe.preprocessing,
    )
    return result.data, result.chain.transform(fold_val)


def _apply_pipeline_recipe_fold_transforms(
    pipeline: FittedPipeline,
    fold_train: ModelData,
    fold_val: ModelData,
) -> tuple[ModelData, ModelData]:
    """Refit recipe transforms while preserving fixed feature selections.

    Args:
        pipeline ('FittedPipeline'): Fitted pipeline used for prediction or transformation.
        fold_train (ModelData): Training rows for the current fold.
        fold_val (ModelData): Validation rows for the current fold.
    """
    result = _fit_pipeline_transforms(pipeline, fold_train)
    return result.data, result.chain.transform(fold_val)


def _fit_pipeline_transforms(
    pipeline: FittedPipeline, data: ModelData
) -> TransformFitResult:
    """Refit transforms, retaining manual encoded and model feature subsets."""
    recipe = deepcopy(pipeline.recipe)
    model_selected = getattr(pipeline, "model_selected_features", None)
    if recipe.selection is None and pipeline.selected_features is not None:
        encoded = fit_transform_chain(data, encoder=recipe.encoder)
        missing = set(pipeline.selected_features) - set(encoded.data.feature_names)
        if missing:
            raise ValueError(
                "Manual encoded features are absent in this fold/sample: "
                f"{sorted(missing)}. Use stable encoder levels or a raw feature subset."
            )
        train = encoded.data.with_features(
            encoded.data.features.select(pipeline.selected_features)
        )
        processed = fit_transform_chain(
            train,
            preprocessing=recipe.preprocessing,
            model_selected_features=model_selected,
        )
        processed.raw_data = encoded.raw_data
        processed.chain.input_feature_names = encoded.chain.input_feature_names
        processed.chain.encoder = encoded.chain.encoder
        processed.chain.selected_features = list(pipeline.selected_features)
        return processed
    return fit_transform_chain(
        data,
        encoder=recipe.encoder,
        selector=recipe.selection,
        preprocessing=recipe.preprocessing,
        model_selected_features=model_selected,
    )


def _refit_pipeline(
    template: FittedPipeline, data: ModelData, params: dict
) -> FittedPipeline:
    """Build an independent fitted member without repeating hyperparameter tuning."""
    from ins_gbm.persistence.metadata import build_metadata
    from ins_gbm.pipeline import FittedPipeline

    recipe = replace(deepcopy(template.recipe), tuning=None, params=dict(params))
    result = _fit_pipeline_transforms(replace(template, recipe=recipe), data)
    chain = result.chain
    fitted = recipe.model.fit(result.data, params=params)
    metadata = build_metadata(
        fitted, chain.selected_features,
        input_feature_names=chain.input_feature_names,
        model_selected_features=chain.model_selected_features,
        selection_scope=recipe.selection_scope,
    )
    return FittedPipeline(
        fitted_model=fitted,
        recipe=recipe,
        input_feature_names=chain.input_feature_names,
        raw_train_data=result.raw_data,
        selected_features=chain.selected_features,
        model_selected_features=chain.model_selected_features,
        selection_results=(
            [result.fitted_selector] if result.fitted_selector is not None else None
        ),
        tuning_history=None,
        encoder=chain.encoder,
        preprocessors=chain.preprocessors,
        metadata=metadata,
        input_schema=result.raw_data.schema,
    )
