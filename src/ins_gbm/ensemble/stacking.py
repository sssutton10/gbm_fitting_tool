from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData

if TYPE_CHECKING:
    from ins_gbm.pipeline import FittedPipeline


def _apply_pipeline_transforms(pipeline: "FittedPipeline", data: ModelData) -> ModelData:
    current = data
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


@dataclass
class FittedStackingEnsemble:
    """A stacking ensemble with a fitted meta-learner."""
    meta_learner: Any
    fitted_pipelines: list["FittedPipeline"]
    oof_predictions: np.ndarray  # shape (n_train, n_base_models)

    def predict(self, data: ModelData) -> pl.Series:
        """Stack base model predictions and apply the meta-learner."""
        base_preds = np.stack(
            [_predict_from_pipeline(p, data) for p in self.fitted_pipelines],
            axis=1,
        )
        meta_preds = self.meta_learner.predict(base_preds)
        # Clip to keep predictions positive (meta-learner may predict < 0)
        meta_preds = np.clip(meta_preds, 1e-9, None)
        return pl.Series(meta_preds)


@dataclass
class StackingEnsemble:
    """Stacking ensemble using OOF predictions as meta-features.

    For each base pipeline:
    1. Re-fit its full recipe inside ``cv_folds`` cross-validation folds on
       the pipeline's training data.
    2. Collect out-of-fold predictions (never from test data).

    The meta-learner is trained on the stacked OOF predictions.
    Final test predictions use the pre-fitted base models (from the original
    ``FittedPipeline`` objects) plus the trained meta-learner — the test set is
    never touched during meta-learner training.

    Parameters
    ----------
    cv_folds : int
        Number of CV folds for generating OOF predictions.
    seed : int
        Random seed for fold splitting.
    meta_learner : sklearn estimator or None
        The meta-model trained on OOF predictions.  Defaults to Ridge regression.
        Must implement ``.fit(X, y)`` and ``.predict(X)``.
    """
    cv_folds: int = 5
    seed: int = 42
    meta_learner: Optional[Any] = None

    def fit(self, fitted_pipelines: list["FittedPipeline"]) -> FittedStackingEnsemble:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold

        meta = self.meta_learner if self.meta_learner is not None else Ridge()

        # Use training data from the first pipeline as the reference target
        ref_train = fitted_pipelines[0].train_data
        n = ref_train.n_rows
        indices = np.arange(n)
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        fold_splits = list(kf.split(indices))

        oof_matrix = np.zeros((n, len(fitted_pipelines)))

        for p_idx, pipeline in enumerate(fitted_pipelines):
            recipe = pipeline.recipe
            train_data = pipeline.train_data

            for train_idx, val_idx in fold_splits:
                fold_train = _slice_model_data(train_data, train_idx)
                fold_val = _slice_model_data(train_data, val_idx)

                # Apply encoder per fold
                current_train = fold_train
                current_val = fold_val
                if recipe.encoder is not None:
                    schema = getattr(current_train, "schema", None)
                    fitted_enc = recipe.encoder.fit(current_train.features, schema)
                    current_train = current_train.with_features(fitted_enc.transform(current_train.features))
                    current_val = current_val.with_features(fitted_enc.transform(current_val.features))

                # Apply selector per fold
                if recipe.selection is not None:
                    fitted_sel = recipe.selection.fit(current_train)
                    sel_feats = fitted_sel.selected_features()
                    current_train = current_train.with_features(current_train.features.select(sel_feats))
                    current_val = current_val.with_features(current_val.features.select(sel_feats))

                # Apply preprocessors per fold
                for prep in recipe.preprocessing:
                    fitted_prep = prep.fit(current_train.features)
                    current_train = current_train.with_features(fitted_prep.transform(current_train.features))
                    current_val = current_val.with_features(fitted_prep.transform(current_val.features))

                fitted_model = recipe.model.fit(current_train)
                fold_preds = fitted_model.predict(current_val, "response").to_numpy()
                oof_matrix[val_idx, p_idx] = fold_preds

        # Train meta-learner on OOF predictions
        y_train = ref_train.target.to_numpy().astype(np.float64)
        meta.fit(oof_matrix, y_train)

        return FittedStackingEnsemble(
            meta_learner=meta,
            fitted_pipelines=list(fitted_pipelines),
            oof_predictions=oof_matrix,
        )


def _slice_model_data(data: ModelData, indices: np.ndarray) -> ModelData:
    return ModelData(
        features=data.features[indices],
        target=data.target[indices],
        exposure=data.exposure[indices] if data.exposure is not None else None,
        weight=data.weight[indices] if data.weight is not None else None,
        feature_names=data.feature_names,
        schema=data.schema,
        objective=data.objective,
    )
