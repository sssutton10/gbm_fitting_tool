from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.ensemble._utils import _apply_recipe_fold_transforms, _predict_from_pipeline

if TYPE_CHECKING:
    from ins_gbm.pipeline import FittedPipeline


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
        # Clip to positive: Ridge meta-learner can produce values <= 0
        return pl.Series(np.clip(self.meta_learner.predict(base_preds), 1e-9, None))


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
        ref_train = fitted_pipelines[0].raw_train_data
        n = ref_train.n_rows
        fold_splits = list(KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed).split(range(n)))
        oof_matrix = np.zeros((n, len(fitted_pipelines)))

        for p_idx, pipeline in enumerate(fitted_pipelines):
            for train_idx, val_idx in fold_splits:
                fold_train = slice_model_data(pipeline.raw_train_data, train_idx)
                fold_val = slice_model_data(pipeline.raw_train_data, val_idx)
                current_train, current_val = _apply_recipe_fold_transforms(
                    pipeline.recipe, fold_train, fold_val
                )
                fitted_model = pipeline.recipe.model.fit(current_train)
                oof_matrix[val_idx, p_idx] = fitted_model.predict(current_val, "response").to_numpy()

        meta.fit(oof_matrix, ref_train.target.to_numpy().astype(np.float64))

        return FittedStackingEnsemble(
            meta_learner=meta,
            fitted_pipelines=list(fitted_pipelines),
            oof_predictions=oof_matrix,
        )
