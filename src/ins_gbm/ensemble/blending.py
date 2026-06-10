from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.ensemble._utils import _apply_recipe_fold_transforms, _predict_from_pipeline

if TYPE_CHECKING:
    from ins_gbm.pipeline import FittedPipeline


@dataclass
class FittedBlendingEnsemble:
    """A blending ensemble with fitted weights."""
    weights: list[float]
    fitted_pipelines: list["FittedPipeline"]

    def predict(self, data: ModelData) -> pl.Series:
        """Return the weighted-average prediction on *data*."""
        stacked = np.stack(
            [_predict_from_pipeline(p, data) for p in self.fitted_pipelines],
            axis=1,
        )
        return pl.Series(stacked @ np.array(self.weights))


@dataclass
class BlendingEnsemble:
    """Fit blend weights for a list of pre-fitted pipelines.

    Parameters
    ----------
    mode : {"fixed", "validation", "oof"}
        How blend weights are determined:
        - ``"fixed"`` — user supplies ``weights`` directly.
        - ``"validation"`` — weights are scipy-optimized on a user-supplied
          ``validation_data`` that must not be the test set.
        - ``"oof"`` — weights are optimized on out-of-fold predictions from
          the base pipelines' training data (requires re-fitting).
    weights : list[float] or None
        Required when ``mode="fixed"``. Must sum to 1.
    cv_folds : int
        Number of CV folds used for ``"oof"`` mode.
    seed : int
        Random seed for ``"oof"`` mode fold splitting.
    """
    mode: Literal["fixed", "validation", "oof"] = "fixed"
    weights: Optional[list[float]] = None
    cv_folds: int = 5
    seed: int = 42

    def fit(
        self,
        fitted_pipelines: list["FittedPipeline"],
        validation_data: Optional[ModelData] = None,
    ) -> FittedBlendingEnsemble:
        """Compute blend weights and return a :class:`FittedBlendingEnsemble`.

        Parameters
        ----------
        fitted_pipelines : list[FittedPipeline]
            Pre-fitted base pipelines to blend.
        validation_data : ModelData or None
            Required when ``mode="validation"``.  Must not be the test set.
        """
        if self.mode == "fixed":
            return self._fit_fixed(fitted_pipelines)
        elif self.mode == "validation":
            return self._fit_validation(fitted_pipelines, validation_data)
        elif self.mode == "oof":
            return self._fit_oof(fitted_pipelines)
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}. Choose from 'fixed', 'validation', 'oof'.")

    def _fit_fixed(self, pipelines) -> FittedBlendingEnsemble:
        if self.weights is None:
            raise ValueError(
                "weights must be provided when mode='fixed'. "
                "Pass a list of floats that sum to 1."
            )
        total = sum(self.weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Fixed weights must sum to 1.0 (got {total:.6f}). "
                "Normalize your weights before passing them."
            )
        if len(self.weights) != len(pipelines):
            raise ValueError(
                f"len(weights)={len(self.weights)} != len(fitted_pipelines)={len(pipelines)}"
            )
        return FittedBlendingEnsemble(
            weights=list(self.weights),
            fitted_pipelines=list(pipelines),
        )

    def _fit_validation(self, pipelines, validation_data: Optional[ModelData]) -> FittedBlendingEnsemble:
        if validation_data is None:
            raise ValueError(
                "validation_data is required when mode='validation'. "
                "Supply a held-out blend set that is not the final test set."
            )
        preds = np.stack(
            [_predict_from_pipeline(p, validation_data) for p in pipelines],
            axis=1,
        )
        actual = validation_data.target.to_numpy().astype(np.float64)
        return FittedBlendingEnsemble(
            weights=_optimize_weights(preds, actual).tolist(),
            fitted_pipelines=list(pipelines),
        )

    def _fit_oof(self, pipelines) -> FittedBlendingEnsemble:
        """Re-fit each pipeline's recipe inside CV folds to get OOF predictions."""
        from sklearn.model_selection import KFold

        ref_train = pipelines[0].train_data
        n = ref_train.n_rows
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        oof_preds = np.zeros((n, len(pipelines)))

        for p_idx, pipeline in enumerate(pipelines):
            for train_idx, val_idx in kf.split(range(n)):
                fold_train = slice_model_data(pipeline.train_data, train_idx)
                fold_val = slice_model_data(pipeline.train_data, val_idx)
                current_train, current_val = _apply_recipe_fold_transforms(
                    pipeline.recipe, fold_train, fold_val
                )
                fitted_model = pipeline.recipe.model.fit(current_train)
                oof_preds[val_idx, p_idx] = fitted_model.predict(current_val, "response").to_numpy()

        actual = ref_train.target.to_numpy().astype(np.float64)
        return FittedBlendingEnsemble(
            weights=_optimize_weights(oof_preds, actual).tolist(),
            fitted_pipelines=list(pipelines),
        )


def _optimize_weights(preds: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Scipy-optimize blend weights (sum to 1, non-negative) to minimise MSE."""
    from scipy.optimize import minimize

    n_models = preds.shape[1]
    x0 = np.ones(n_models) / n_models

    def objective(w):
        return np.mean((actual - preds @ w) ** 2)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"ftol": 1e-9},
    )
    return result.x
