from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from ins_gbm.data.dtypes import FIT_DTYPE, series_to_fit_array
from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.ensemble._utils import (
    _apply_pipeline_recipe_fold_transforms,
    _predict_from_pipeline,
    _validate_ensemble_pipelines,
)

if TYPE_CHECKING:
    from ins_gbm.pipeline import FittedPipeline


@dataclass
class FittedBlendingEnsemble:
    """A blending ensemble with fitted weights.

    Args:
        weights (list[float]): Nonnegative blending or observation weights.
        fitted_pipelines (list['FittedPipeline']): Base pipelines with fitted models.
    """

    weights: list[float]
    fitted_pipelines: list[FittedPipeline]

    def predict(self, data: ModelData) -> pl.Series:
        """Return the weighted-average prediction on *data*.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
        """
        stacked = np.stack(
            [_predict_from_pipeline(p, data) for p in self.fitted_pipelines],
            axis=1,
        )
        return pl.Series(stacked @ np.asarray(self.weights, dtype=FIT_DTYPE))


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

    refit : {"fixed", "retune"}
        Use fitted model parameters in each OOF fold, or retune each fold.
    """

    mode: Literal["fixed", "validation", "oof"] = "fixed"
    weights: list[float] | None = None
    cv_folds: int = 5
    seed: int = 42
    refit: str = "fixed"

    def fit(
        self,
        fitted_pipelines: list[FittedPipeline],
        validation_data: ModelData | None = None,
    ) -> FittedBlendingEnsemble:
        """Compute blend weights and return a :class:`FittedBlendingEnsemble`.

        Parameters
        ----------
        fitted_pipelines : list[FittedPipeline]
            Pre-fitted base pipelines to blend.
        validation_data : ModelData or None
            Required when ``mode="validation"``.  Must not be the test set.
        """
        _validate_ensemble_pipelines(
            fitted_pipelines, require_training=self.mode == "oof"
        )
        if self.refit not in {"fixed", "retune"}:
            raise ValueError("refit must be 'fixed' or 'retune'")
        if self.mode == "fixed":
            return self._fit_fixed(fitted_pipelines)
        elif self.mode == "validation":
            return self._fit_validation(fitted_pipelines, validation_data)
        elif self.mode == "oof":
            return self._fit_oof(fitted_pipelines)
        else:
            raise ValueError(
                f"Unknown mode: {self.mode!r}. Choose from 'fixed', 'validation', 'oof'."
            )

    def _fit_fixed(self, pipelines) -> FittedBlendingEnsemble:
        """Use the configured fixed blending weights.

        Args:
            pipelines (object): Base fitted pipelines to combine.
        """
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

    def _fit_validation(
        self, pipelines, validation_data: ModelData | None
    ) -> FittedBlendingEnsemble:
        """Optimize blending weights on validation predictions.

        Args:
            pipelines (object): Base fitted pipelines to combine.
            validation_data (Optional[ModelData]): Holdout data used to learn blend weights.
        """
        if validation_data is None:
            raise ValueError(
                "validation_data is required when mode='validation'. "
                "Supply a held-out blend set that is not the final test set."
            )
        preds = np.stack(
            [_predict_from_pipeline(p, validation_data) for p in pipelines],
            axis=1,
        )
        actual = series_to_fit_array(validation_data.target)
        return FittedBlendingEnsemble(
            weights=_optimize_weights(preds, actual).tolist(),
            fitted_pipelines=list(pipelines),
        )

    def _fit_oof(self, pipelines) -> FittedBlendingEnsemble:
        """Re-fit each pipeline's recipe inside CV folds to get OOF predictions.

        Args:
            pipelines (object): Base fitted pipelines to combine.
        """
        from sklearn.model_selection import KFold

        training_data = [pipeline._require_raw_train_data() for pipeline in pipelines]
        ref_train = training_data[0]
        n = ref_train.n_rows
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        oof_preds = np.zeros((n, len(pipelines)), dtype=FIT_DTYPE)

        for p_idx, (pipeline, pipeline_data) in enumerate(
            zip(pipelines, training_data)
        ):
            for train_idx, val_idx in kf.split(range(n)):
                fold_train = slice_model_data(pipeline_data, train_idx)
                fold_val = slice_model_data(pipeline_data, val_idx)
                if self.refit == "retune" and pipeline.recipe.tuning is not None:
                    from ins_gbm.pipeline import ModelPipeline

                    model_selected = getattr(pipeline, "model_selected_features", None)
                    run_kwargs = (
                        {"feature_names": model_selected, "feature_stage": "model"}
                        if model_selected is not None
                        else {
                            "feature_names": pipeline.selected_features,
                            "feature_stage": "encoded",
                        }
                        if pipeline.recipe.selection is None
                        and pipeline.selected_features is not None
                        else {}
                    )
                    fold_pipeline = ModelPipeline(fold_train, pipeline.recipe).run(
                        **run_kwargs
                    )
                    predictions = fold_pipeline.predict(fold_val, "response")
                else:
                    current_train, current_val = _apply_pipeline_recipe_fold_transforms(
                        pipeline, fold_train, fold_val
                    )
                    fitted_model = pipeline.recipe.model.fit(
                        current_train, params=pipeline.fitted_model.params
                    )
                    predictions = fitted_model.predict(current_val, "response")
                oof_preds[val_idx, p_idx] = predictions.to_numpy()

        actual = series_to_fit_array(ref_train.target)
        return FittedBlendingEnsemble(
            weights=_optimize_weights(oof_preds, actual).tolist(),
            fitted_pipelines=list(pipelines),
        )


def _optimize_weights(preds: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Scipy-optimize blend weights (sum to 1, non-negative) to minimise MSE.

    Args:
        preds (np.ndarray): Matrix of base model predictions.
        actual (np.ndarray): Observed outcomes aligned with predictions.
    """
    from scipy.optimize import minimize

    n_models = preds.shape[1]
    x0 = np.ones(n_models) / n_models

    def objective(w):
        """Objective.

        Args:
            w (object): Candidate blending weights.
        """
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
