from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData

if TYPE_CHECKING:
    from ins_gbm.pipeline import FittedPipeline


def _apply_pipeline_transforms(pipeline: "FittedPipeline", data: ModelData) -> ModelData:
    """Apply a pipeline's fitted encoder, selector, and preprocessors to *data*."""
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
        blended = stacked @ np.array(self.weights)
        return pl.Series(blended)


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

    # ── Fixed ─────────────────────────────────────────────────────────────────

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

    # ── Validation ────────────────────────────────────────────────────────────

    def _fit_validation(self, pipelines, validation_data: Optional[ModelData]) -> FittedBlendingEnsemble:
        if validation_data is None:
            raise ValueError(
                "validation_data is required when mode='validation'. "
                "Supply a held-out blend set that is not the final test set."
            )
        # Build matrix of base model predictions on validation data
        preds = np.stack(
            [_predict_from_pipeline(p, validation_data) for p in pipelines],
            axis=1,
        )
        actual = validation_data.target.to_numpy().astype(np.float64)
        optimized_weights = _optimize_weights(preds, actual)
        return FittedBlendingEnsemble(
            weights=optimized_weights.tolist(),
            fitted_pipelines=list(pipelines),
        )

    # ── OOF ──────────────────────────────────────────────────────────────────

    def _fit_oof(self, pipelines) -> FittedBlendingEnsemble:
        """Re-fit each pipeline's recipe inside CV folds to get OOF predictions."""
        from sklearn.model_selection import KFold
        from ins_gbm.pipeline import ModelPipeline

        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)

        # Use the first pipeline's training data as the reference
        ref_train = pipelines[0].train_data
        n = ref_train.n_rows
        indices = np.arange(n)

        oof_preds = np.zeros((n, len(pipelines)))

        for p_idx, pipeline in enumerate(pipelines):
            recipe = pipeline.recipe
            train_data = pipeline.train_data

            for train_idx, val_idx in kf.split(indices):
                from ins_gbm.data.model_data import ModelData as _MD

                def _slice(data, idx):
                    return _MD(
                        features=data.features[idx],
                        target=data.target[idx],
                        exposure=data.exposure[idx] if data.exposure is not None else None,
                        weight=data.weight[idx] if data.weight is not None else None,
                        feature_names=data.feature_names,
                        schema=data.schema,
                        objective=data.objective,
                    )

                fold_train = _slice(train_data, train_idx)
                fold_val = _slice(train_data, val_idx)

                # Apply encoder
                current_train = fold_train
                current_val = fold_val
                if recipe.encoder is not None:
                    schema = getattr(current_train, "schema", None)
                    fitted_enc = recipe.encoder.fit(current_train.features, schema)
                    current_train = current_train.with_features(fitted_enc.transform(current_train.features))
                    current_val = current_val.with_features(fitted_enc.transform(current_val.features))

                # Apply selector
                if recipe.selection is not None:
                    fitted_sel = recipe.selection.fit(current_train)
                    sel_feats = fitted_sel.selected_features()
                    current_train = current_train.with_features(current_train.features.select(sel_feats))
                    current_val = current_val.with_features(current_val.features.select(sel_feats))

                # Apply preprocessors
                for prep in recipe.preprocessing:
                    fitted_prep = prep.fit(current_train.features)
                    current_train = current_train.with_features(fitted_prep.transform(current_train.features))
                    current_val = current_val.with_features(fitted_prep.transform(current_val.features))

                # Fit and predict
                fitted_model = recipe.model.fit(current_train)
                fold_preds = fitted_model.predict(current_val, "response").to_numpy()
                oof_preds[val_idx, p_idx] = fold_preds

        actual = ref_train.target.to_numpy().astype(np.float64)
        optimized_weights = _optimize_weights(oof_preds, actual)
        return FittedBlendingEnsemble(
            weights=optimized_weights.tolist(),
            fitted_pipelines=list(pipelines),
        )


def _optimize_weights(preds: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Scipy-optimize blend weights (sum to 1, non-negative) to minimize MSE."""
    from scipy.optimize import minimize

    n_models = preds.shape[1]
    x0 = np.ones(n_models) / n_models

    def objective(w):
        blended = preds @ w
        return np.mean((actual - blended) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_models

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9},
    )
    return result.x
