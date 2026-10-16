from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ins_gbm.data.dtypes import FIT_DTYPE
from ins_gbm.data.folds import CVConfig
from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.ensemble._utils import (
    _apply_pipeline_recipe_fold_transforms,
    _predict_from_pipeline,
    _resolve_ensemble_folds,
    _validate_ensemble_pipelines,
)

if TYPE_CHECKING:
    from ins_gbm.pipeline import FittedPipeline


def _stacking_inputs(
    preds: np.ndarray, data: ModelData, objective: str
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize response means by exposure/offset before taking log features.

    Poisson response is a rate when exposure is absent (the default).
    With count targets and explicit exposure, dividing by exposure gives a rate.
    An explicit offset is also divided out here and restored once in the final
    response. Gamma only removes the offset.
    """
    data.validate_for_prediction()
    baseline = np.ones(data.n_rows, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if data.offset is not None:
            baseline *= np.exp(data.offset.to_numpy().astype(np.float64))
        if objective == "poisson" and data.exposure is not None:
            baseline *= data.exposure.to_numpy().astype(np.float64)
    if not np.all(np.isfinite(baseline)) or np.any(baseline <= 0):
        raise ValueError("stacking exposure/offset baseline must be finite and positive")
    preds = np.asarray(preds, dtype=np.float64)
    if not np.all(np.isfinite(preds)) or np.any(preds <= 0):
        raise ValueError("distribution-aware stacking requires finite positive predictions")
    # Taking differences of logs avoids overflow in preds / baseline.
    features = np.log(preds) - np.log(baseline)[:, None]
    return features, baseline


@dataclass
class FittedStackingEnsemble:
    """A stacking ensemble with a fitted meta-learner.

    Args:
        meta_learner (Any): Optional regression estimator for stacking.
        fitted_pipelines (list['FittedPipeline']): Base pipelines with fitted models.
        oof_predictions (np.ndarray | None): OOF predictions; omitted on load.
        distribution_aware (bool): Normalize exposure/offset for the meta-model.
    """

    meta_learner: Any
    fitted_pipelines: list[FittedPipeline]
    oof_predictions: np.ndarray | None  # omitted from persisted artifacts
    distribution_aware: bool = False

    def predict(self, data: ModelData) -> pl.Series:
        """Stack base model predictions and apply the meta-learner.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
        """
        base_preds = np.stack(
            [_predict_from_pipeline(p, data) for p in self.fitted_pipelines],
            axis=1,
        )
        return self.predict_from_predictions(base_preds, data)

    def predict_from_predictions(self, base_preds: np.ndarray, data: ModelData) -> pl.Series:
        """Apply the fitted meta-model to aligned response-scale base means."""
        if self.distribution_aware:
            objective = self.fitted_pipelines[0].fitted_model.objective
            features, baseline = _stacking_inputs(base_preds, data, objective)
            with np.errstate(over="ignore", invalid="ignore"):
                response = self.meta_learner.predict(features) * baseline
            if not np.all(np.isfinite(response)) or np.any(response <= 0):
                raise ValueError("stacking response predictions must be finite and positive")
            return pl.Series(response)
        # Ridge/custom learners retain response-scale inputs and positive clipping.
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
    meta_learner : {None, "auto", "ridge"} or sklearn estimator
        None/"auto" selects PoissonRegressor or GammaRegressor. Built-in
        distribution learners use log base means with exposure/offset removed,
        and restore the scoring baseline exactly once. By default Poisson
        targets are rates with exposure omitted, fitted directly with observation
        weights. Optional count targets with exposure are normalized internally
        to rates with baseline times observation weights. Gamma targets are
        severity, normalized only for explicit offsets, with observation weights.
        "ridge" and custom estimators use response-scale inputs and targets.
        Custom estimators must accept sample_weight when data has weights.
    meta_alpha : float
        Nonnegative regularization for built-in learners. Defaults to 1.0.

    refit : {"fixed", "retune"}
        Use fitted model parameters in each OOF fold, or retune each fold.
    use_data_folds : bool or None
        None uses ``ModelData.cv_fold`` when present, True requires it, and
        False generates random folds.
    cv : CVConfig or None
        Fold configuration overriding ``cv_folds`` and ``seed``. Cannot be
        combined with ``use_data_folds``.
    """

    cv_folds: int = 5
    seed: int = 42
    meta_learner: Any | None = None
    refit: str = "fixed"
    use_data_folds: bool | None = None
    cv: CVConfig | None = None
    meta_alpha: float = 1.0

    def fit(self, fitted_pipelines: list[FittedPipeline]) -> FittedStackingEnsemble:
        """Fit the ensemble from pre-fitted base pipelines.

        Args:
            fitted_pipelines (list['FittedPipeline']): Base pipelines with fitted models.
        """
        _validate_ensemble_pipelines(fitted_pipelines)
        if self.refit not in {"fixed", "retune"}:
            raise ValueError("refit must be 'fixed' or 'retune'")
        training_data = [
            pipeline._require_raw_train_data() for pipeline in fitted_pipelines
        ]
        ref_train = training_data[0]
        n = ref_train.n_rows
        fold_splits = _resolve_ensemble_folds(
            ref_train, self.cv, self.cv_folds, self.seed, self.use_data_folds
        )
        oof_matrix = np.zeros((n, len(fitted_pipelines)), dtype=FIT_DTYPE)

        for p_idx, (pipeline, pipeline_data) in enumerate(
            zip(fitted_pipelines, training_data)
        ):
            for train_idx, val_idx in fold_splits:
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
                oof_matrix[val_idx, p_idx] = predictions.to_numpy()

        return self.fit_from_predictions(fitted_pipelines, oof_matrix, ref_train)

    def fit_from_predictions(
        self, fitted_pipelines: list[FittedPipeline], oof_matrix: np.ndarray,
        ref_train: ModelData,
    ) -> FittedStackingEnsemble:
        """Fit a fresh meta-model using already generated inner OOF predictions."""
        from sklearn.linear_model import GammaRegressor, PoissonRegressor, Ridge

        distribution_aware = self.meta_learner is None or (
            isinstance(self.meta_learner, str) and self.meta_learner == "auto"
        )
        objective = fitted_pipelines[0].fitted_model.objective
        if distribution_aware or isinstance(self.meta_learner, str):
            if not np.isfinite(self.meta_alpha) or self.meta_alpha < 0:
                raise ValueError("meta_alpha must be finite and nonnegative")
        if distribution_aware:
            estimator = PoissonRegressor if objective == "poisson" else GammaRegressor
            meta = estimator(alpha=self.meta_alpha, max_iter=1000)
        elif isinstance(self.meta_learner, str):
            if self.meta_learner != "ridge":
                raise ValueError("meta_learner must be 'auto', 'ridge', or an estimator")
            meta = Ridge(alpha=self.meta_alpha)
        else:
            meta = deepcopy(self.meta_learner)
        if not callable(getattr(meta, "fit", None)) or not callable(
            getattr(meta, "predict", None)
        ):
            raise ValueError("meta_learner must implement fit and predict")
        ref_train.validate(require_multiple_folds=False)
        target = ref_train.target.to_numpy().astype(np.float64)
        weights = (
            ref_train.weight.to_numpy().astype(np.float64)
            if ref_train.weight is not None else None
        )
        features = oof_matrix.astype(np.float64)
        if distribution_aware:
            features, baseline = _stacking_inputs(oof_matrix, ref_train, objective)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                target = target / baseline
                if objective == "poisson" and (
                    ref_train.exposure is not None or ref_train.offset is not None
                ):
                    weights = baseline if weights is None else baseline * weights
            if not np.all(np.isfinite(target)) or np.any(target < 0) or (
                objective == "gamma" and np.any(target <= 0)
            ):
                raise ValueError("normalized stacking targets are outside the distribution domain")
            if weights is not None and (
                not np.all(np.isfinite(weights)) or weights.sum() <= 0
            ):
                raise ValueError("stacking weights must be finite with positive total")
        fit_kwargs = {}
        if weights is not None:
            parameters = signature(meta.fit).parameters
            if "sample_weight" not in parameters and not any(
                p.kind == Parameter.VAR_KEYWORD for p in parameters.values()
            ):
                raise ValueError("meta_learner.fit must accept sample_weight for weighted data")
            fit_kwargs["sample_weight"] = weights
        meta.fit(features, target, **fit_kwargs)

        return FittedStackingEnsemble(
            meta_learner=meta,
            fitted_pipelines=list(fitted_pipelines),
            oof_predictions=oof_matrix,
            distribution_aware=distribution_aware,
        )
