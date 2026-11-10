from __future__ import annotations

import logging
from contextlib import closing
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from ins_gbm.data.dtypes import FIT_DTYPE
from ins_gbm.data.folds import CVConfig
from ins_gbm.data.model_data import ModelData
from ins_gbm.ensemble._utils import (
    _predict_from_pipeline,
    _resolve_ensemble_folds,
    _validate_ensemble_pipelines,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ins_gbm.pipeline import FittedPipeline


@dataclass
class FittedBlendingEnsemble:
    """A blending ensemble with fitted weights.

    Args:
        weights (list[float]): Nonnegative blending or observation weights.
        fitted_pipelines (list['FittedPipeline']): Base pipelines with fitted models.
        oof_configuration (dict): Requested and effective OOF worker settings.
    """

    weights: list[float]
    fitted_pipelines: list[FittedPipeline]
    oof_configuration: dict = field(default_factory=dict)

    def __setstate__(self, state):
        """Keep older persisted blends usable after adding OOF settings."""
        self.__dict__.update(state)
        self.__dict__.setdefault("oof_configuration", {})

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
    use_data_folds : bool or None
        None uses ``ModelData.cv_fold`` when present, True requires it, and
        False generates random folds. Ignored outside OOF mode.
    cv : CVConfig or None
        Fold configuration overriding ``cv_folds`` and ``seed``. Cannot be
        combined with ``use_data_folds``.
    metric : {"mse", "auto", "poisson_deviance", "gamma_deviance"}
        Loss for optimized weights. Default is MSE; "auto" selects deviance
        for the fitted models' objective. Observation weights are respected;
        Poisson deviance uses exposure-weighted rates, as in evaluation.
    n_jobs : int
        Concurrent OOF fold processes. Default 4; 1 preserves serial model threading.
    threads_per_worker : int
        Threads per OOF process, capped to available CPUs. Default 2.
    show_progress_bar : bool
        Show completed OOF fold fits. Default True. These three settings apply
        only to direct OOF blending, not nested ensemble cross-validation.
    """

    mode: Literal["fixed", "validation", "oof"] = "fixed"
    weights: list[float] | None = None
    cv_folds: int = 5
    seed: int = 42
    refit: str = "fixed"
    use_data_folds: bool | None = None
    cv: CVConfig | None = None
    metric: str = "mse"
    n_jobs: int = 4
    threads_per_worker: int = 2
    show_progress_bar: bool = True

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
        return FittedBlendingEnsemble(
            weights=self._learn_weights(preds, validation_data, pipelines),
            fitted_pipelines=list(pipelines),
        )

    def _fit_oof(self, pipelines) -> FittedBlendingEnsemble:
        """Re-fit each pipeline's recipe inside CV folds to get OOF predictions.

        Args:
            pipelines (object): Base fitted pipelines to combine.
        """
        training_data = [pipeline._require_raw_train_data() for pipeline in pipelines]
        ref_train = training_data[0]
        n = ref_train.n_rows
        fold_splits = _resolve_ensemble_folds(
            ref_train, self.cv, self.cv_folds, self.seed, self.use_data_folds
        )
        oof_preds = np.zeros((n, len(pipelines)), dtype=FIT_DTYPE)

        from tqdm.auto import tqdm

        from ins_gbm.ensemble._oof import (
            _FoldRecipe, _fit_fold, _parallel_folds, _worker_settings,
        )

        settings = _worker_settings(
            self.n_jobs, self.threads_per_worker, len(pipelines) * len(fold_splits)
        )
        threads = settings["effective_threads_per_worker"]
        tasks = (
            (
                _FoldRecipe.from_pipeline(pipeline), pipeline_data, p_idx, fold_idx,
                train_idx, val_idx, self.refit, threads,
            )
            for p_idx, (pipeline, pipeline_data) in enumerate(zip(pipelines, training_data))
            for fold_idx, (train_idx, val_idx) in enumerate(fold_splits)
        )
        started = perf_counter()
        results = (
            (_fit_fold(*task) for task in tasks)
            if self.n_jobs == 1
            else _parallel_folds(tasks, settings["effective_n_jobs"], threads)
        )
        with closing(results), tqdm(
            total=len(pipelines) * len(fold_splits), desc="OOF blend fits",
            unit="fit", disable=not self.show_progress_bar,
        ) as progress:
            for p_idx, _, val_idx, predictions in results:
                oof_preds[val_idx, p_idx] = predictions
                progress.update(1)
        logger.info(
            "OOF prediction generation completed in %.3f seconds", perf_counter() - started
        )
        return FittedBlendingEnsemble(
            weights=self._learn_weights(oof_preds, ref_train, pipelines),
            fitted_pipelines=list(pipelines),
            oof_configuration=settings,
        )

    def _learn_weights(self, preds, data, pipelines) -> list[float]:
        """Resolve the loss against the base models' common objective."""
        objective = pipelines[0].fitted_model.objective
        metric = f"{objective}_deviance" if self.metric == "auto" else self.metric
        if (
            metric in {"poisson_deviance", "gamma_deviance"}
            and metric != f"{objective}_deviance"
        ):
            raise ValueError(
                f"metric {metric!r} is incompatible with objective {objective!r}"
            )
        started = perf_counter()
        weights = _optimize_weights(
            preds,
            data.target.to_numpy(),
            metric=metric,
            sample_weight=data.weight.to_numpy() if data.weight is not None else None,
            exposure=data.exposure.to_numpy() if data.exposure is not None else None,
        ).tolist()
        logger.info(
            "Blend weight optimization completed in %.3f seconds", perf_counter() - started
        )
        return weights


def _optimize_weights(
    preds: np.ndarray,
    actual: np.ndarray,
    *,
    metric: str = "mse",
    sample_weight: np.ndarray | None = None,
    exposure: np.ndarray | None = None,
) -> np.ndarray:
    """Minimize the chosen loss with nonnegative weights summing to one.

    Args:
        preds (np.ndarray): Matrix of base model predictions.
        actual (np.ndarray): Observed outcomes aligned with predictions.
    """
    from scipy.optimize import minimize
    from ins_gbm.evaluation.metrics import (
        _poisson_rate_metric_inputs,
        gamma_deviance,
        poisson_deviance,
    )

    if metric not in {"mse", "poisson_deviance", "gamma_deviance"}:
        raise ValueError(
            f"Unknown metric: {metric!r}. Choose 'mse', 'auto', "
            "'poisson_deviance', or 'gamma_deviance'."
        )
    # SLSQP uses small finite-difference steps; float32 can hide loss changes.
    preds = np.asarray(preds, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if not np.all(np.isfinite(preds)) or not np.all(np.isfinite(actual)):
        raise ValueError("predictions and actual values must be finite")
    if metric != "mse":
        if np.any(preds <= 0):
            raise ValueError("predicted values must be positive for deviance")
        if np.any(actual < 0) or (metric == "gamma_deviance" and np.any(actual == 0)):
            raise ValueError(f"actual values are outside the domain of {metric}")

    y = pl.Series(actual)
    observation_weight = pl.Series(sample_weight) if sample_weight is not None else None
    exposure_series = pl.Series(exposure) if exposure is not None else None

    n_models = preds.shape[1]
    x0 = np.ones(n_models) / n_models

    def objective(w):
        """Objective.

        Args:
            w (object): Candidate blending weights.
        """
        blended = preds @ w
        if metric == "mse":
            return float(np.average((actual - blended) ** 2, weights=sample_weight))
        predicted = pl.Series(blended)
        if metric == "poisson_deviance":
            rate_y, rate_predicted, weights = _poisson_rate_metric_inputs(
                y, predicted, exposure_series, observation_weight
            )
            return poisson_deviance(rate_y, rate_predicted, weights)
        return gamma_deviance(y, predicted, observation_weight)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"ftol": 1e-9},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Blend weight optimization failed: {result.message}")
    return result.x
