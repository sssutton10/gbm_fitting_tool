from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from ins_gbm.data.folds import CVConfig
from ins_gbm.data.model_data import ModelData

if TYPE_CHECKING:
    from ins_gbm.evaluation.cv_report import CVResult
    from ins_gbm.pipeline import FittedPipeline


@dataclass
class EnsembleResult:
    """Result of running an :class:`EnsemblePipeline`.

    Args:
        ensemble (Any): Fitted blending or stacking ensemble.
        base_pipelines (list['FittedPipeline']): Fitted pipelines used by the ensemble.
    """

    ensemble: Any  # FittedBlendingEnsemble or FittedStackingEnsemble
    base_pipelines: list[FittedPipeline]
    configuration: dict = field(default_factory=dict)

    def save(self, output_dir: str) -> None:
        """Persist members and the meta-model without training rows or OOF arrays."""
        from ins_gbm.persistence.ensemble_io import save_ensemble

        save_ensemble(self, output_dir)

    def predict(self, data: ModelData) -> pl.Series:
        """Return response means: expected counts for Poisson, severity for Gamma.

        Poisson rates can be obtained by dividing the returned expected counts
        by data.exposure. Pass count targets and exposure separately; the
        ensemble performs any internal rate normalization itself.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
        """
        return self.ensemble.predict(data)

    def evaluate(self, holdout_data: ModelData):
        """Evaluate the fitted ensemble on separately supplied holdout data.

        Args:
            holdout_data (ModelData): Separate data used for evaluation.
        """
        from ins_gbm.evaluation.report import EvaluationReport
        from ins_gbm.models.base import FittedModel

        def _predict_fn(data: ModelData, prediction_type: str) -> pl.Series:
            """Predict through the ensemble using the fitted model interface.

            Args:
                data (ModelData): Model data to fit, transform, predict, or evaluate.
                prediction_type (str): Prediction scale: "response", "rate", or "link"; "rate" is
                    unavailable for Gamma.
            """
            response = self.ensemble.predict(data)
            if prediction_type == "link":
                return response.log()
            if prediction_type == "rate" and data.exposure is not None:
                return response / data.exposure
            return response

        def _importance_fn(_importance_type: str | None = None):
            """Return ensemble feature importance when available.

            Args:
                _importance_type (Optional[str]): Optional importance measure requested by the caller.
            """
            return pl.DataFrame(
                {
                    "feature": pl.Series([], dtype=pl.Utf8),
                    "importance": pl.Series([], dtype=pl.Float64),
                }
            )

        first = self.base_pipelines[0]
        proxy_model = FittedModel(
            model=self.ensemble,
            params={},
            framework="ensemble",
            objective=first.fitted_model.objective,
            feature_names=first.fitted_model.feature_names,
            predict_fn=_predict_fn,
            importance_fn=_importance_fn,
        )
        comparison_predictions = None
        if holdout_data.comparisons is not None:
            comparison_predictions = {
                name: holdout_data.comparisons[name]
                for name in holdout_data.comparisons.columns
            }
        return EvaluationReport(
            fitted_model=proxy_model,
            evaluation_data=holdout_data,
            train_data=None,
            comparison_predictions=comparison_predictions,
        )


@dataclass
class EnsemblePipeline:
    """Unified interface for blending and stacking pre-fitted pipelines.

    Parameters
    ----------
    fitted_pipelines : list[FittedPipeline]
        Base pipelines to combine. Seed averaging and bagging require exactly
        one template with its original training rows attached.
    method : {"blending", "stacking", "seed_average", "bagging"}
        Ensemble strategy.
    blend_mode : {"fixed", "validation", "oof"}
        Blending weight strategy (used only when ``method="blending"``).
    blend_weights : list[float] or None
        Fixed weights for ``blend_mode="fixed"``.
    validation_data : ModelData or None
        Held-out blend set for ``blend_mode="validation"``.
    cv_folds : int
        CV folds for stacking OOF generation or blending OOF mode.
    seed : int
        Random seed.
    meta_learner : sklearn estimator or None
        None/"auto" selects Poisson/Gamma stacking. "ridge" retains the
        previous response-scale Ridge behavior; custom estimators are supported.

    refit : {"fixed", "retune"}
        Use fitted model parameters in each OOF fold, or retune each fold.
    use_data_folds : bool or None
        None reuses ``ModelData.cv_fold`` when present, True requires it, and
        False generates random OOF folds.
    cv : CVConfig or None
        Override ``cv_folds`` and ``seed`` with the shared fold configuration.
        Cannot be combined with ``use_data_folds``.
    n_members : int
        Number of seed-average/bagging members, fitted sequentially. Default 5.
        Tuned parameters are reused; transforms/selection are refitted.
    sample_fraction : float
        Fraction of training rows per bag, in (0, 1]. Default 1.0.
    bootstrap : bool
        Bagging samples with replacement by default; False samples without it.
    meta_alpha : float
        Regularization for built-in stacking learners. Default 1.0.
    metric : {"mse", "auto", "poisson_deviance", "gamma_deviance"}
        Loss for optimizing blend weights. "auto" selects deviance for the
        base models' distribution. Defaults to MSE. Used only for blending.
    """

    fitted_pipelines: list[FittedPipeline]
    method: Literal["blending", "stacking", "seed_average", "bagging"] = "blending"
    blend_mode: str = "fixed"
    blend_weights: list[float] | None = None
    validation_data: ModelData | None = None
    cv_folds: int = 5
    seed: int = 42
    meta_learner: Any | None = None
    refit: str = "fixed"
    use_data_folds: bool | None = None
    cv: CVConfig | None = None
    metric: str = "mse"
    n_members: int = 5
    sample_fraction: float = 1.0
    bootstrap: bool = True
    meta_alpha: float = 1.0

    def cross_validate(
        self, data: ModelData, *, cv: CVConfig | None = None,
        base_refit: str = "fixed", inner_cv: CVConfig | None = None,
        validation_fraction: float = 0.2, show_progress_bar: bool = True,
    ) -> CVResult:
        """Evaluate this ensemble on training-data outer folds, returning CVResult.

        Rebuild all models and transforms within each outer training fold.
        Weight learning/stacking uses inner folds; validation blending uses an
        internal split. The reserved final holdout is never needed. Outer
        ``base_refit`` defaults to fitted params; ``self.refit`` independently
        controls inner fitting. ``inner_cv`` overrides this candidate's inner
        fold settings. Outputs are Poisson expected counts or Gamma severity.
        """
        from ins_gbm.evaluation.candidates import cross_validate_candidates

        return cross_validate_candidates(
            {"ensemble": self}, data, cv=cv, base_refit=base_refit,
            inner_cv=inner_cv, validation_fraction=validation_fraction,
            show_progress_bar=show_progress_bar,
        )["ensemble"]

    def run(self) -> EnsembleResult:
        """Fit the ensemble using the base pipelines' complete training data."""

        if self.method == "blending":
            fitted_ensemble = self._run_blending()
        elif self.method == "stacking":
            fitted_ensemble = self._run_stacking()
        elif self.method in {"seed_average", "bagging"}:
            from ins_gbm.ensemble.resampling import ResamplingEnsemble

            fitted_ensemble, settings = ResamplingEnsemble(
                method=self.method, n_members=self.n_members, seed=self.seed,
                sample_fraction=self.sample_fraction, bootstrap=self.bootstrap,
            ).fit(self.fitted_pipelines)
            return EnsembleResult(
                ensemble=fitted_ensemble,
                base_pipelines=fitted_ensemble.fitted_pipelines,
                configuration=settings,
            )
        else:
            raise ValueError(
                f"Unknown method: {self.method!r}. Choose 'blending', 'stacking', "
                "'seed_average', or 'bagging'."
            )

        return EnsembleResult(
            ensemble=fitted_ensemble,
            base_pipelines=list(self.fitted_pipelines),
            configuration={
                "method": self.method, "blend_mode": self.blend_mode,
                "metric": self.metric, "refit": self.refit,
                "cv_folds": self.cv.n_splits if self.cv is not None else self.cv_folds,
                "seed": self.cv.seed if self.cv is not None else self.seed,
                "folds": self.cv.folds if self.cv is not None else (
                    "auto" if self.use_data_folds is None
                    else "predefined" if self.use_data_folds else "random"
                ),
                "meta_learner": type(fitted_ensemble.meta_learner).__name__
                if self.method == "stacking" else None,
                "meta_alpha": self.meta_alpha,
            },
        )

    def _run_blending(self):
        """Fit a blending ensemble from the configured pipelines."""
        from ins_gbm.ensemble.blending import BlendingEnsemble

        blender = BlendingEnsemble(
            mode=self.blend_mode,
            weights=self.blend_weights,
            cv_folds=self.cv_folds,
            seed=self.seed,
            refit=self.refit,
            use_data_folds=self.use_data_folds,
            cv=self.cv,
            metric=self.metric,
        )
        return blender.fit(self.fitted_pipelines, validation_data=self.validation_data)

    def _run_stacking(self):
        """Fit a stacking ensemble from the configured pipelines."""
        from ins_gbm.ensemble.stacking import StackingEnsemble

        stacker = StackingEnsemble(
            cv_folds=self.cv_folds,
            seed=self.seed,
            meta_learner=self.meta_learner,
            meta_alpha=self.meta_alpha,
            refit=self.refit,
            use_data_folds=self.use_data_folds,
            cv=self.cv,
        )
        return stacker.fit(self.fitted_pipelines)
