from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import polars as pl

from ins_gbm.data.model_data import ModelData

if TYPE_CHECKING:
    from ins_gbm.pipeline import FittedPipeline


@dataclass
class EnsembleResult:
    """Result of running an :class:`EnsemblePipeline`."""
    ensemble: Any  # FittedBlendingEnsemble or FittedStackingEnsemble
    report: Any    # EvaluationReport
    base_pipelines: list["FittedPipeline"]

    def predict(self, data: ModelData) -> pl.Series:
        """Generate ensemble predictions on *data*."""
        return self.ensemble.predict(data)


@dataclass
class EnsemblePipeline:
    """Unified interface for blending and stacking pre-fitted pipelines.

    Parameters
    ----------
    fitted_pipelines : list[FittedPipeline]
        Pre-fitted base pipelines to combine.
    method : {"blending", "stacking"}
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
        Meta-learner for stacking.  Defaults to Ridge.
    """
    fitted_pipelines: list["FittedPipeline"]
    method: Literal["blending", "stacking"] = "blending"
    blend_mode: str = "fixed"
    blend_weights: Optional[list[float]] = None
    validation_data: Optional[ModelData] = None
    cv_folds: int = 5
    seed: int = 42
    meta_learner: Optional[Any] = None

    def run(self) -> EnsembleResult:
        """Fit the ensemble and return an :class:`EnsembleResult` with an evaluation report.

        The evaluation uses the test data from the first base pipeline.
        The test set is never used for weight or meta-learner fitting.
        """
        from ins_gbm.evaluation.report import EvaluationReport
        from ins_gbm.models.base import FittedModel

        if self.method == "blending":
            fitted_ensemble = self._run_blending()
        elif self.method == "stacking":
            fitted_ensemble = self._run_stacking()
        else:
            raise ValueError(
                f"Unknown method: {self.method!r}. Choose 'blending' or 'stacking'."
            )

        test_data = self.fitted_pipelines[0].test_data
        objective = self.fitted_pipelines[0].fitted_model.objective

        # Proxy FittedModel so EvaluationReport can call .predict() and .feature_importance()
        def _predict_fn(data: ModelData, prediction_type: str) -> pl.Series:
            return fitted_ensemble.predict(data)

        def _importance_fn():
            return pl.DataFrame({"feature": pl.Series([], dtype=pl.Utf8),
                                  "importance": pl.Series([], dtype=pl.Float64)})

        proxy_model = FittedModel(
            model=fitted_ensemble,
            params={},
            framework="ensemble",
            objective=objective,
            feature_names=self.fitted_pipelines[0].fitted_model.feature_names,
            predict_fn=_predict_fn,
            importance_fn=_importance_fn,
        )

        report = EvaluationReport(
            fitted_model=proxy_model,
            test_data=test_data,
            train_data=self.fitted_pipelines[0].train_data,
        )

        return EnsembleResult(
            ensemble=fitted_ensemble,
            report=report,
            base_pipelines=list(self.fitted_pipelines),
        )

    def _run_blending(self):
        from ins_gbm.ensemble.blending import BlendingEnsemble
        blender = BlendingEnsemble(
            mode=self.blend_mode,
            weights=self.blend_weights,
            cv_folds=self.cv_folds,
            seed=self.seed,
        )
        return blender.fit(self.fitted_pipelines, validation_data=self.validation_data)

    def _run_stacking(self):
        from ins_gbm.ensemble.stacking import StackingEnsemble
        stacker = StackingEnsemble(
            cv_folds=self.cv_folds,
            seed=self.seed,
            meta_learner=self.meta_learner,
        )
        return stacker.fit(self.fitted_pipelines)
