from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import polars as pl

from ins_gbm.data.model_data import ModelData
from ins_gbm.models.base import FittedModel, PredictionType
from ins_gbm.tuning.tuner import HyperparameterTuner
from ins_gbm.persistence.metadata import ReproducibilityMetadata
from ins_gbm.progress import ProgressCallback, ProgressEvent, PipelineCancelled
from ins_gbm.preprocessing.chain import FittedTransformChain


@dataclass
class ModelRecipe:
    """Cloneable, unfitted pipeline configuration.

    Used by ``ModelPipeline.run()``, the hyperparameter tuner (each CV trial
    refits the full recipe), and the stacking ensemble (refits inside CV folds).
    """
    model: Any
    encoder: Optional[Any] = None
    selection: Optional[Any] = None
    preprocessing: list = field(default_factory=list)
    tuning: Optional[HyperparameterTuner] = None
    # Manual hyperparameters used when ``tuning`` is None. Ignored when tuning
    # is enabled (the tuned best params take precedence).
    params: Optional[dict] = None


@dataclass
class FittedPipeline:
    """Result of running a ``ModelPipeline``.

    The raw training data is retained for OOF ensemble workflows. The expanded
    transformed training matrix is reconstructed only when ``train_data`` is
    explicitly accessed and is never cached on this object.
    """
    fitted_model: FittedModel
    recipe: ModelRecipe
    input_feature_names: list[str]
    raw_train_data: ModelData
    selected_features: Optional[list[str]]
    selection_results: Optional[list[Any]]
    tuning_history: Optional[pl.DataFrame]
    encoder: Optional[Any]
    preprocessors: list
    metadata: ReproducibilityMetadata

    @property
    def train_data(self) -> ModelData:
        """Reconstruct transformed training data without retaining the matrix."""
        return self._prepare_data(self.raw_train_data)

    def _prepare_data(self, data: ModelData) -> ModelData:
        """Select fitted raw inputs and apply the fitted transform chain."""
        return FittedTransformChain(
            input_feature_names=self.input_feature_names,
            encoder=self.encoder,
            selected_features=self.selected_features,
            preprocessors=self.preprocessors,
        ).transform(data)

    def predict(
        self,
        data: ModelData,
        prediction_type: PredictionType = "response",
    ) -> pl.Series:
        """Apply the fitted transform chain to *data* and return predictions.

        Applies transforms in the same order as ModelPipeline.run():
        encode → select → preprocess → model.predict().
        Pass raw (pre-transform) data; the fitted transformers handle encoding.
        """
        current = self._prepare_data(data)
        return self.fitted_model.predict(current, prediction_type=prediction_type)

    def predict_raw(
        self,
        features: pl.DataFrame,
        exposure: Optional[pl.Series] = None,
        weight: Optional[pl.Series] = None,
        prediction_type: PredictionType = "response",
    ) -> pl.Series:
        """Score a raw feature DataFrame without a target column.

        Constructs a ModelData with a placeholder target (never used for
        prediction) so the full transform chain can be applied.
        """
        n = features.height
        if exposure is not None and len(exposure) != n:
            raise ValueError(
                f"exposure length {len(exposure)} != features height {n}"
            )
        if weight is not None and len(weight) != n:
            raise ValueError(
                f"weight length {len(weight)} != features height {n}"
            )
        obj = self.fitted_model.objective
        placeholder = (
            pl.Series("_target", [0.0] * n)
            if obj == "poisson"
            else pl.Series("_target", [1.0] * n)
        )
        data = ModelData(
            features=features,
            target=placeholder,
            exposure=exposure,
            weight=weight,
            feature_names=list(features.columns),
            schema=self.raw_train_data.schema,
            objective=obj,
        )
        return self.predict(data, prediction_type=prediction_type)

    def evaluate(self, holdout_data: ModelData):
        """Evaluate this fitted pipeline on separately supplied holdout data.

        The fitted transform chain is applied to the holdout, but neither the
        raw nor transformed holdout is stored on the fitted pipeline.
        """
        from ins_gbm.evaluation.report import EvaluationReport

        current = self._prepare_data(holdout_data)

        comparison_predictions = None
        if current.comparisons is not None:
            comparison_predictions = {
                name: current.comparisons[name] for name in current.comparisons.columns
            }
        return EvaluationReport(
            fitted_model=self.fitted_model,
            evaluation_data=current,
            train_data=None,
            comparison_predictions=comparison_predictions,
        )


@dataclass
class ModelPipeline:
    """Full-data train → tune → fit orchestrator.

    Execution order
    ---------------
    1. (Optional) Tune with cross-validation: encoder, selector, and preprocessor
       are refit independently on each CV fold to prevent leakage.
    2. Fit encoder → selector → preprocessors → model on every supplied row
       using the best hyperparameters.

    Use :meth:`FittedPipeline.evaluate` to evaluate a separately supplied
    holdout after fitting.
    """
    data: ModelData
    recipe: ModelRecipe
    progress: Optional[ProgressCallback] = None
    should_stop: Optional[Any] = None

    def _emit(self, stage: str, message: str, **kwargs) -> None:
        if self.progress is not None:
            self.progress(ProgressEvent(stage=stage, message=message, **kwargs))

    def _check_cancel(self) -> None:
        if self.should_stop is not None and self.should_stop():
            raise PipelineCancelled("pipeline cancelled by caller")

    def run(self, feature_names: Optional[list[str]] = None) -> FittedPipeline:
        """Fit the recipe, optionally using an ordered subset of raw features."""
        from ins_gbm.persistence.metadata import build_metadata
        from ins_gbm.preprocessing.steps import validate_preprocessing_steps

        validate_preprocessing_steps(self.recipe.preprocessing)
        train_data = (
            self.data.select_features(feature_names)
            if feature_names is not None
            else self.data
        )
        input_feature_names = list(train_data.feature_names)
        raw_train_data = train_data
        self._check_cancel()

        # ── 1. Tune (optional) ────────────────────────────────────────────────
        tuning_history: Optional[pl.DataFrame] = None
        best_params: dict = {}
        if self.recipe.tuning is not None:
            self._emit(
                "tuning", "starting hyperparameter tuning",
                total=self.recipe.tuning.n_trials,
            )
            best_params, tuning_history = self.recipe.tuning.tune(
                train_data,
                self.recipe.model,
                encoder=self.recipe.encoder,
                selector=self.recipe.selection,
                preprocessors=self.recipe.preprocessing,
                schema=getattr(train_data, "schema", None),
                progress=self.progress,
                should_stop=self.should_stop,
            )
            self._check_cancel()

        # ── 2. Fit on full training data ──────────────────────────────────────
        current_train = train_data
        fitted_encoder: Optional[Any] = None

        if self.recipe.encoder is not None:
            self._emit("encode", "fitting encoder on full training data")
            self._check_cancel()
            schema = getattr(current_train, "schema", None)
            fitted_encoder = self.recipe.encoder.fit(current_train.features, schema)
            current_train = current_train.with_features(
                fitted_encoder.transform(current_train.features)
            )

        selected_features: Optional[list[str]] = None
        selection_results: Optional[list[Any]] = None
        selection_metadata: Optional[list[dict]] = None
        if self.recipe.selection is not None:
            self._emit("select", "running feature selection")
            self._check_cancel()
            fitted_sel = self.recipe.selection.fit(current_train)
            selected_features = fitted_sel.selected_features()
            stage_results = getattr(fitted_sel, "stage_results", None)
            if callable(stage_results):
                selection_results = stage_results()
            get_selection_metadata = getattr(fitted_sel, "selection_metadata", None)
            if callable(get_selection_metadata):
                selection_metadata = get_selection_metadata()
            current_train = current_train.with_features(
                current_train.features.select(selected_features)
            )

        fitted_preprocessors: list = []
        for prep in self.recipe.preprocessing:
            self._emit("preprocess", f"fitting preprocessor {type(prep).__name__}")
            self._check_cancel()
            # Pass target so supervised reducers (e.g. PLS) can fit; unsupervised
            # reducers accept and ignore it (fit(features, target=None)).
            fitted_prep = prep.fit(current_train.features, current_train.target)
            current_train = current_train.with_features(
                fitted_prep.transform(current_train.features)
            )
            fitted_preprocessors.append(fitted_prep)

        self._emit("fit", "fitting model on full training data")
        self._check_cancel()
        fitted_model = self.recipe.model.fit(
            current_train,
            params=best_params if best_params else self.recipe.params,
        )
        self._check_cancel()

        # ── 3. Capture reproducibility metadata ───────────────────────────────
        metadata = build_metadata(
            fitted_model=fitted_model,
            selected_features=selected_features,
            input_feature_names=input_feature_names,
            tuning_seed=(
                getattr(self.recipe.tuning, "seed", None)
                if self.recipe.tuning is not None
                else None
            ),
            selection_stages=selection_metadata,
        )

        return FittedPipeline(
            fitted_model=fitted_model,
            recipe=self.recipe,
            input_feature_names=input_feature_names,
            raw_train_data=raw_train_data,
            selected_features=selected_features,
            selection_results=selection_results,
            tuning_history=tuning_history,
            encoder=fitted_encoder,
            preprocessors=fitted_preprocessors,
            metadata=metadata,
        )
