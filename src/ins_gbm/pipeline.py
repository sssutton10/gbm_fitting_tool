from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Optional

import polars as pl

from ins_gbm.data.model_data import ModelData
from ins_gbm.data.schema import FeatureSchema
from ins_gbm.models.base import FittedModel, PredictionType
from ins_gbm.tuning.tuner import HyperparameterTuner
from ins_gbm.persistence.metadata import ReproducibilityMetadata
from ins_gbm.progress import ProgressCallback, ProgressEvent, PipelineCancelled
from ins_gbm.preprocessing.chain import FittedTransformChain


FeatureStage = Literal["raw", "encoded"]


@dataclass
class ModelRecipe:
    """Cloneable, unfitted pipeline configuration.

    Used by ``ModelPipeline.run()``, the hyperparameter tuner, and the stacking
    ensemble (which refits recipes inside CV folds).
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

    The raw training data is retained in memory for OOF ensemble workflows, but
    is omitted from persisted artifacts. The expanded transformed training
    matrix is reconstructed only when ``train_data`` is explicitly accessed and
    is never cached on this object.
    """
    fitted_model: FittedModel
    recipe: ModelRecipe
    input_feature_names: list[str]
    raw_train_data: Optional[ModelData]
    selected_features: Optional[list[str]]
    selection_results: Optional[list[Any]]
    tuning_history: Optional[pl.DataFrame]
    encoder: Optional[Any]
    preprocessors: list
    metadata: ReproducibilityMetadata
    input_schema: Optional[FeatureSchema] = None

    @property
    def train_data(self) -> ModelData:
        """Reconstruct transformed training data without retaining the matrix."""
        return self._prepare_data(self._require_raw_train_data())

    def _require_raw_train_data(self) -> ModelData:
        """Return attached training rows or explain how to restore them."""
        if self.raw_train_data is None:
            raise RuntimeError(
                "Training data is not attached to this fitted pipeline. "
                "Reload it with load_pipeline(..., training_data=original_training_data) "
                "before accessing train_data or fitting an OOF ensemble."
            )
        return self.raw_train_data

    def _input_schema(self) -> Optional[FeatureSchema]:
        """Return the compact input schema, including for legacy artifacts."""
        schema = getattr(self, "input_schema", None)
        if schema is None and self.raw_train_data is not None:
            return self.raw_train_data.schema
        return schema

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
            schema=self._input_schema(),
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

    def retune(
        self,
        tuner: HyperparameterTuner,
        *,
        progress: Optional[ProgressCallback] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> "FittedPipeline":
        """Tune again while freezing the fitted encoder and feature selection.

        The original pipeline is not mutated. Preprocessors are fit independently
        inside each tuning fold, then refit on all attached training rows before
        fitting the returned pipeline's model.
        """
        from ins_gbm.persistence.metadata import build_metadata
        from ins_gbm.preprocessing.steps import validate_preprocessing_steps

        raw_train_data = self._require_raw_train_data()
        validate_preprocessing_steps(self.recipe.preprocessing)

        def emit(stage: str, message: str, **kwargs) -> None:
            if progress is not None:
                progress(ProgressEvent(stage=stage, message=message, **kwargs))

        def check_cancel() -> None:
            if should_stop is not None and should_stop():
                raise PipelineCancelled("pipeline cancelled by caller")

        check_cancel()
        tuning_data = FittedTransformChain(
            input_feature_names=self.input_feature_names,
            encoder=self.encoder,
            selected_features=self.selected_features,
        ).transform(raw_train_data)

        emit(
            "tuning",
            "starting hyperparameter retuning",
            total=tuner.n_trials,
        )
        best_params, tuning_history = tuner.tune(
            tuning_data,
            self.recipe.model,
            preprocessors=self.recipe.preprocessing,
            progress=progress,
            should_stop=should_stop,
        )
        check_cancel()

        current_train = tuning_data
        fitted_preprocessors: list[Any] = []
        for prep in self.recipe.preprocessing:
            emit("preprocess", f"fitting preprocessor {type(prep).__name__}")
            check_cancel()
            fitted_prep = prep.fit(current_train.features, current_train.target)
            current_train = current_train.with_features(
                fitted_prep.transform(current_train.features)
            )
            fitted_preprocessors.append(fitted_prep)

        emit("fit", "fitting model on full training data")
        check_cancel()
        fitted_model = self.recipe.model.fit(
            current_train,
            params=best_params if best_params else self.recipe.params,
        )
        check_cancel()

        tuned_recipe = replace(self.recipe, tuning=tuner)
        metadata = build_metadata(
            fitted_model=fitted_model,
            selected_features=self.selected_features,
            input_feature_names=self.input_feature_names,
            tuning_seed=getattr(tuner, "seed", None),
            selection_stages=getattr(self.metadata, "selection_stages", None),
        )

        return replace(
            self,
            fitted_model=fitted_model,
            recipe=tuned_recipe,
            raw_train_data=raw_train_data,
            input_schema=self._input_schema(),
            tuning_history=tuning_history,
            preprocessors=fitted_preprocessors,
            metadata=metadata,
        )


@dataclass
class ModelPipeline:
    """Full-data select → tune → fit orchestrator.

    Execution order
    ---------------
    1. Fit the encoder and complete feature-selection workflow on every supplied
       training row, producing one final selected feature matrix.
    2. (Optional) Tune with cross-validation on that fixed feature matrix.
       Preprocessors are still refit independently on each CV fold.
    3. Fit preprocessors and the final model on every supplied training row
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

    def run(
        self,
        feature_names: Optional[list[str]] = None,
        *,
        feature_stage: FeatureStage = "raw",
    ) -> FittedPipeline:
        """Fit the recipe with an optional raw or post-encoding feature subset."""
        from ins_gbm.persistence.metadata import build_metadata
        from ins_gbm.preprocessing.steps import validate_preprocessing_steps

        validate_preprocessing_steps(self.recipe.preprocessing)
        if feature_stage not in ("raw", "encoded"):
            raise ValueError("feature_stage must be 'raw' or 'encoded'")
        if feature_stage == "encoded":
            if feature_names is None:
                raise ValueError(
                    "feature_names is required when feature_stage='encoded'"
                )
            if not feature_names:
                raise ValueError("feature_names must contain at least one feature")
            if len(set(feature_names)) != len(feature_names):
                raise ValueError("feature_names must be unique")
            if self.recipe.selection is not None:
                raise ValueError(
                    "feature_stage='encoded' cannot be combined with "
                    "recipe.selection; encoded feature_names are the fixed "
                    "final selection"
                )

        train_data = self.data
        if feature_stage == "raw" and feature_names is not None:
            train_data = self.data.select_features(feature_names)
        input_feature_names = list(train_data.feature_names)
        raw_train_data = train_data
        self._check_cancel()

        # ── 1. Encode and complete feature selection ─────────────────────────
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
        if feature_stage == "encoded":
            selected_features = list(feature_names or [])
            missing = [
                name
                for name in selected_features
                if name not in current_train.features.columns
            ]
            if missing:
                raise ValueError(
                    f"Encoded features missing after encoding: {missing}"
                )
            current_train = current_train.with_features(
                current_train.features.select(selected_features)
            )
        elif self.recipe.selection is not None:
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

        # ── 2. Tune on the fixed final feature selection (optional) ───────────
        tuning_history: Optional[pl.DataFrame] = None
        best_params: dict = {}
        if self.recipe.tuning is not None:
            self._emit(
                "tuning", "starting hyperparameter tuning",
                total=self.recipe.tuning.n_trials,
            )
            best_params, tuning_history = self.recipe.tuning.tune(
                current_train,
                self.recipe.model,
                preprocessors=self.recipe.preprocessing,
                progress=self.progress,
                should_stop=self.should_stop,
            )
            self._check_cancel()

        # ── 3. Fit preprocessors and model on full training data ──────────────
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

        # ── 4. Capture reproducibility metadata ───────────────────────────────
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
            input_schema=raw_train_data.schema,
            selected_features=selected_features,
            selection_results=selection_results,
            tuning_history=tuning_history,
            encoder=fitted_encoder,
            preprocessors=fitted_preprocessors,
            metadata=metadata,
        )
