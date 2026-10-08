from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import polars as pl

from ins_gbm.data.model_data import ModelData
from ins_gbm.data.schema import FeatureSchema
from ins_gbm.data.selection_candidates import (
    select_encoded_candidates,
    select_raw_candidates,
    selector_without_candidates,
)
from ins_gbm.models.base import FittedModel, PredictionType
from ins_gbm.persistence.metadata import ReproducibilityMetadata
from ins_gbm.preprocessing.chain import FittedTransformChain, select_model_features
from ins_gbm.progress import PipelineCancelled, ProgressCallback, ProgressEvent
from ins_gbm.selection.importance import FittedImportancePruner
from ins_gbm.tuning.tuner import HyperparameterTuner

FeatureStage = Literal["raw", "encoded", "model"]
SelectionOutcome = tuple[
    ModelData, list[str] | None, list[Any] | None, list[dict] | None
]


@dataclass
class ModelRecipe:
    """Cloneable, unfitted pipeline configuration.

    Used by ``ModelPipeline.run()``, the hyperparameter tuner, and the stacking
    ensemble (which refits recipes inside CV folds).

    Args:
        model (Any): Model wrapper or fitted model to use.
        encoder (Optional[Any]): Optional encoder applied before model fitting.
        selection (Optional[Any]): Optional feature selection configuration.
        preprocessing (list): Ordered preprocessing steps applied before fitting.
        tuning (Optional[HyperparameterTuner]): Optional hyperparameter tuner.
        params (Optional[dict]): Optional model or estimator parameter mapping.
        selection_scope (Literal['fold', 'fixed']): Feature selection scope: "fixed" or "fold".
            Defaults to 'fixed'.
    """

    model: Any
    encoder: Any | None = None
    selection: Any | None = None
    preprocessing: list = field(default_factory=list)
    tuning: HyperparameterTuner | None = None
    # Fixed/base hyperparameters. Tuning suggestions override overlapping keys.
    params: dict | None = None
    selection_scope: Literal["fold", "fixed"] = "fixed"

    def fit(self, data: ModelData, **kwargs) -> FittedPipeline:
        """Fit this recipe; convenient equivalent of ``ModelPipeline(...).run()``.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
            kwargs (object): Additional keyword arguments forwarded to the pipeline run.
        """
        return ModelPipeline(data=data, recipe=self).run(**kwargs)

    def cross_validate(
        self,
        data: ModelData,
        *,
        cv=None,
        feature_names=None,
        feature_stage: FeatureStage = "raw",
    ):
        """Evaluate folds with an optional raw, encoded, or model feature subset.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
            cv (object): Cross-validation configuration or explicit fold assignments. Optional.
            feature_names (object): Ordered names of input features to use. Optional.
            feature_stage (FeatureStage): Stage at which feature names apply: "raw", "encoded", or
                "model". Defaults to 'raw'.
        """
        from ins_gbm.evaluation.cv_report import CrossValidationReport

        return CrossValidationReport(recipe=self, data=data, cv=cv).run(
            feature_names, feature_stage=feature_stage
        )


@dataclass
class FittedPipeline:
    """Result of running a ``ModelPipeline``.

    The raw training data is retained in memory for OOF ensemble workflows, but
    is omitted from persisted artifacts. The expanded transformed training
    matrix is reconstructed only when ``train_data`` is explicitly accessed and
    is never cached on this object. CatBoost's importance callback separately
    retains its training Pool until LossFunctionChange is requested or saved.

    Args:
        fitted_model (FittedModel): Fitted model to evaluate or persist.
        recipe (ModelRecipe): Unfitted pipeline recipe.
        input_feature_names (list[str]): Names of features expected before fitted transforms.
        raw_train_data (Optional[ModelData]): Original training data retained for evaluation or
            refitting.
        selected_features (Optional[list[str]]): Feature names retained by selection.
        selection_results (Optional[list[Any]]): Results from feature selection.
        tuning_history (Optional[pl.DataFrame]): Trial history returned by tuning, if run.
        encoder (Optional[Any]): Optional encoder applied before model fitting.
        preprocessors (list): Ordered preprocessing steps or their fitted counterparts.
        metadata (ReproducibilityMetadata): Reproducibility metadata for the fitted pipeline.
        input_schema (Optional[FeatureSchema]): Input feature schema used by the fitted
            pipeline.
        model_selected_features (Optional[list[str]]): Optional feature subset applied after
            preprocessing.
    """

    fitted_model: FittedModel
    recipe: ModelRecipe
    input_feature_names: list[str]
    raw_train_data: ModelData | None
    selected_features: list[str] | None
    selection_results: list[Any] | None
    tuning_history: pl.DataFrame | None
    encoder: Any | None
    preprocessors: list
    metadata: ReproducibilityMetadata
    input_schema: FeatureSchema | None = None
    model_selected_features: list[str] | None = None

    @property
    def train_data(self) -> ModelData:
        """Reconstruct transformed training data without retaining the matrix."""
        return self._prepare_data(self._require_raw_train_data())

    def _require_raw_train_data(self) -> ModelData:
        """Return attached training rows or explain how to restore them."""
        if self.raw_train_data is None:
            raise RuntimeError(
                "Training data is not attached to this fitted pipeline. "
                "Reload it with load_pipeline(..., "
                "training_data=original_training_data) "
                "before accessing train_data or fitting an OOF ensemble."
            )
        return self.raw_train_data

    def _input_schema(self) -> FeatureSchema | None:
        """Return the compact input schema, including for legacy artifacts."""
        schema = getattr(self, "input_schema", None)
        if schema is None and self.raw_train_data is not None:
            return self.raw_train_data.schema
        return schema

    def _prepare_data(self, data: ModelData) -> ModelData:
        """Select fitted raw inputs and apply the fitted transform chain.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
        """
        return FittedTransformChain(
            input_feature_names=self.input_feature_names,
            encoder=self.encoder,
            selected_features=self.selected_features,
            preprocessors=self.preprocessors,
            model_selected_features=getattr(self, "model_selected_features", None),
        ).transform(data)

    def predict(
        self,
        data,
        prediction_type: PredictionType = "response",
        *,
        exposure=None,
        weight=None,
        offset=None,
    ) -> pl.Series:
        """Apply the fitted transform chain to *data* and return predictions.

        Applies transforms in the same order as ModelPipeline.run():
        encode → select → preprocess → model.predict().
        Pass raw (pre-transform) data; the fitted transformers handle encoding.

        Args:
            data (object): Model data to fit, transform, predict, or evaluate.
            prediction_type (PredictionType): Prediction scale: "response", "rate", or "link";
                "rate" is unavailable for Gamma. Defaults to 'response'.
            exposure (object): Positive exposure series aligned with rows, when used. Optional.
            weight (object): Nonnegative observation weight series aligned with rows. Optional.
            offset (object): Optional model offset on the link scale.
        """
        if isinstance(data, pl.DataFrame):
            return self.predict_raw(
                data,
                exposure=exposure,
                weight=weight,
                offset=offset,
                prediction_type=prediction_type,
            )
        if any(value is not None for value in (exposure, weight, offset)):
            raise ValueError(
                "exposure, weight, and offset cannot override fields of ModelData"
            )
        current = self._prepare_data(data)
        return self.fitted_model.predict(current, prediction_type=prediction_type)

    def predict_raw(
        self,
        features: pl.DataFrame,
        exposure: pl.Series | None = None,
        weight: pl.Series | None = None,
        offset: pl.Series | None = None,
        prediction_type: PredictionType = "response",
    ) -> pl.Series:
        """Score a raw feature DataFrame without a target column.

        Constructs a ModelData with a placeholder target (never used for
        prediction) so the full transform chain can be applied.

        Args:
            features (pl.DataFrame): Input feature frame; rows align with the target and optional
                series.
            exposure (Optional[pl.Series]): Positive exposure series aligned with rows, when used.
                Optional.
            weight (Optional[pl.Series]): Nonnegative observation weight series aligned with rows.
                Optional.
            offset (Optional[pl.Series]): Optional model offset on the link scale.
            prediction_type (PredictionType): Prediction scale: "response", "rate", or "link";
                "rate" is unavailable for Gamma. Defaults to 'response'.
        """
        n = features.height
        if exposure is not None and len(exposure) != n:
            raise ValueError(f"exposure length {len(exposure)} != features height {n}")
        if weight is not None and len(weight) != n:
            raise ValueError(f"weight length {len(weight)} != features height {n}")
        if offset is not None and len(offset) != n:
            raise ValueError(f"offset length {len(offset)} != features height {n}")
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
            offset=offset,
        )
        return self.predict(data, prediction_type=prediction_type)

    def evaluate(self, holdout_data: ModelData):
        """Evaluate this fitted pipeline on separately supplied holdout data.

        The fitted transform chain is applied to the holdout, but neither the
        raw nor transformed holdout is stored on the fitted pipeline.

        Args:
            holdout_data (ModelData): Separate data used for evaluation.
        """
        from ins_gbm.evaluation.report import EvaluationReport

        holdout_data.validate()
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

    def save(self, output_dir: str) -> None:
        """Persist this fitted pipeline.

        Args:
            output_dir (str): Directory for saved artifacts.
        """
        from ins_gbm.persistence.io import save_pipeline

        save_pipeline(self, output_dir)

    def retune(
        self,
        tuner: HyperparameterTuner,
        *,
        progress: ProgressCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> FittedPipeline:
        """Tune again while freezing the fitted encoder and feature selection.

        The original pipeline is not mutated. Preprocessors are fit independently
        inside each tuning fold, then refit on all attached training rows before
        fitting the returned pipeline's model.

        Args:
            tuner (HyperparameterTuner): Tuner used to refit hyperparameters.
            progress (Optional[ProgressCallback]): Optional callback receiving progress events.
            should_stop (Optional[Callable[[], bool]]): Optional callback that requests cancellation
                when true.
        """
        from ins_gbm.persistence.metadata import build_metadata
        from ins_gbm.preprocessing.steps import validate_preprocessing_steps

        raw_train_data = self._require_raw_train_data()
        validate_preprocessing_steps(self.recipe.preprocessing)
        runner = ModelPipeline(
            data=raw_train_data,
            recipe=self.recipe,
            progress=progress,
            should_stop=should_stop,
        )
        runner._check_cancel()
        tuning_data = FittedTransformChain(
            input_feature_names=self.input_feature_names,
            encoder=self.encoder,
            selected_features=self.selected_features,
        ).transform(raw_train_data)

        runner._emit(
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
            base_params=self.recipe.params,
            **(
                {"model_selected_features": self.model_selected_features}
                if getattr(self, "model_selected_features", None) is not None
                else {}
            ),
        )
        runner._check_cancel()
        current_train, fitted_preprocessors = runner._fit_preprocessors(tuning_data)

        model_selected_features = getattr(self, "model_selected_features", None)
        if model_selected_features is not None:
            current_train = select_model_features(
                current_train, model_selected_features
            )

        fitted_model = runner._fit_model(current_train, best_params)

        tuned_recipe = replace(self.recipe, tuning=tuner)
        metadata = build_metadata(
            fitted_model=fitted_model,
            selected_features=self.selected_features,
            model_selected_features=model_selected_features,
            input_feature_names=self.input_feature_names,
            tuning_seed=getattr(tuner, "seed", None),
            selection_stages=getattr(self.metadata, "selection_stages", None),
            selection_scope="fixed",
            tuning_metric=(
                getattr(tuner, "metric", None) or f"{fitted_model.objective}_deviance"
            ),
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

    Args:
        data (ModelData): Model data to fit, transform, predict, or evaluate.
        recipe (ModelRecipe): Unfitted pipeline recipe.
        progress (Optional[ProgressCallback]): Optional callback receiving progress events.
        should_stop (Optional[Any]): Optional callback that requests cancellation when true.
    """

    data: ModelData
    recipe: ModelRecipe
    progress: ProgressCallback | None = None
    should_stop: Any | None = None

    def _emit(self, stage: str, message: str, **kwargs) -> None:
        """Send a pipeline progress event to the configured callback.

        Args:
            stage (str): Pipeline stage associated with the event.
            message (str): Human-readable progress message.
            kwargs (object): Additional keyword arguments forwarded to the pipeline run.
        """
        if self.progress is not None:
            self.progress(ProgressEvent(stage=stage, message=message, **kwargs))

    def _check_cancel(self) -> None:
        """Raise when the pipeline cancellation callback requests a stop."""
        if self.should_stop is not None and self.should_stop():
            raise PipelineCancelled("pipeline cancelled by caller")

    def _validate_feature_selection(
        self,
        feature_names: list[str] | None,
        feature_stage: FeatureStage,
    ) -> None:
        """Check the requested feature subset and selection stage.

        Args:
            feature_names (list[str] | None): Ordered names of input features to use.
            feature_stage (FeatureStage): Stage at which feature names apply: "raw", "encoded", or
                "model".
        """
        if feature_stage not in ("raw", "encoded", "model"):
            raise ValueError("feature_stage must be 'raw', 'encoded', or 'model'")
        if feature_stage == "raw":
            return
        if feature_names is None:
            raise ValueError(
                f"feature_names is required when feature_stage={feature_stage!r}"
            )
        if not feature_names:
            raise ValueError("feature_names must contain at least one feature")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names must be unique")
        if self.recipe.selection is not None:
            raise ValueError(
                f"feature_stage={feature_stage!r} cannot be combined with "
                "recipe.selection; supplied feature_names are the fixed "
                "final selection"
            )

    def _tune(
        self,
        data: ModelData,
        feature_names: list[str] | None,
        feature_stage: FeatureStage,
        *,
        fold_local: bool,
    ) -> tuple[dict, pl.DataFrame | None]:
        """Tune model parameters with the requested feature selection.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
            feature_names (list[str] | None): Ordered names of input features to use.
            feature_stage (FeatureStage): Stage at which feature names apply: "raw", "encoded", or
                "model".
            fold_local (bool): Whether tuning refits selection inside each CV fold.
        """
        tuner = self.recipe.tuning
        if tuner is None:
            return {}, None
        message = (
            "starting fold-local hyperparameter tuning"
            if fold_local
            else "starting hyperparameter tuning"
        )
        self._emit("tuning", message, total=tuner.n_trials)
        options = {
            "preprocessors": self.recipe.preprocessing,
            "base_params": self.recipe.params,
            "progress": self.progress,
            "should_stop": self.should_stop,
        }
        if feature_stage == "model":
            options["model_selected_features"] = list(feature_names)
        if fold_local:
            options.update(encoder=self.recipe.encoder, selector=self.recipe.selection)
        result = tuner.tune(data, self.recipe.model, **options)
        self._check_cancel()
        return result

    def _encode(self, data: ModelData) -> tuple[ModelData, Any | None]:
        """Fit the encoder and transform training features.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
        """
        if self.recipe.encoder is None:
            return data, None
        self._emit("encode", "fitting encoder on full training data")
        self._check_cancel()
        encoder = self.recipe.encoder.fit(data.features, getattr(data, "schema", None))
        return data.with_features(encoder.transform(data.features)), encoder

    def _select(
        self,
        data: ModelData,
        feature_names: list[str] | None,
        feature_stage: FeatureStage,
    ) -> SelectionOutcome:
        """Apply a fixed feature subset or fit the configured selector.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
            feature_names (list[str] | None): Ordered names of input features to use.
            feature_stage (FeatureStage): Stage at which feature names apply: "raw", "encoded", or
                "model".
        """
        if feature_stage == "encoded":
            selected = list(feature_names or [])
            missing = [name for name in selected if name not in data.features.columns]
            if missing:
                raise ValueError(f"Encoded features missing after encoding: {missing}")
            selected_data = data.with_features(data.features.select(selected))
            return selected_data, selected, None, None
        if self.recipe.selection is None:
            return data, None, None, None
        self._emit("select", "running feature selection")
        self._check_cancel()
        selection_data = select_encoded_candidates(data, self.recipe.selection)
        fitted = selector_without_candidates(self.recipe.selection).fit(selection_data)
        selected = fitted.selected_features()
        stage_results = getattr(fitted, "stage_results", None)
        selection_metadata = getattr(fitted, "selection_metadata", None)
        results = stage_results() if callable(stage_results) else None
        metadata = selection_metadata() if callable(selection_metadata) else None
        selected_data = data.with_features(data.features.select(selected))
        return selected_data, selected, results, metadata

    def _fit_preprocessors(self, data: ModelData) -> tuple[ModelData, list[Any]]:
        """Fit and apply preprocessing steps in order.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
        """
        fitted_preprocessors = []
        for prep in self.recipe.preprocessing:
            self._emit("preprocess", f"fitting preprocessor {type(prep).__name__}")
            self._check_cancel()
            fitted = prep.fit(data.features, data.target)
            data = data.with_features(fitted.transform(data.features))
            fitted_preprocessors.append(fitted)
        return data, fitted_preprocessors

    def _fit_model(self, data: ModelData, best_params: dict) -> FittedModel:
        """Fit the model with tuned or configured parameters.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
            best_params (dict): Best parameter values selected by tuning.
        """
        self._emit("fit", "fitting model on full training data")
        self._check_cancel()
        fitted = self.recipe.model.fit(
            data, params=best_params if best_params else self.recipe.params
        )
        self._check_cancel()
        return fitted

    def run(
        self,
        feature_names: list[str] | FittedImportancePruner | None = None,
        *,
        feature_stage: FeatureStage = "raw",
    ) -> FittedPipeline:
        """Fit with an optional raw, encoded, or fitted-model feature subset.

        Args:
            feature_names (list[str] | FittedImportancePruner | None): Ordered names of input
                features to use. Optional.
            feature_stage (FeatureStage): Stage at which feature names apply: "raw", "encoded", or
                "model". Defaults to 'raw'.
        """
        from ins_gbm.persistence.metadata import build_metadata
        from ins_gbm.preprocessing.steps import validate_preprocessing_steps

        self.data.validate(require_multiple_folds=False)
        validate_preprocessing_steps(self.recipe.preprocessing)
        if isinstance(feature_names, FittedImportancePruner):
            if feature_stage != "raw":
                raise ValueError(
                    "feature_stage is inferred from a fitted pruner result"
                )
            feature_names = feature_names.selected_features()
            feature_stage = "model"
        self._validate_feature_selection(feature_names, feature_stage)

        train_data = self.data
        if feature_stage == "raw" and feature_names is not None:
            train_data = self.data.select_features(feature_names)
        train_data = select_raw_candidates(train_data, self.recipe.selection)
        input_feature_names = list(train_data.feature_names)
        raw_train_data = train_data
        self._check_cancel()

        if self.recipe.selection_scope not in {"fold", "fixed"}:
            raise ValueError("selection_scope must be 'fold' or 'fixed'")

        fold_local_tuning = (
            self.recipe.tuning is not None
            and self.recipe.selection_scope == "fold"
            and feature_stage in {"raw", "model"}
        )
        if fold_local_tuning:
            best_params, tuning_history = self._tune(
                train_data, feature_names, feature_stage, fold_local=True
            )
        else:
            best_params, tuning_history = {}, None

        current_train, fitted_encoder = self._encode(train_data)
        current_train, selected_features, selection_results, selection_metadata = (
            self._select(current_train, feature_names, feature_stage)
        )

        if self.recipe.tuning is not None and not fold_local_tuning:
            best_params, tuning_history = self._tune(
                current_train, feature_names, feature_stage, fold_local=False
            )

        current_train, fitted_preprocessors = self._fit_preprocessors(current_train)

        model_selected_features = (
            list(feature_names) if feature_stage == "model" else None
        )
        if model_selected_features is not None:
            current_train = select_model_features(
                current_train, model_selected_features
            )

        fitted_model = self._fit_model(current_train, best_params)

        # ── 4. Capture reproducibility metadata ───────────────────────────────
        metadata = build_metadata(
            fitted_model=fitted_model,
            selected_features=selected_features,
            model_selected_features=model_selected_features,
            input_feature_names=input_feature_names,
            tuning_seed=(
                getattr(self.recipe.tuning, "seed", None)
                if self.recipe.tuning is not None
                else None
            ),
            selection_stages=selection_metadata,
            selection_scope=self.recipe.selection_scope,
            tuning_metric=(
                getattr(self.recipe.tuning, "metric", None)
                or f"{fitted_model.objective}_deviance"
                if self.recipe.tuning
                else None
            ),
        )

        return FittedPipeline(
            fitted_model=fitted_model,
            recipe=self.recipe,
            input_feature_names=input_feature_names,
            raw_train_data=raw_train_data,
            input_schema=raw_train_data.schema,
            selected_features=selected_features,
            model_selected_features=model_selected_features,
            selection_results=selection_results,
            tuning_history=tuning_history,
            encoder=fitted_encoder,
            preprocessors=fitted_preprocessors,
            metadata=metadata,
        )
