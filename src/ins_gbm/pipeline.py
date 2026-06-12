from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import polars as pl

from ins_gbm.data.model_data import ModelData
from ins_gbm.data.splitter import TrainTestSplit
from ins_gbm.models.base import FittedModel, PredictionType
from ins_gbm.tuning.tuner import HyperparameterTuner
from ins_gbm.persistence.metadata import ReproducibilityMetadata
from ins_gbm.progress import ProgressCallback, ProgressEvent, PipelineCancelled


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

    ``train_data`` and ``test_data`` hold the *transformed* versions of each
    split — i.e., the features as they were passed to the model.  The raw
    parquet data is not stored here.
    """
    fitted_model: FittedModel
    recipe: ModelRecipe
    train_data: ModelData
    test_data: ModelData
    selected_features: Optional[list[str]]
    tuning_history: Optional[pl.DataFrame]
    report: Any  # EvaluationReport — imported lazily to avoid circular imports
    encoder: Optional[Any]
    preprocessors: list
    metadata: ReproducibilityMetadata

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
        current = data
        if self.encoder is not None:
            current = current.with_features(self.encoder.transform(current.features))
        if self.selected_features is not None:
            missing = [
                f for f in self.selected_features if f not in current.features.columns
            ]
            if missing:
                raise ValueError(
                    f"Selected features missing after encoding: {missing}"
                )
            current = current.with_features(
                current.features.select(self.selected_features)
            )
        for prep in self.preprocessors:
            current = current.with_features(prep.transform(current.features))
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
            schema=self.train_data.schema,
            objective=obj,
        )
        return self.predict(data, prediction_type=prediction_type)


@dataclass
class ModelPipeline:
    """Full train → tune → fit → evaluate orchestrator.

    Execution order
    ---------------
    1. Split data into train and an untouched test set.
    2. (Optional) Tune on training data: encoder, selector, and preprocessor
       are refit independently on each CV fold to prevent leakage.
    3. Refit encoder → selector → preprocessors → model on the *full*
       training set using the best hyperparameters.
    4. Apply the same fitted transformations to the test set.
    5. Evaluate *once* on the untouched (now transformed) test set.
    """
    data: ModelData
    split: TrainTestSplit
    recipe: ModelRecipe
    progress: Optional[ProgressCallback] = None
    should_stop: Optional[Any] = None

    def _emit(self, stage: str, message: str, **kwargs) -> None:
        if self.progress is not None:
            self.progress(ProgressEvent(stage=stage, message=message, **kwargs))

    def _check_cancel(self) -> None:
        if self.should_stop is not None and self.should_stop():
            raise PipelineCancelled("pipeline cancelled by caller")

    def run(self) -> FittedPipeline:
        from ins_gbm.evaluation.report import EvaluationReport
        from ins_gbm.persistence.metadata import build_metadata

        # ── 1. Split ──────────────────────────────────────────────────────────
        self._emit("split", "splitting data into train/test")
        train_data, test_data = self.split.split(self.data)
        self._check_cancel()

        # ── 2. Tune (optional) ────────────────────────────────────────────────
        tuning_history: Optional[pl.DataFrame] = None
        best_params: dict = {}
        if self.recipe.tuning is not None:
            self._emit(
                "tuning", "starting hyperparameter tuning",
                total=self.recipe.tuning.n_trials,
            )
            # Pass only the first preprocessor to the tuner; multiple-preprocessor
            # chains are handled in the full refit step below.
            single_prep = (
                self.recipe.preprocessing[0]
                if self.recipe.preprocessing
                else None
            )
            best_params, tuning_history = self.recipe.tuning.tune(
                train_data,
                self.recipe.model,
                encoder=self.recipe.encoder,
                selector=self.recipe.selection,
                preprocessor=single_prep,
                schema=getattr(train_data, "schema", None),
                progress=self.progress,
                should_stop=self.should_stop,
            )
            self._check_cancel()

        # ── 3. Refit on full training data ────────────────────────────────────
        current_train = train_data
        current_test = test_data
        fitted_encoder: Optional[Any] = None

        if self.recipe.encoder is not None:
            self._emit("encode", "fitting encoder on full training data")
            self._check_cancel()
            schema = getattr(current_train, "schema", None)
            fitted_encoder = self.recipe.encoder.fit(current_train.features, schema)
            current_train = current_train.with_features(
                fitted_encoder.transform(current_train.features)
            )
            current_test = current_test.with_features(
                fitted_encoder.transform(current_test.features)
            )

        selected_features: Optional[list[str]] = None
        if self.recipe.selection is not None:
            self._emit("select", "running feature selection")
            self._check_cancel()
            fitted_sel = self.recipe.selection.fit(current_train)
            selected_features = fitted_sel.selected_features()
            current_train = current_train.with_features(
                current_train.features.select(selected_features)
            )
            current_test = current_test.with_features(
                current_test.features.select(selected_features)
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
            current_test = current_test.with_features(
                fitted_prep.transform(current_test.features)
            )
            fitted_preprocessors.append(fitted_prep)

        self._emit("fit", "fitting model on full training data")
        self._check_cancel()
        fitted_model = self.recipe.model.fit(
            current_train,
            params=best_params if best_params else self.recipe.params,
        )

        # ── 4. Evaluate once on the test set ──────────────────────────────────
        self._emit("evaluate", "evaluating on test set")
        comp_preds = None
        if current_test.comparisons is not None:
            comp_preds = {
                col: current_test.comparisons[col]
                for col in current_test.comparisons.columns
            }
        report = EvaluationReport(
            fitted_model=fitted_model,
            test_data=current_test,
            train_data=current_train,
            comparison_predictions=comp_preds,
        )

        # ── 5. Capture reproducibility metadata ───────────────────────────────
        metadata = build_metadata(
            fitted_model=fitted_model,
            selected_features=selected_features,
            split_seed=getattr(self.split, "seed", None),
            tuning_seed=(
                getattr(self.recipe.tuning, "seed", None)
                if self.recipe.tuning is not None
                else None
            ),
        )

        return FittedPipeline(
            fitted_model=fitted_model,
            recipe=self.recipe,
            train_data=current_train,
            test_data=current_test,
            selected_features=selected_features,
            tuning_history=tuning_history,
            report=report,
            encoder=fitted_encoder,
            preprocessors=fitted_preprocessors,
            metadata=metadata,
        )
