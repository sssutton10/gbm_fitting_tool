from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from ins_gbm.data.folds import CVConfig, resolve_folds
from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.data.schema import FeatureSchema

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ins_gbm.data.model_data import Objective
    from ins_gbm.pipeline import FeatureStage, ModelRecipe

GBM_MODEL_LABEL: str = "gbm"


def _cv_data_signature(
    objective: str,
    actual: pl.Series,
    exposure: pl.Series | None,
    weight: pl.Series | None,
    row_folds: pl.Series,
) -> str:
    """Fingerprint ordered evaluation inputs and assignments, not features.

    Args:
        objective (str): Model objective: "poisson" or "gamma".
        actual (pl.Series): Observed outcomes aligned with predictions.
        exposure (Optional[pl.Series]): Positive exposure series aligned with rows, when used.
        weight (Optional[pl.Series]): Nonnegative observation weight series aligned with rows.
        row_folds (pl.Series): Fold assignment for each evaluation row.
    """
    digest = hashlib.sha256()
    digest.update(f"cv-v1:{objective}:{len(actual)}".encode())
    for name, series in (
        ("actual", actual),
        ("exposure", exposure),
        ("weight", weight),
    ):
        digest.update(name.encode())
        if series is None:
            digest.update(b":absent")
        else:
            values = np.asarray(series.to_numpy(), dtype="<f8")
            digest.update(values.tobytes())
    digest.update(b"folds:")
    # Fold labels may be integers or strings. Length-prefixed values avoid
    # ambiguous concatenation and keep the hash independent of Polars internals.
    for value in row_folds.to_list():
        encoded = f"{type(value).__name__}:{value}".encode()
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass
class CVResult:
    """Store cross-validation metrics, aligned predictions, and provenance.

    Args:
        fold_metrics (pl.DataFrame): Per-fold metric table.
        summary (pl.DataFrame): Aggregated cross-validation metric table.
        fold_col (Optional[str]): Column containing fold assignments.
        predictions (Optional[pl.DataFrame]): Predictions aligned with evaluation rows.
            Optional.
        actual (Optional[pl.Series]): Observed outcomes aligned with predictions. Optional.
        exposure (Optional[pl.Series]): Positive exposure series aligned with rows, when used.
            Optional.
        weight (Optional[pl.Series]): Nonnegative observation weight series aligned with rows.
            Optional.
        objective (Optional['Objective']): Model objective: "poisson" or "gamma". Optional.
        feature_names (Optional[list[str]]): Ordered names of input features to use. Optional.
        fold_params (Optional[dict]): Best fitting parameters for each validation fold. Optional.
        row_folds (Optional[pl.Series]): Fold assignment for each evaluation row. Optional.
        cv_config (Optional[dict]): Cross-validation configuration used for this result.
            Optional.
        data_signature (Optional[str]): Fingerprint of ordered evaluation inputs and fold assignments. Optional.
    """

    fold_metrics: pl.DataFrame  # columns: fold, model, metric, value
    summary: pl.DataFrame  # columns: model, metric, mean, std
    fold_col: str | None  # None = random folds were used
    predictions: pl.DataFrame | None = None
    actual: pl.Series | None = None
    exposure: pl.Series | None = None
    weight: pl.Series | None = None
    objective: Objective | None = None
    feature_names: list[str] | None = None
    fold_params: dict | None = None
    row_folds: pl.Series | None = None
    cv_config: dict | None = None
    data_signature: str | None = None

    def save(self, output_dir: str) -> None:
        """Save metrics and aligned OOF predictions, without training data.

        Args:
            output_dir (str): Directory for saved artifacts.
        """
        from ins_gbm.persistence.cv_io import save_cv_result

        save_cv_result(self, output_dir)

    def double_lift_score(
        self,
        model_a: str = GBM_MODEL_LABEL,
        model_b: str = "benchmark",
        *,
        n_bins: int = 10,
        deviation: str = "absolute",
    ) -> float:
        """Return the signed OOF double-lift score; positive favors model B.

        Args:
            model_a (str): Name of the reference model.
            model_b (str): Name of the candidate model. Defaults to 'benchmark'.
            n_bins (int): Number of bins used to summarize predictions. Defaults to 10.
            deviation (str): Deviation measure: "absolute" or "relative". Defaults to 'absolute'.
        """
        from ins_gbm.evaluation.metrics import (
            _double_lift_metric_inputs,
            double_lift_score,
            double_lift_table,
        )

        predicted_a, predicted_b = self._comparison_predictions(model_a, model_b)
        actual, predicted_a, predicted_b, weights = _double_lift_metric_inputs(
            self.objective,
            self.actual,
            predicted_a,
            predicted_b,
            self.exposure,
            self.weight,
        )
        table = double_lift_table(
            actual,
            predicted_a,
            predicted_b,
            weights=weights,
            n_bins=min(n_bins, len(actual)),
        )
        return double_lift_score(table, deviation=deviation)

    def plot_double_lift(
        self,
        model_a: str = GBM_MODEL_LABEL,
        model_b: str = "benchmark",
        *,
        n_bins: int = 10,
        output_path: str | None = None,
    ) -> Figure:
        """Plot an out-of-fold double-lift chart for two stored predictions.

        Args:
            model_a (str): Name of the reference model.
            model_b (str): Name of the candidate model. Defaults to 'benchmark'.
            n_bins (int): Number of bins used to summarize predictions. Defaults to 10.
            output_path (Optional[str]): Optional file path for the generated figure.
        """
        from ins_gbm.evaluation.metrics import _double_lift_metric_inputs
        from ins_gbm.evaluation.plots import plot_double_lift

        predicted_a, predicted_b = self._comparison_predictions(model_a, model_b)
        actual, predicted_a, predicted_b, weights = _double_lift_metric_inputs(
            self.objective,
            self.actual,
            predicted_a,
            predicted_b,
            self.exposure,
            self.weight,
        )
        return plot_double_lift(
            actual,
            predicted_a,
            predicted_b,
            weights=weights,
            n_bins=min(n_bins, len(actual)),
            labels=(model_a, model_b),
            output_path=output_path,
        )

    def _comparison_predictions(
        self,
        model_a: str,
        model_b: str,
    ) -> tuple[pl.Series, pl.Series]:
        """Retrieve and validate two aligned prediction series.

        Args:
            model_a (str): Name of the reference model.
            model_b (str): Name of the candidate model.
        """
        if self.predictions is None or self.actual is None or self.objective is None:
            raise ValueError(
                "This CVResult does not contain out-of-fold prediction data"
            )
        missing = [
            name for name in (model_a, model_b) if name not in self.predictions.columns
        ]
        if missing:
            raise KeyError(
                f"Unknown prediction columns {missing}. "
                f"Available: {self.predictions.columns}"
            )
        return self.predictions[model_a], self.predictions[model_b]


@dataclass
class CrossValidationReport:
    """Run and summarize cross-validation for a model recipe.

    Args:
        recipe (ModelRecipe): Unfitted pipeline recipe.
        data (ModelData): Model data to fit, transform, predict, or evaluate.
        n_folds (int): Number of cross-validation folds. Defaults to 5.
        benchmark_col (Optional[str]): Optional column containing benchmark predictions. Optional.
        fold_col (Optional[str]): Column containing fold assignments. Optional.
        seed (int): Random seed for reproducible fitting or splitting. Defaults to 42.
        show_progress_bar (bool): Whether to display a tuning progress bar. Defaults to True.
        cv (Optional[CVConfig]): Cross-validation configuration or explicit fold assignments.
            Optional.
    """

    recipe: ModelRecipe
    data: ModelData
    n_folds: int = 5
    benchmark_col: str | None = None
    fold_col: str | None = None
    seed: int = 42
    show_progress_bar: bool = True
    cv: CVConfig | None = None

    def run(
        self,
        feature_names: list[str] | None = None,
        *,
        feature_stage: FeatureStage = "raw",
    ) -> CVResult:
        """Run CV with an optional raw, encoded, or model feature subset.

        Args:
            feature_names (Optional[list[str]]): Ordered names of input features to use. Optional.
            feature_stage (FeatureStage): Stage at which feature names apply: "raw", "encoded", or
                "model". Defaults to 'raw'.
        """
        self.data.validate()
        self._validate()
        clean_data, fold_ids, benchmark, fold_features = self._prepare_data(
            feature_names, feature_stage
        )
        config = self.cv or CVConfig(
            n_splits=self.n_folds,
            seed=self.seed,
            folds="predefined" if fold_ids is not None else "random",
        )
        uses_predefined = config.folds == "predefined" or (
            config.folds == "auto" and fold_ids is not None
        )
        unique_folds, folds = resolve_folds(clean_data, config)
        rows, predictions, row_folds, fold_params = self._evaluate_folds(
            clean_data, unique_folds, folds, benchmark, fold_features, feature_stage
        )
        fold_metrics = pl.DataFrame(rows).sort(["fold", "model", "metric"])
        summary = (
            fold_metrics.group_by(["model", "metric"])
            .agg(
                [
                    pl.col("value").mean().alias("mean"),
                    pl.col("value").std(ddof=1).alias("std"),
                ]
            )
            .sort(["model", "metric"])
        )
        prediction_columns = {GBM_MODEL_LABEL: pl.Series(GBM_MODEL_LABEL, predictions)}
        if benchmark is not None:
            prediction_columns["benchmark"] = benchmark.rename("benchmark")
        row_folds_series = pl.Series("fold", row_folds)
        return CVResult(
            fold_metrics=fold_metrics,
            summary=summary,
            fold_col=(self.fold_col or "cv_fold") if uses_predefined else None,
            predictions=pl.DataFrame(prediction_columns),
            actual=self.data.target,
            exposure=self.data.exposure,
            weight=self.data.weight,
            objective=self.data.objective,
            feature_names=(
                list(feature_names)
                if feature_stage == "encoded"
                else list(clean_data.feature_names)
            ),
            fold_params=fold_params,
            row_folds=row_folds_series,
            cv_config={
                "folds": "predefined" if uses_predefined else "random",
                "n_splits": len(folds),
                "seed": None if uses_predefined else config.seed,
            },
            data_signature=_cv_data_signature(
                clean_data.objective,
                clean_data.target,
                clean_data.exposure,
                clean_data.weight,
                row_folds_series,
            ),
        )

    def _prepare_data(
        self,
        feature_names: Any,
        feature_stage: FeatureStage,
    ) -> tuple[ModelData, pl.Series | None, pl.Series | None, Any]:
        """Remove fold and benchmark columns before fitting any model.

        Args:
            feature_names (Any): Ordered names of input features to use.
            feature_stage (FeatureStage): Stage at which feature names apply: "raw", "encoded", or
                "model".
        """
        features = self.data.features
        fold_ids = features[self.fold_col] if self.fold_col else self.data.cv_fold
        benchmark = None
        if self.benchmark_col is not None:
            benchmark = features[self.benchmark_col]
        elif self.data.comparisons is not None:
            if self.data.comparisons.width != 1:
                raise ValueError(
                    "Cross-validation accepts one comparison prediction; "
                    "select one before constructing ModelData"
                )
            benchmark = self.data.comparisons.to_series(0)
        cols_to_drop = [
            name for name in (self.fold_col, self.benchmark_col) if name is not None
        ]
        clean_features = features.drop(cols_to_drop) if cols_to_drop else features
        clean_data = replace(
            self.data,
            features=clean_features,
            feature_names=list(clean_features.columns),
            schema=self._clean_schema(cols_to_drop),
            cv_fold=fold_ids,
        )
        from ins_gbm.selection.importance import FittedImportancePruner

        model_selection = (
            feature_names if isinstance(feature_names, FittedImportancePruner) else None
        )
        if feature_stage not in ("raw", "encoded", "model"):
            raise ValueError("feature_stage must be 'raw', 'encoded', or 'model'")
        if model_selection is not None and feature_stage != "raw":
            raise ValueError("feature_stage is inferred from a fitted pruner result")
        if (
            feature_stage == "raw"
            and feature_names is not None
            and model_selection is None
        ):
            clean_data = clean_data.select_features(feature_names)
        fold_feature_names = (
            model_selection
            if model_selection is not None
            else feature_names
            if feature_stage != "raw"
            else None
        )
        return clean_data, fold_ids, benchmark, fold_feature_names

    def _evaluate_folds(
        self,
        clean_data: ModelData,
        unique_folds: list,
        folds: list[tuple[np.ndarray, np.ndarray]],
        benchmark: pl.Series | None,
        fold_features: Any,
        feature_stage: FeatureStage,
    ) -> tuple[list[dict], np.ndarray, list, dict]:
        """Fit each fold and collect aligned predictions and metrics.

        Args:
            clean_data (ModelData): Data after excluding fold and benchmark columns from features.
            unique_folds (list): Distinct fold identifiers to evaluate.
            folds (list[tuple[np.ndarray, np.ndarray]]): Fold policy: "auto", "random", or
                "predefined".
            benchmark (Optional[pl.Series]): Optional benchmark predictions aligned with held-out rows.
            fold_features (Any): Feature subset applied within each fold.
            feature_stage (FeatureStage): Stage at which feature names apply: "raw", "encoded", or
                "model".
        """
        all_fold_rows: list[dict] = []
        oof_gbm = np.full(clean_data.n_rows, np.nan, dtype=np.float64)
        row_folds: list = [None] * clean_data.n_rows
        fold_params: dict = {}

        fold_progress = tqdm(
            zip(unique_folds, folds),
            total=len(folds),
            desc="Cross-validation",
            unit="fold",
            disable=not self.show_progress_bar,
        )
        for fold_id, (train_idx, held_idx) in fold_progress:
            fold_id = fold_id.item() if isinstance(fold_id, np.generic) else fold_id
            for row_idx in held_idx:
                row_folds[int(row_idx)] = fold_id
            train_data = slice_model_data(clean_data, train_idx)
            held_data = slice_model_data(clean_data, held_idx)
            from ins_gbm.pipeline import ModelPipeline

            fitted_pipeline = ModelPipeline(train_data, self.recipe).run(
                feature_names=fold_features,
                feature_stage=feature_stage,
            )
            fold_params[fold_id] = dict(fitted_pipeline.fitted_model.params)
            gbm_preds = fitted_pipeline.predict(held_data, prediction_type="response")
            oof_gbm[held_idx] = gbm_preds.to_numpy()
            all_fold_rows.extend(
                self._fold_metrics(
                    fold_id,
                    clean_data.objective,
                    held_data,
                    gbm_preds,
                    benchmark,
                    held_idx,
                )
            )
        return all_fold_rows, oof_gbm, row_folds, fold_params

    def _fold_metrics(
        self,
        fold_id: Any,
        objective: Objective,
        held_data: ModelData,
        gbm_preds: pl.Series,
        benchmark: pl.Series | None,
        held_idx: np.ndarray,
    ) -> list[dict]:
        """Score one held-out fold, including its optional benchmark.

        Args:
            fold_id (Any): Identifier of the current fold.
            objective (Objective): Model objective: "poisson" or "gamma".
            held_data (ModelData): Held-out data for this fold.
            gbm_preds (pl.Series): Predictions from the GBM on held-out rows.
            benchmark (Optional[pl.Series]): Optional benchmark predictions aligned with held-out rows.
            held_idx (np.ndarray): Original row indices of held-out observations.
        """
        from ins_gbm.evaluation.metrics import (
            _double_lift_metric_inputs,
            compute_metrics,
            double_lift_score,
            double_lift_table,
        )

        def metric_rows(model: str, predictions: pl.Series) -> list[dict]:
            """Build metric rows for one model and prediction series.

            Args:
                model (str): Model wrapper or fitted model to use.
                predictions (pl.Series): Predictions aligned with evaluation rows.
            """
            metrics = compute_metrics(
                objective=objective,
                actual=held_data.target,
                predicted=predictions,
                exposure=held_data.exposure,
                weight=held_data.weight,
            )
            return [
                {"fold": fold_id, "model": model, **row}
                for row in metrics.iter_rows(named=True)
            ]

        rows = metric_rows(GBM_MODEL_LABEL, gbm_preds)
        if benchmark is None:
            return rows
        bench_held = benchmark.gather(held_idx.tolist())
        rows.extend(metric_rows("benchmark", bench_held))
        if len(held_idx) < 2:
            return rows
        dl_actual, dl_gbm, dl_benchmark, dl_weights = _double_lift_metric_inputs(
            objective,
            held_data.target,
            gbm_preds,
            bench_held,
            held_data.exposure,
            held_data.weight,
        )
        dl_table = double_lift_table(
            dl_actual,
            dl_gbm,
            dl_benchmark,
            weights=dl_weights,
            n_bins=min(10, len(held_idx)),
        )
        score = double_lift_score(dl_table)
        for model in (GBM_MODEL_LABEL, "benchmark"):
            rows.append(
                {
                    "fold": fold_id,
                    "model": model,
                    "metric": "double_lift_score",
                    "value": score,
                }
            )
        return rows

    def _validate(self) -> None:
        """Validate cross-validation settings and input data."""
        if self.cv is not None:
            pass
        elif self.fold_col is None:
            if self.n_folds < 2:
                raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
            if self.n_folds > self.data.n_rows:
                raise ValueError(
                    f"n_folds ({self.n_folds}) exceeds number of rows "
                    f"({self.data.n_rows})"
                )
        else:
            if self.fold_col not in self.data.features.columns:
                raise ValueError(f"fold_col {self.fold_col!r} not found in features")
            unique_vals = self.data.features[self.fold_col].drop_nulls().unique()
            if unique_vals.len() < 2:
                raise ValueError(
                    f"fold_col {self.fold_col!r} must have at least 2 distinct "
                    "non-null values"
                )

        if self.benchmark_col is not None:
            if self.benchmark_col not in self.data.features.columns:
                raise ValueError(
                    f"benchmark_col {self.benchmark_col!r} not found in features"
                )
            if self.fold_col is not None and self.fold_col == self.benchmark_col:
                raise ValueError(
                    "fold_col and benchmark_col must not be the same column"
                )
            bench = self.data.features[self.benchmark_col]
            if bench.null_count() > 0:
                raise ValueError(
                    f"benchmark_col {self.benchmark_col!r} contains null values"
                )
            if self.data.objective in ("poisson", "gamma") and (bench <= 0).any():
                raise ValueError(
                    f"benchmark_col {self.benchmark_col!r} must contain "
                    "positive values "
                    f"for {self.data.objective} deviance"
                )

    def _clean_schema(self, cols_to_drop: list[str]) -> FeatureSchema | None:
        """Remove held-out fields from the input feature schema.

        Args:
            cols_to_drop (list[str]): Names of fold or benchmark columns to exclude.
        """
        schema = self.data.schema
        if schema is None:
            return None
        drop_set = set(cols_to_drop)
        return FeatureSchema(
            numeric=[c for c in schema.numeric if c not in drop_set],
            categorical=[c for c in schema.categorical if c not in drop_set],
            ordinal=[c for c in schema.ordinal if c not in drop_set],
            passthrough=[c for c in schema.passthrough if c not in drop_set],
        )

    def _make_folds(
        self,
        fold_id_series: pl.Series | None,
        unique_folds: list,
        n_rows: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Build train and validation indices from assigned fold IDs.

        Args:
            fold_id_series (Optional[pl.Series]): Assigned fold ID for each row.
            unique_folds (list): Distinct fold identifiers to evaluate.
            n_rows (int): Number of rows receiving fold assignments.
        """
        if fold_id_series is not None:
            fold_arr = fold_id_series.to_numpy()
            folds = []
            for fold_id in unique_folds:
                held_mask = fold_arr == fold_id
                held_idx = np.where(held_mask)[0]
                train_idx = np.where(~held_mask)[0]
                folds.append((train_idx, held_idx))
            return folds
        else:
            rng = np.random.default_rng(self.seed)
            indices = np.arange(n_rows)
            rng.shuffle(indices)
            fold_size = n_rows // self.n_folds
            folds = []
            for i in range(self.n_folds):
                start = i * fold_size
                end = start + fold_size if i < self.n_folds - 1 else n_rows
                held_idx = indices[start:end]
                train_idx = np.concatenate([indices[:start], indices[end:]])
                folds.append((train_idx, held_idx))
            return folds
