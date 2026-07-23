from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.data.schema import FeatureSchema
from ins_gbm.ensemble._utils import _apply_recipe_fold_transforms

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ins_gbm.data.model_data import Objective
    from ins_gbm.pipeline import ModelRecipe

GBM_MODEL_LABEL: str = "gbm"


@dataclass
class CVResult:
    fold_metrics: pl.DataFrame   # columns: fold, model, metric, value
    summary: pl.DataFrame        # columns: model, metric, mean, std
    fold_col: Optional[str]      # None = random folds were used
    predictions: Optional[pl.DataFrame] = None
    actual: Optional[pl.Series] = None
    exposure: Optional[pl.Series] = None
    weight: Optional[pl.Series] = None
    objective: Optional["Objective"] = None
    feature_names: Optional[list[str]] = None

    def double_lift_score(
        self,
        model_a: str = GBM_MODEL_LABEL,
        model_b: str = "benchmark",
        *,
        n_bins: int = 10,
        deviation: str = "absolute",
    ) -> float:
        """Return the signed OOF double-lift score; positive favors model B."""
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
        output_path: Optional[str] = None,
    ) -> "Figure":
        """Plot an out-of-fold double-lift chart for two stored predictions."""
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
        if self.predictions is None or self.actual is None or self.objective is None:
            raise ValueError(
                "This CVResult does not contain out-of-fold prediction data"
            )
        missing = [
            name for name in (model_a, model_b)
            if name not in self.predictions.columns
        ]
        if missing:
            raise KeyError(
                f"Unknown prediction columns {missing}. "
                f"Available: {self.predictions.columns}"
            )
        return self.predictions[model_a], self.predictions[model_b]


@dataclass
class CrossValidationReport:
    recipe: ModelRecipe
    data: ModelData
    n_folds: int = 5
    benchmark_col: Optional[str] = None
    fold_col: Optional[str] = None
    seed: int = 42

    def run(self, feature_names: Optional[list[str]] = None) -> CVResult:
        """Run CV, optionally using an ordered subset of raw predictor features."""
        from ins_gbm.evaluation.metrics import (
            _double_lift_metric_inputs,
            compute_metrics,
            double_lift_score,
            double_lift_table,
        )

        self._validate()

        features = self.data.features

        if self.fold_col is not None:
            fold_id_series = features[self.fold_col]
            unique_folds = fold_id_series.drop_nulls().unique().sort().to_list()
        else:
            fold_id_series = None
            unique_folds = list(range(self.n_folds))

        benchmark_preds: Optional[pl.Series] = None
        if self.benchmark_col is not None:
            benchmark_preds = features[self.benchmark_col]

        cols_to_drop = []
        if self.fold_col is not None:
            cols_to_drop.append(self.fold_col)
        if self.benchmark_col is not None:
            cols_to_drop.append(self.benchmark_col)

        clean_features = features.drop(cols_to_drop) if cols_to_drop else features
        clean_schema = self._clean_schema(cols_to_drop)

        clean_data = replace(
            self.data,
            features=clean_features,
            feature_names=list(clean_features.columns),
            schema=clean_schema,
        )
        if feature_names is not None:
            clean_data = clean_data.select_features(feature_names)

        folds = self._make_folds(fold_id_series, unique_folds, clean_data.n_rows)
        all_fold_rows: list[dict] = []
        oof_gbm = np.full(clean_data.n_rows, np.nan, dtype=np.float64)

        for fold_id, (train_idx, held_idx) in zip(unique_folds, folds):
            train_data = slice_model_data(clean_data, train_idx)
            held_data = slice_model_data(clean_data, held_idx)

            current_train, current_held = _apply_recipe_fold_transforms(
                self.recipe, train_data, held_data
            )

            fitted_model = self.recipe.model.fit(
                current_train,
                params=self.recipe.params,
            )
            gbm_preds = fitted_model.predict(current_held, prediction_type="response")
            oof_gbm[held_idx] = gbm_preds.to_numpy()

            gbm_metrics = compute_metrics(
                objective=clean_data.objective,
                actual=current_held.target,
                predicted=gbm_preds,
                exposure=current_held.exposure,
                weight=current_held.weight,
            )
            for row in gbm_metrics.iter_rows(named=True):
                all_fold_rows.append({"fold": fold_id, "model": GBM_MODEL_LABEL, **row})

            if benchmark_preds is not None:
                bench_held = benchmark_preds.gather(held_idx.tolist())
                bench_metrics = compute_metrics(
                    objective=clean_data.objective,
                    actual=current_held.target,
                    predicted=bench_held,
                    exposure=current_held.exposure,
                    weight=current_held.weight,
                )
                for row in bench_metrics.iter_rows(named=True):
                    all_fold_rows.append({"fold": fold_id, "model": "benchmark", **row})

                if len(held_idx) >= 2:
                    dl_actual, dl_gbm, dl_benchmark, dl_weights = (
                        _double_lift_metric_inputs(
                            clean_data.objective,
                            current_held.target,
                            gbm_preds,
                            bench_held,
                            current_held.exposure,
                            current_held.weight,
                        )
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
                        all_fold_rows.append({
                            "fold": fold_id,
                            "model": model,
                            "metric": "double_lift_score",
                            "value": score,
                        })

        fold_metrics = pl.DataFrame(all_fold_rows)
        fold_metrics = fold_metrics.sort(["fold", "model", "metric"])
        summary = (
            fold_metrics
            .group_by(["model", "metric"])
            .agg([
                pl.col("value").mean().alias("mean"),
                pl.col("value").std(ddof=1).alias("std"),
            ])
            .sort(["model", "metric"])
        )

        prediction_columns = {
            GBM_MODEL_LABEL: pl.Series(GBM_MODEL_LABEL, oof_gbm),
        }
        if benchmark_preds is not None:
            prediction_columns["benchmark"] = benchmark_preds.rename("benchmark")

        return CVResult(
            fold_metrics=fold_metrics,
            summary=summary,
            fold_col=self.fold_col,
            predictions=pl.DataFrame(prediction_columns),
            actual=self.data.target,
            exposure=self.data.exposure,
            weight=self.data.weight,
            objective=self.data.objective,
            feature_names=list(clean_data.feature_names),
        )

    def _validate(self) -> None:
        if self.fold_col is None:
            if self.n_folds < 2:
                raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
            if self.n_folds > self.data.n_rows:
                raise ValueError(
                    f"n_folds ({self.n_folds}) exceeds number of rows ({self.data.n_rows})"
                )
        else:
            if self.fold_col not in self.data.features.columns:
                raise ValueError(
                    f"fold_col {self.fold_col!r} not found in features"
                )
            unique_vals = self.data.features[self.fold_col].drop_nulls().unique()
            if unique_vals.len() < 2:
                raise ValueError(
                    f"fold_col {self.fold_col!r} must have at least 2 distinct non-null values"
                )

        if self.benchmark_col is not None:
            if self.benchmark_col not in self.data.features.columns:
                raise ValueError(
                    f"benchmark_col {self.benchmark_col!r} not found in features"
                )
            if self.fold_col is not None and self.fold_col == self.benchmark_col:
                raise ValueError("fold_col and benchmark_col must not be the same column")
            bench = self.data.features[self.benchmark_col]
            if bench.null_count() > 0:
                raise ValueError(
                    f"benchmark_col {self.benchmark_col!r} contains null values"
                )
            if self.data.objective in ("poisson", "gamma") and (bench <= 0).any():
                raise ValueError(
                    f"benchmark_col {self.benchmark_col!r} must contain positive values "
                    f"for {self.data.objective} deviance"
                )

    def _clean_schema(self, cols_to_drop: list[str]) -> Optional[FeatureSchema]:
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
        fold_id_series: Optional[pl.Series],
        unique_folds: list,
        n_rows: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
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
