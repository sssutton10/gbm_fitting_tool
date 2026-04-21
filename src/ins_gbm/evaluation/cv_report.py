from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.data.schema import FeatureSchema
from ins_gbm.ensemble._utils import _apply_recipe_fold_transforms

if TYPE_CHECKING:
    from ins_gbm.pipeline import ModelRecipe

GBM_MODEL_LABEL: str = "gbm"


@dataclass
class CVResult:
    fold_metrics: pl.DataFrame   # columns: fold, model, metric, value
    summary: pl.DataFrame        # columns: model, metric, mean, std
    fold_col: Optional[str]      # None = random folds were used


@dataclass
class CrossValidationReport:
    recipe: ModelRecipe
    data: ModelData
    n_folds: int = 5
    benchmark_col: Optional[str] = None
    fold_col: Optional[str] = None
    seed: int = 42

    def run(self) -> CVResult:
        from ins_gbm.evaluation.metrics import compute_metrics

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

        clean_data = ModelData(
            features=clean_features,
            target=self.data.target,
            exposure=self.data.exposure,
            weight=self.data.weight,
            feature_names=list(clean_features.columns),
            schema=clean_schema,
            objective=self.data.objective,
        )

        folds = self._make_folds(fold_id_series, unique_folds, clean_data.n_rows)
        all_fold_rows: list[dict] = []

        for fold_id, (train_idx, held_idx) in zip(unique_folds, folds):
            train_data = slice_model_data(clean_data, train_idx)
            held_data = slice_model_data(clean_data, held_idx)

            current_train, current_held = _apply_recipe_fold_transforms(
                self.recipe, train_data, held_data
            )

            fitted_model = self.recipe.model.fit(current_train)
            gbm_preds = fitted_model.predict(current_held, prediction_type="response")

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

        return CVResult(fold_metrics=fold_metrics, summary=summary, fold_col=self.fold_col)

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
