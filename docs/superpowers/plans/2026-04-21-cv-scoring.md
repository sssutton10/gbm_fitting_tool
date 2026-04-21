# CV Stability Report, Benchmark Comparison & Pipeline Scoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `CrossValidationReport` for fold-stability metrics and benchmark comparison, a `compare_reports()` function for side-by-side model comparison, and `FittedPipeline.predict()` / `predict_raw()` methods for scoring new data.

**Architecture:** Four tasks build on each other: first extract a shared `compute_metrics()` helper (Task 1) that both the existing `EvaluationReport` and the new CV report use, then add scoring methods to `FittedPipeline` and fix the `load_pipeline` bug (Task 2), then build `CrossValidationReport` (Task 3), then build `compare_reports()` and wire up exports (Task 4).

**Tech Stack:** Polars, NumPy, LightGBM (for tests), cloudpickle (fix in persistence).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/ins_gbm/evaluation/metrics.py` | Modify | Add `compute_metrics()`, `METRIC_DIRECTIONS` |
| `src/ins_gbm/evaluation/report.py` | Modify | Use `compute_metrics()` in `_single_metrics()` |
| `src/ins_gbm/evaluation/cv_report.py` | Create | `CrossValidationReport`, `CVResult` |
| `src/ins_gbm/evaluation/comparison.py` | Create | `compare_reports()` |
| `src/ins_gbm/evaluation/__init__.py` | Modify | Export new public symbols |
| `src/ins_gbm/pipeline.py` | Modify | Add `FittedPipeline.predict()`, `predict_raw()` |
| `src/ins_gbm/persistence/io.py` | Modify | Fix `load_pipeline` to use `cloudpickle.load` |
| `tests/evaluation/test_metrics.py` | Modify | Tests for `compute_metrics()` |
| `tests/evaluation/test_cv_report.py` | Create | Tests for `CrossValidationReport` |
| `tests/evaluation/test_comparison.py` | Create | Tests for `compare_reports()` |
| `tests/test_pipeline.py` | Modify | Tests for `predict()` and `predict_raw()` |
| `tests/persistence/test_persistence.py` | Modify | Cloudpickle roundtrip test |

---

## Task 1: Shared metric helper

**Files:**
- Modify: `src/ins_gbm/evaluation/metrics.py`
- Modify: `src/ins_gbm/evaluation/report.py`
- Test: `tests/evaluation/test_metrics.py`

- [ ] **Step 1.1: Write failing tests for `compute_metrics()` and `METRIC_DIRECTIONS`**

Add to `tests/evaluation/test_metrics.py`:

```python
import polars as pl
import pytest
from ins_gbm.evaluation.metrics import (
    METRIC_DIRECTIONS,
    compute_metrics,
)


def test_compute_metrics_poisson_returns_four_metrics():
    actual = pl.Series([1.0, 2.0, 3.0])
    predicted = pl.Series([1.1, 1.9, 3.2])
    exposure = pl.Series([1.0, 1.0, 1.0])
    result = compute_metrics(
        objective="poisson",
        actual=actual,
        predicted=predicted,
        exposure=exposure,
    )
    assert set(result["metric"].to_list()) == {"poisson_deviance", "gini", "rmse", "mae"}
    assert result["value"].dtype == pl.Float64


def test_compute_metrics_gamma_returns_four_metrics():
    actual = pl.Series([100.0, 200.0, 300.0])
    predicted = pl.Series([110.0, 190.0, 320.0])
    weight = pl.Series([1.0, 1.0, 1.0])
    result = compute_metrics(
        objective="gamma",
        actual=actual,
        predicted=predicted,
        weight=weight,
    )
    assert set(result["metric"].to_list()) == {"gamma_deviance", "gini", "rmse", "mae"}


def test_compute_metrics_matches_individual_functions():
    from ins_gbm.evaluation.metrics import poisson_deviance, normalized_gini, rmse, mae
    actual = pl.Series([1.0, 0.0, 2.0])
    predicted = pl.Series([0.9, 0.1, 2.1])
    exposure = pl.Series([1.5, 0.5, 1.0])
    result = compute_metrics(
        objective="poisson",
        actual=actual,
        predicted=predicted,
        exposure=exposure,
    )
    expected_deviance = poisson_deviance(actual, predicted, weights=exposure)
    row = result.filter(pl.col("metric") == "poisson_deviance")["value"][0]
    assert abs(row - expected_deviance) < 1e-10


def test_metric_directions_has_all_keys():
    assert METRIC_DIRECTIONS["gini"] == "higher"
    assert METRIC_DIRECTIONS["poisson_deviance"] == "lower"
    assert METRIC_DIRECTIONS["gamma_deviance"] == "lower"
    assert METRIC_DIRECTIONS["rmse"] == "lower"
    assert METRIC_DIRECTIONS["mae"] == "lower"
```

- [ ] **Step 1.2: Run to verify FAIL**

```
source /c/Users/sssut/anaconda3/etc/profile.d/conda.sh && conda activate base
pytest tests/evaluation/test_metrics.py::test_compute_metrics_poisson_returns_four_metrics -v
```

Expected: `ImportError: cannot import name 'compute_metrics'`

- [ ] **Step 1.3: Add `compute_metrics()` and `METRIC_DIRECTIONS` to `metrics.py`**

In `src/ins_gbm/evaluation/metrics.py`, add `Literal` to the existing `typing` import line so it reads:

```python
from typing import Literal, Optional
```

Add after the existing `mae` function:

```python
Objective = Literal["poisson", "gamma"]

METRIC_DIRECTIONS: dict[str, str] = {
    "gini": "higher",
    "poisson_deviance": "lower",
    "gamma_deviance": "lower",
    "rmse": "lower",
    "mae": "lower",
}


def compute_metrics(
    *,
    objective: Objective,
    actual: pl.Series,
    predicted: pl.Series,
    exposure: Optional[pl.Series] = None,
    weight: Optional[pl.Series] = None,
) -> pl.DataFrame:
    rows: list[dict] = []
    if objective == "poisson":
        rows.append({
            "metric": "poisson_deviance",
            "value": poisson_deviance(actual, predicted, weights=exposure),
        })
    else:
        rows.append({
            "metric": "gamma_deviance",
            "value": gamma_deviance(actual, predicted, weights=weight),
        })
    gini_weights = exposure if exposure is not None else weight
    rows.append({"metric": "gini",
                 "value": normalized_gini(actual, predicted, weights=gini_weights)})
    rows.append({"metric": "rmse", "value": rmse(actual, predicted)})
    rows.append({"metric": "mae", "value": mae(actual, predicted)})
    return pl.DataFrame(rows)
```

- [ ] **Step 1.4: Run new tests**

```
pytest tests/evaluation/test_metrics.py -v
```

Expected: all four new tests PASS (existing tests must also pass).

- [ ] **Step 1.5: Refactor `EvaluationReport._single_metrics()` to use `compute_metrics()`**

Replace the `_single_metrics` method in `src/ins_gbm/evaluation/report.py`:

```python
def _single_metrics(self) -> pl.DataFrame:
    from ins_gbm.evaluation.metrics import compute_metrics
    return compute_metrics(
        objective=self.fitted_model.objective,
        actual=self.test_data.target,
        predicted=self.fitted_model.predict(self.test_data, prediction_type="response"),
        exposure=self.test_data.exposure,
        weight=self.test_data.weight,
    )
```

- [ ] **Step 1.6: Run all evaluation tests to verify no regressions**

```
pytest tests/evaluation/ -v
```

Expected: all existing tests PASS.

- [ ] **Step 1.7: Commit**

```bash
git add src/ins_gbm/evaluation/metrics.py src/ins_gbm/evaluation/report.py tests/evaluation/test_metrics.py
git commit -m "feat(metrics): add compute_metrics() helper and METRIC_DIRECTIONS constant"
```

---

## Task 2: Fix `load_pipeline` + add `FittedPipeline.predict()` and `predict_raw()`

**Files:**
- Modify: `src/ins_gbm/persistence/io.py`
- Modify: `src/ins_gbm/pipeline.py`
- Test: `tests/persistence/test_persistence.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 2.1: Write failing tests for `FittedPipeline.predict()` (no transforms)**

Add to `tests/test_pipeline.py`:

```python
def test_predict_no_transforms_matches_model_predict(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    result = ModelPipeline(
        data=data,
        split=TrainTestSplit(train_ratio=0.8, seed=0),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()
    # No encoder/selector/preprocessors — predict() is a direct passthrough to the model
    direct = result.fitted_model.predict(result.test_data, prediction_type="response")
    via_predict = result.predict(result.test_data, prediction_type="response")
    assert direct.to_list() == pytest.approx(via_predict.to_list(), rel=1e-6)
```

- [ ] **Step 2.2: Write failing test for `predict_raw()` with encoder**

Add to `tests/test_pipeline.py`:

```python
from ins_gbm.data.model_data import ModelData
from ins_gbm.data.schema import infer_schema
from ins_gbm.preprocessing.encoder import OneHotEncoder


def test_predict_raw_matches_pipeline_predictions(poisson_raw):
    schema = infer_schema(poisson_raw, ["x1", "x2", "x3"])
    data = ModelData(
        features=poisson_raw.select(["x1", "x2", "x3"]),
        target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"],
        weight=None,
        feature_names=["x1", "x2", "x3"],
        schema=schema,
        objective="poisson",
    ).validate()

    result = ModelPipeline(
        data=data,
        split=TrainTestSplit(train_ratio=0.7, seed=42),
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            encoder=OneHotEncoder(),
        ),
    ).run()

    # Recreate the exact same split to recover the raw test rows
    _, raw_test = TrainTestSplit(train_ratio=0.7, seed=42).split(data)

    via_predict_raw = result.predict_raw(
        features=raw_test.features,
        exposure=raw_test.exposure,
    )
    direct = result.fitted_model.predict(result.test_data, prediction_type="response")
    assert via_predict_raw.to_list() == pytest.approx(direct.to_list(), rel=1e-6)


def test_predict_raw_wrong_exposure_length_raises(poisson_raw):
    schema = infer_schema(poisson_raw, ["x1", "x3"])
    data = ModelData(
        features=poisson_raw.select(["x1", "x3"]),
        target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"],
        weight=None,
        feature_names=["x1", "x3"],
        schema=schema,
        objective="poisson",
    ).validate()
    result = ModelPipeline(
        data=data,
        split=TrainTestSplit(train_ratio=0.8, seed=0),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()
    bad_exposure = pl.Series([1.0, 2.0])  # wrong length
    with pytest.raises(ValueError, match="exposure length"):
        result.predict_raw(features=poisson_raw.select(["x1", "x3"]), exposure=bad_exposure)
```

- [ ] **Step 2.3: Run to verify FAIL**

```
pytest tests/test_pipeline.py::test_predict_no_transforms_matches_model_predict -v
```

Expected: `AttributeError: 'FittedPipeline' object has no attribute 'predict'`

- [ ] **Step 2.4: Fix `load_pipeline` to use `cloudpickle.load`**

In `src/ins_gbm/persistence/io.py`, replace `load_pipeline`:

```python
def load_pipeline(output_dir: str) -> "FittedPipeline":
    """Load a FittedPipeline previously saved with :func:`save_pipeline`."""
    import cloudpickle

    path = os.path.join(output_dir, _PIPELINE_FILE)
    try:
        with open(path, "rb") as f:
            return cloudpickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No pipeline artifact found at {path!r}. "
            "Was save_pipeline() called with the same output_dir?"
        )
```

- [ ] **Step 2.5: Add `predict()` and `predict_raw()` to `FittedPipeline`**

In `src/ins_gbm/pipeline.py`, add `PredictionType` to the existing `models.base` import so it reads:

```python
from ins_gbm.models.base import FittedModel, PredictionType
```

Then add the following two methods to the `FittedPipeline` dataclass (after all existing fields):

```python
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
```

- [ ] **Step 2.6: Run new tests**

```
pytest tests/test_pipeline.py -v -k "predict"
```

Expected: all three new `predict` tests PASS.

- [ ] **Step 2.7: Add cloudpickle roundtrip test to persistence tests**

Add to `tests/persistence/test_persistence.py`:

```python
def test_load_pipeline_and_predict_raw(poisson_parquet, poisson_raw, tmp_path):
    """Loaded pipeline can score new raw data via predict_raw()."""
    result = _build_pipeline(poisson_parquet)
    out = str(tmp_path / "pipeline_out")
    save_pipeline(result, out)
    loaded = load_pipeline(out)

    preds = loaded.predict_raw(
        features=poisson_raw.select(["x1", "x3"]),
        exposure=poisson_raw["exposure"],
    )
    assert isinstance(preds, pl.Series)
    assert preds.len() == poisson_raw.height
    assert (preds > 0).all()
```

- [ ] **Step 2.8: Run persistence tests**

```
pytest tests/persistence/ -v
```

Expected: all persistence tests PASS.

- [ ] **Step 2.9: Commit**

```bash
git add src/ins_gbm/pipeline.py src/ins_gbm/persistence/io.py \
        tests/test_pipeline.py tests/persistence/test_persistence.py
git commit -m "feat(pipeline): add FittedPipeline.predict/predict_raw; fix cloudpickle load"
```

---

## Task 3: `CrossValidationReport` and `CVResult`

**Files:**
- Create: `src/ins_gbm/evaluation/cv_report.py`
- Create: `tests/evaluation/test_cv_report.py`

- [ ] **Step 3.1: Write failing tests**

Create `tests/evaluation/test_cv_report.py`:

```python
import numpy as np
import polars as pl
import pytest

from ins_gbm.data.model_data import ModelData
from ins_gbm.data.schema import infer_schema
from ins_gbm.evaluation.cv_report import CrossValidationReport, CVResult
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import ModelRecipe


def _poisson_data(raw: pl.DataFrame) -> ModelData:
    schema = infer_schema(raw, ["x1", "x3"])
    return ModelData(
        features=raw.select(["x1", "x3"]),
        target=raw["claim_count"],
        exposure=raw["exposure"],
        weight=None,
        feature_names=["x1", "x3"],
        schema=schema,
        objective="poisson",
    ).validate()


def test_run_returns_cv_result(poisson_raw):
    data = _poisson_data(poisson_raw)
    report = CrossValidationReport(
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        data=data,
        n_folds=3,
        seed=0,
    )
    result = report.run()
    assert isinstance(result, CVResult)


def test_random_folds_fold_metrics_row_count(poisson_raw):
    data = _poisson_data(poisson_raw)
    result = CrossValidationReport(
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        data=data,
        n_folds=3,
        seed=0,
    ).run()
    # 3 folds × 4 metrics = 12 rows for "gbm"
    gbm_rows = result.fold_metrics.filter(pl.col("model") == "gbm")
    assert gbm_rows.height == 12


def test_summary_has_mean_and_std(poisson_raw):
    data = _poisson_data(poisson_raw)
    result = CrossValidationReport(
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        data=data,
        n_folds=3,
        seed=0,
    ).run()
    gbm_summary = result.summary.filter(pl.col("model") == "gbm")
    assert "mean" in gbm_summary.columns
    assert "std" in gbm_summary.columns
    assert gbm_summary["std"].null_count() == 0


def test_predefined_fold_col_uses_exact_fold_ids(poisson_raw):
    fold_series = pl.Series("fold_id", [i % 3 for i in range(poisson_raw.height)])
    raw = poisson_raw.with_columns(fold_series)
    schema = infer_schema(raw, ["x1", "x3"])
    data = ModelData(
        features=raw.select(["x1", "x3", "fold_id"]),
        target=raw["claim_count"],
        exposure=raw["exposure"],
        weight=None,
        feature_names=["x1", "x3", "fold_id"],
        schema=schema,
        objective="poisson",
    ).validate()
    result = CrossValidationReport(
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        data=data,
        n_folds=99,        # ignored when fold_col is set
        fold_col="fold_id",
        seed=0,
    ).run()
    fold_ids_used = result.fold_metrics["fold"].unique().sort().to_list()
    assert fold_ids_used == [0, 1, 2]
    assert result.fold_col == "fold_id"


def test_fold_col_dropped_before_fitting(poisson_raw):
    """fold_id must not appear as a model feature."""
    fold_series = pl.Series("fold_id", [i % 3 for i in range(poisson_raw.height)])
    raw = poisson_raw.with_columns(fold_series)
    schema = infer_schema(raw, ["x1", "x3"])
    data = ModelData(
        features=raw.select(["x1", "x3", "fold_id"]),
        target=raw["claim_count"],
        exposure=raw["exposure"],
        weight=None,
        feature_names=["x1", "x3", "fold_id"],
        schema=schema,
        objective="poisson",
    ).validate()
    # Should complete without error — LightGBM would error on a non-numeric fold_id
    # if it weren't dropped first
    result = CrossValidationReport(
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        data=data,
        fold_col="fold_id",
        seed=0,
    ).run()
    assert isinstance(result, CVResult)


def test_benchmark_col_adds_benchmark_rows(poisson_raw):
    # Use x1 as a dummy benchmark prediction (positive values after clipping)
    bench = poisson_raw["x1"].abs() + 0.1
    raw = poisson_raw.with_columns(bench.alias("bench_pred"))
    schema = infer_schema(raw, ["x1", "x3"])
    data = ModelData(
        features=raw.select(["x1", "x3", "bench_pred"]),
        target=raw["claim_count"],
        exposure=raw["exposure"],
        weight=None,
        feature_names=["x1", "x3", "bench_pred"],
        schema=schema,
        objective="poisson",
    ).validate()
    result = CrossValidationReport(
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        data=data,
        n_folds=3,
        benchmark_col="bench_pred",
        seed=0,
    ).run()
    models_in_metrics = result.fold_metrics["model"].unique().sort().to_list()
    assert "benchmark" in models_in_metrics
    assert "gbm" in models_in_metrics


def test_n_folds_less_than_2_raises(poisson_raw):
    data = _poisson_data(poisson_raw)
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        CrossValidationReport(
            recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
            data=data,
            n_folds=1,
        ).run()


def test_missing_fold_col_raises(poisson_raw):
    data = _poisson_data(poisson_raw)
    with pytest.raises(ValueError, match="fold_col"):
        CrossValidationReport(
            recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
            data=data,
            fold_col="nonexistent",
        ).run()


def test_fold_col_equals_benchmark_col_raises(poisson_raw):
    fold_series = pl.Series("fold_id", [i % 3 for i in range(poisson_raw.height)])
    raw = poisson_raw.with_columns(fold_series)
    schema = infer_schema(raw, ["x1", "x3"])
    data = ModelData(
        features=raw.select(["x1", "x3", "fold_id"]),
        target=raw["claim_count"],
        exposure=raw["exposure"],
        weight=None,
        feature_names=["x1", "x3", "fold_id"],
        schema=schema,
        objective="poisson",
    ).validate()
    with pytest.raises(ValueError, match="same column"):
        CrossValidationReport(
            recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
            data=data,
            fold_col="fold_id",
            benchmark_col="fold_id",
        ).run()
```

- [ ] **Step 3.2: Run to verify FAIL**

```
pytest tests/evaluation/test_cv_report.py::test_run_returns_cv_result -v
```

Expected: `ImportError: cannot import name 'CrossValidationReport'`

- [ ] **Step 3.3: Create `src/ins_gbm/evaluation/cv_report.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.data.schema import FeatureSchema
from ins_gbm.pipeline import ModelRecipe


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

            current_train = train_data
            current_held = held_data

            if self.recipe.encoder is not None:
                schema = getattr(current_train, "schema", None)
                fitted_enc = self.recipe.encoder.fit(current_train.features, schema)
                current_train = current_train.with_features(
                    fitted_enc.transform(current_train.features)
                )
                current_held = current_held.with_features(
                    fitted_enc.transform(current_held.features)
                )

            if self.recipe.selection is not None:
                fitted_sel = self.recipe.selection.fit(current_train)
                sel_feats = fitted_sel.selected_features()
                current_train = current_train.with_features(
                    current_train.features.select(sel_feats)
                )
                current_held = current_held.with_features(
                    current_held.features.select(sel_feats)
                )

            for prep in self.recipe.preprocessing:
                fitted_prep = prep.fit(current_train.features)
                current_train = current_train.with_features(
                    fitted_prep.transform(current_train.features)
                )
                current_held = current_held.with_features(
                    fitted_prep.transform(current_held.features)
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
                all_fold_rows.append({"fold": fold_id, "model": "gbm", **row})

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
```

- [ ] **Step 3.4: Run all CV report tests**

```
pytest tests/evaluation/test_cv_report.py -v
```

Expected: all nine tests PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/ins_gbm/evaluation/cv_report.py tests/evaluation/test_cv_report.py
git commit -m "feat(evaluation): add CrossValidationReport with random and predefined-fold CV"
```

---

## Task 4: `compare_reports()` + exports

**Files:**
- Create: `src/ins_gbm/evaluation/comparison.py`
- Modify: `src/ins_gbm/evaluation/__init__.py`
- Create: `tests/evaluation/test_comparison.py`

- [ ] **Step 4.1: Write failing tests**

Create `tests/evaluation/test_comparison.py`:

```python
import polars as pl
import pytest

from ins_gbm.data.loader import load_model_data
from ins_gbm.data.model_data import ModelData
from ins_gbm.data.schema import infer_schema
from ins_gbm.data.splitter import TrainTestSplit
from ins_gbm.evaluation.comparison import compare_reports
from ins_gbm.evaluation.cv_report import CrossValidationReport
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import ModelPipeline, ModelRecipe


def _run_pipeline(poisson_parquet, seed: int):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    return ModelPipeline(
        data=data,
        split=TrainTestSplit(train_ratio=0.7, seed=seed),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()


def _run_cv(poisson_raw, seed: int):
    schema = infer_schema(poisson_raw, ["x1", "x3"])
    data = ModelData(
        features=poisson_raw.select(["x1", "x3"]),
        target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"],
        weight=None,
        feature_names=["x1", "x3"],
        schema=schema,
        objective="poisson",
    ).validate()
    return CrossValidationReport(
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        data=data,
        n_folds=3,
        seed=seed,
    ).run()


def test_compare_two_evaluation_reports(poisson_parquet):
    r1 = _run_pipeline(poisson_parquet, seed=0).report
    r2 = _run_pipeline(poisson_parquet, seed=1).report
    df = compare_reports({"model_a": r1, "model_b": r2})
    assert "metric" in df.columns
    assert "model_a" in df.columns
    assert "model_b" in df.columns
    assert "preferred" in df.columns


def test_compare_two_cv_results(poisson_raw):
    r1 = _run_cv(poisson_raw, seed=0)
    r2 = _run_cv(poisson_raw, seed=1)
    df = compare_reports({"cv_a": r1, "cv_b": r2})
    assert set(df.columns) == {"metric", "cv_a", "cv_b", "preferred"}
    # CV values should show +/- notation
    gini_row = df.filter(pl.col("metric") == "gini")
    assert "+/-" in gini_row["cv_a"][0]


def test_compare_mixed_inputs(poisson_parquet, poisson_raw):
    eval_report = _run_pipeline(poisson_parquet, seed=0).report
    cv_result = _run_cv(poisson_raw, seed=0)
    df = compare_reports({"single_test": eval_report, "cv": cv_result})
    assert df.height > 0
    # single_test values should NOT have +/-
    gini_row = df.filter(pl.col("metric") == "gini")
    assert "+/-" not in gini_row["single_test"][0]
    assert "+/-" in gini_row["cv"][0]


def test_preferred_uses_metric_direction(poisson_parquet):
    r1 = _run_pipeline(poisson_parquet, seed=0).report
    r2 = _run_pipeline(poisson_parquet, seed=1).report
    df = compare_reports({"a": r1, "b": r2})
    # For each metric, preferred must be one of the report names or "tie"
    valid = {"a", "b", "tie"}
    for val in df["preferred"].to_list():
        assert val in valid


def test_preferred_is_tie_for_identical_inputs(poisson_parquet):
    r1 = _run_pipeline(poisson_parquet, seed=42).report
    df = compare_reports({"x": r1, "y": r1})
    assert (df["preferred"] == "tie").all()


def test_comparison_mode_report_raises(poisson_parquet):
    from ins_gbm.evaluation.report import EvaluationReport
    r1 = _run_pipeline(poisson_parquet, seed=0)
    r2 = _run_pipeline(poisson_parquet, seed=1)
    comparison_report = EvaluationReport.compare(
        models={
            "a": (r1.fitted_model, r1.train_data, r1.test_data),
            "b": (r2.fitted_model, r2.train_data, r2.test_data),
        },
        test_data=r1.test_data,
    )
    with pytest.raises(ValueError, match="comparison-mode"):
        compare_reports({"bad": comparison_report})


def test_compare_reports_exported_from_evaluation_package():
    from ins_gbm.evaluation import compare_reports as fn
    assert callable(fn)
```

- [ ] **Step 4.2: Run to verify FAIL**

```
pytest tests/evaluation/test_comparison.py::test_compare_two_evaluation_reports -v
```

Expected: `ImportError: cannot import name 'compare_reports'`

- [ ] **Step 4.3: Create `src/ins_gbm/evaluation/comparison.py`**

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Union

import polars as pl

if TYPE_CHECKING:
    from ins_gbm.evaluation.report import EvaluationReport
    from ins_gbm.evaluation.cv_report import CVResult


def compare_reports(
    reports: dict[str, "Union[EvaluationReport, CVResult]"],
) -> pl.DataFrame:
    """Compare two or more EvaluationReport or CVResult objects side by side.

    Returns a DataFrame with one row per metric, one column per report key,
    and a 'preferred' column indicating which report wins on each metric.
    CV values are formatted as 'mean +/- std'; single test-set values as 'mean'.
    """
    from ins_gbm.evaluation.report import EvaluationReport
    from ins_gbm.evaluation.cv_report import CVResult
    from ins_gbm.evaluation.metrics import METRIC_DIRECTIONS

    for name, report in reports.items():
        if isinstance(report, EvaluationReport) and report._comparison_models is not None:
            raise ValueError(
                f"Report {name!r} is a comparison-mode EvaluationReport and cannot be "
                "used with compare_reports(). Pass individual single-model reports or "
                "CVResult objects instead."
            )

    report_data: dict[str, dict[str, tuple[float, float | None]]] = {}
    all_metrics: set[str] = set()

    for name, report in reports.items():
        if isinstance(report, CVResult):
            gbm_rows = report.summary.filter(pl.col("model") == "gbm")
            d: dict[str, tuple[float, float | None]] = {}
            for row in gbm_rows.iter_rows(named=True):
                d[row["metric"]] = (row["mean"], row["std"])
            report_data[name] = d
        else:
            d = {}
            for row in report.metrics().iter_rows(named=True):
                d[row["metric"]] = (row["value"], None)
            report_data[name] = d
        all_metrics.update(report_data[name].keys())

    names = list(reports.keys())
    rows = []

    for metric in sorted(all_metrics):
        row: dict = {"metric": metric}
        values: dict[str, float | None] = {}

        for name in names:
            if metric in report_data[name]:
                mean, std = report_data[name][metric]
                row[name] = f"{mean:.4f} +/- {std:.4f}" if std is not None else f"{mean:.4f}"
                values[name] = mean
            else:
                row[name] = None
                values[name] = None

        direction = METRIC_DIRECTIONS.get(metric, "lower")
        valid = {n: v for n, v in values.items() if v is not None}

        if not valid:
            row["preferred"] = None
        elif len(valid) == 1:
            row["preferred"] = next(iter(valid))
        else:
            best_val = max(valid.values()) if direction == "higher" else min(valid.values())
            best_names = [n for n, v in valid.items() if abs(v - best_val) < 1e-6]
            row["preferred"] = "tie" if len(best_names) > 1 else best_names[0]

        rows.append(row)

    return pl.DataFrame(rows)
```

- [ ] **Step 4.4: Update `src/ins_gbm/evaluation/__init__.py`**

Replace current contents (empty) with:

```python
from ins_gbm.evaluation.metrics import METRIC_DIRECTIONS, compute_metrics
from ins_gbm.evaluation.cv_report import CrossValidationReport, CVResult
from ins_gbm.evaluation.comparison import compare_reports

__all__ = [
    "compute_metrics",
    "METRIC_DIRECTIONS",
    "CrossValidationReport",
    "CVResult",
    "compare_reports",
]
```

- [ ] **Step 4.5: Run all comparison tests**

```
pytest tests/evaluation/test_comparison.py -v
```

Expected: all seven tests PASS.

- [ ] **Step 4.6: Run full test suite**

```
source /c/Users/sssut/anaconda3/etc/profile.d/conda.sh && conda activate base
pytest -v
```

Expected: all tests PASS. Fix any regressions before proceeding.

- [ ] **Step 4.7: Commit**

```bash
git add src/ins_gbm/evaluation/comparison.py src/ins_gbm/evaluation/__init__.py \
        tests/evaluation/test_comparison.py
git commit -m "feat(evaluation): add compare_reports() and export new evaluation symbols"
```
