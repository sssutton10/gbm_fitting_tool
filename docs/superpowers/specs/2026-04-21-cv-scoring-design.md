# Design: CV Stability Report, Benchmark Comparison & Pipeline Scoring

**Date:** 2026-04-21  
**Status:** Approved

---

## Overview

Three related additions to `ins_gbm`:

1. **`FittedPipeline` scoring** — `.predict()` and `.predict_raw()` methods so a loaded pipeline can score new data without manual transform chaining.
2. **`CrossValidationReport`** — k-fold (or predefined-fold) CV that produces mean ± std of each metric, with optional benchmark comparison against an existing model's predictions stored as a column.
3. **`compare_reports()`** — standalone function to place two or more `EvaluationReport` or `CVResult` objects side-by-side with a `preferred` column.

---

## Section 1 — FittedPipeline Scoring

### Problem

`FittedPipeline` stores `encoder`, `selected_features`, `preprocessors`, and `fitted_model` but exposes no method to apply that chain to new data. Users must manually wire the transforms, which is error-prone and breaks the deployment contract.

### Changes

**`src/ins_gbm/pipeline.py`** — two new methods on `FittedPipeline`:

```python
def predict(self, data: ModelData) -> pl.Series
def predict_raw(
    self,
    features: pl.DataFrame,
    exposure: pl.Series | None = None,
    weight: pl.Series | None = None,
) -> pl.Series
```

**`predict(model_data)`** chains:
1. `fitted_encoder.transform(data.features)` — if encoder present
2. `data.features.select(selected_features)` — if selection was run
3. Each `fitted_preprocessor.transform(features)` in order
4. `fitted_model.predict(transformed_data, prediction_type="response")`

`predict_raw(features, exposure, weight)` constructs a `ModelData` from the raw inputs using the pipeline's stored `objective` and `schema`, then delegates to `predict()`.

**Bug fix — `src/ins_gbm/persistence/io.py`**: `load_pipeline` currently uses `pickle.load`; switch to `cloudpickle.load` to match the `cloudpickle.dump` used in `save_pipeline`. Without this fix, loading pipelines with closure-based `predict_fn` / `importance_fn` will fail.

### Invariants

- `predict()` must apply transforms in the same order as `ModelPipeline.run()`: encode → select → preprocess → model.
- `predict_raw()` must not re-fit any transform; it only applies the already-fitted objects stored on `FittedPipeline`.
- `selected_features` may be `None` (no selection step); in that case the selection step is skipped.

---

## Section 2 — CrossValidationReport

### New file: `src/ins_gbm/evaluation/cv_report.py`

```python
@dataclass
class CrossValidationReport:
    recipe: ModelRecipe
    data: ModelData
    n_folds: int = 5
    benchmark_col: str | None = None
    fold_col: str | None = None
    seed: int = 42

    def run(self) -> CVResult: ...

@dataclass
class CVResult:
    fold_metrics: pl.DataFrame   # columns: fold, model, metric, value
    summary: pl.DataFrame        # columns: model, metric, mean, std
    fold_col: str | None         # None = random folds were used
```

### Fold strategy

- **`fold_col` provided**: read unique values from `data.features[fold_col]`, treat each as one held-out fold. `n_folds` is ignored. The column is dropped from features before any fitting (both train and held-out folds).
- **`fold_col` is `None`**: random k-fold, `n_folds` folds, seeded by `seed`.

The `fold` column in `fold_metrics` contains the actual fold ID values (integers or strings depending on the source column), making per-fold inspection meaningful for predefined folds (e.g. policy year).

### Per-fold fitting

For each fold, the full recipe is refit on the train rows with leakage guardrails matching `ModelPipeline.run()`:

1. Encoder fit on train rows only, transform both train and held-out.
2. Selector (if present) fit on encoded train rows only.
3. Each preprocessor fit on selected train rows only.
4. Model fit on fully-transformed train rows.
5. Evaluate on held-out rows.

If `recipe.tuning` is set, tuning is skipped inside CV folds (too expensive and not the purpose of a stability report). The recipe's model params are used as-is.

### Benchmark comparison

If `benchmark_col` is set:
- The column is removed from `data.features` before fitting (like `fold_col`).
- On each held-out fold, the benchmark predictions are read from this column for those rows.
- The same metrics computed for the GBM are computed for the benchmark.
- Both appear in `fold_metrics` under `model = "gbm"` and `model = "benchmark"`.
- `summary` has one row per (model, metric) pair.

### Metrics computed

Same set as `EvaluationReport._single_metrics()`: objective deviance (Poisson or Gamma), Gini, RMSE, MAE. Metric direction is recorded internally for use by `compare_reports()`.

---

## Section 3 — compare_reports()

### Location

`src/ins_gbm/evaluation/comparison.py`, exported from `ins_gbm.evaluation`.

### Signature

```python
def compare_reports(
    reports: dict[str, EvaluationReport | CVResult],
) -> pl.DataFrame
```

### Output schema

| column | type | description |
|---|---|---|
| `metric` | Utf8 | metric name |
| `<name>` | Utf8 | one column per report key; value is `"0.412 ± 0.018"` (CV) or `"0.412"` (single test-set) |
| `preferred` | Utf8 | key of the report with the best value; `"tie"` if equal within 1e-6 |

### Preferred logic

Metric direction is fixed:
- Higher is better: `gini`
- Lower is better: `poisson_deviance`, `gamma_deviance`, `rmse`, `mae`

For `CVResult` inputs, comparison uses the **mean** value. Std is shown in the display string but does not affect the `preferred` decision.

For `EvaluationReport` inputs (single test-set), no std — display string is the point estimate only.

Mixed inputs (one `CVResult`, one `EvaluationReport`) are valid; each is rendered in its natural format.

---

## Files Changed

| File | Change |
|---|---|
| `src/ins_gbm/pipeline.py` | Add `FittedPipeline.predict()` and `predict_raw()` |
| `src/ins_gbm/persistence/io.py` | Fix `load_pipeline` to use `cloudpickle.load` |
| `src/ins_gbm/evaluation/cv_report.py` | New file: `CrossValidationReport`, `CVResult` |
| `src/ins_gbm/evaluation/comparison.py` | New file: `compare_reports()` |
| `src/ins_gbm/evaluation/__init__.py` | Export new symbols |
| `tests/evaluation/test_cv_report.py` | New test file |
| `tests/evaluation/test_comparison.py` | New test file |
| `tests/test_pipeline.py` | Tests for `predict()` and `predict_raw()` |
| `tests/persistence/test_persistence.py` | Test that `load_pipeline` roundtrips correctly |

---

## Out of Scope

- Tuning inside CV folds (too expensive; existing `HyperparameterTuner` handles its own internal CV).
- Parallelising CV folds (can be added later with `joblib`).
- Learning curves (different feature request).
- SHAP / interpretability (different feature request).
