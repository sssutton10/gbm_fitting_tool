# Design: CV Stability Report, Benchmark Comparison & Pipeline Scoring

**Date:** 2026-04-21
**Status:** Reviewed copy with suggested corrections
**Source:** `2026-04-21-cv-scoring-design.md`

---

## Review Summary

This copy keeps the original design intent and makes the following corrections:

1. Clarifies that `FittedPipeline.predict()` and `predict_raw()` should accept a `prediction_type` argument, matching `FittedModel.predict()`.
2. Defines how `predict_raw()` constructs `ModelData` when scoring data does not have a target.
3. Fixes transform-chain wording so feature selection is applied to the currently transformed features, not the original raw frame.
4. Adds validation and row-alignment rules for `fold_col` and `benchmark_col`.
5. Defines metric computation as shared helper behavior instead of depending on `EvaluationReport._single_metrics()` directly.
6. Clarifies how `compare_reports()` handles unsupported comparison-mode `EvaluationReport` inputs.
7. Adds focused implementation notes for `feature_names`, `schema`, and test coverage.

---

## Overview

Three related additions to `ins_gbm`:

1. **`FittedPipeline` scoring**: `.predict()` and `.predict_raw()` methods so a loaded pipeline can score new data without manual transform chaining.
2. **`CrossValidationReport`**: k-fold or predefined-fold CV that produces mean and standard deviation for each metric, with optional benchmark comparison against existing predictions stored as a feature column.
3. **`compare_reports()`**: standalone function to place two or more single-model `EvaluationReport` or `CVResult` objects side by side with a `preferred` column.

---

## Section 1: FittedPipeline Scoring

### Problem

`FittedPipeline` stores `encoder`, `selected_features`, `preprocessors`, and `fitted_model`, but exposes no method to apply that chain to new data. Users must manually wire the transforms, which is error-prone and breaks the deployment contract.

### Changes

**`src/ins_gbm/pipeline.py`**: add two methods on `FittedPipeline`:

```python
def predict(
    self,
    data: ModelData,
    prediction_type: PredictionType = "response",
) -> pl.Series

def predict_raw(
    self,
    features: pl.DataFrame,
    exposure: pl.Series | None = None,
    weight: pl.Series | None = None,
    prediction_type: PredictionType = "response",
) -> pl.Series
```

`PredictionType` should be imported from `ins_gbm.models.base`.

### `predict(data)` behavior

`predict(data)` applies the fitted transform chain, preserving row alignment and metadata:

1. Start from the supplied `ModelData`.
2. If `self.encoder` is present, call `self.encoder.transform(current.features)` and replace features with `current.with_features(...)`.
3. If `self.selected_features` is not `None`, select those columns from `current.features`, not from the original raw input frame.
4. For each fitted preprocessor in `self.preprocessors`, call `fitted_preprocessor.transform(current.features)` and replace features with `current.with_features(...)`.
5. Call `self.fitted_model.predict(current, prediction_type=prediction_type)`.

The method must not mutate the supplied `ModelData` or any stored fitted objects.

### `predict_raw(features, exposure, weight)` behavior

`predict_raw()` is intended for deployment-style scoring where a target is not available. Because `ModelData.target` is currently required, `predict_raw()` should construct a temporary target that is valid for the pipeline objective and never used for prediction:

- Use `self.fitted_model.objective` as the objective.
- Use `self.train_data.schema` as the schema when available. The current `ModelData.with_features()` implementation preserves `schema`, so this should still be the raw schema after training.
- Use `feature_names=list(features.columns)`.
- For Poisson, create a zero-valued placeholder target with length `features.height`.
- For Gamma, create a one-valued placeholder target with length `features.height` so it satisfies Gamma positivity if validation is called later.
- Do not call `validate()` inside `predict_raw()` unless exposure/target validation semantics are intentionally updated for scoring without a real target.

Then delegate to `predict()`.

### Validation

`predict()` and `predict_raw()` should fail early with clear `ValueError`s when:

- A required raw column for the fitted encoder is missing.
- A selected feature is missing after encoding.
- A preprocessor or model-required feature is missing after transformation.
- `exposure` or `weight`, when provided, has a different length from `features`.

For Poisson models, `prediction_type="response"` should continue to follow existing model wrapper semantics: with exposure, return expected count; without exposure, return the framework's no-offset response, typically a rate-like value. This behavior should be documented in the method docstring.

### Bug fix: persistence load

**`src/ins_gbm/persistence/io.py`**: `load_pipeline` currently uses `pickle.load`; switch to `cloudpickle.load` to match the `cloudpickle.dump` used in `save_pipeline`. Without this fix, loading pipelines with closure-based `predict_fn` or `importance_fn` can fail.

### Invariants

- `predict()` must apply transforms in the same order as `ModelPipeline.run()`: encode, select, preprocess, model.
- `predict_raw()` must not refit any transform; it only applies the already fitted objects stored on `FittedPipeline`.
- `selected_features` may be `None`; in that case the selection step is skipped.
- The final `ModelData.feature_names` must match the columns passed to `fitted_model.predict()`.

---

## Section 2: CrossValidationReport

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

### Input validation

`CrossValidationReport.run()` should validate before fitting:

- `n_folds >= 2` when `fold_col is None`.
- `n_folds <= data.n_rows` when using random folds.
- `fold_col`, when provided, exists in `data.features`.
- `benchmark_col`, when provided, exists in `data.features`.
- `fold_col` and `benchmark_col` are not the same column.
- Predefined folds contain at least two distinct non-null fold values.
- Benchmark predictions contain no nulls and are positive for Poisson or Gamma deviance metrics.

### Fold strategy

- **`fold_col` provided**: read unique non-null values from `data.features[fold_col]`, treating each as one held-out fold. `n_folds` is ignored. Drop the column from features before any fitting, for both train and held-out folds.
- **`fold_col is None`**: create random k-fold splits with `n_folds`, seeded by `seed`.

The `fold` column in `fold_metrics` contains the actual fold ID values for predefined folds, making per-fold inspection meaningful for groups such as policy year. For random folds, it contains integer fold IDs from `0` to `n_folds - 1`.

### Per-fold data preparation

For each fold:

1. Build train and held-out `ModelData` objects by filtering rows.
2. Before fitting, remove `fold_col` and `benchmark_col` from both feature frames when present.
3. After dropping columns, update `feature_names` to match the remaining feature frame columns.
4. Preserve row order so held-out benchmark predictions remain aligned with held-out targets, exposure, and weight.

If the raw `schema` includes `fold_col` or `benchmark_col`, either remove those columns from a copy of the schema before encoder fitting or require the caller's schema not to include them. Preferred behavior: remove them from a local schema copy so benchmark/fold columns can safely live in `data.features`.

### Per-fold fitting

For each fold, the full recipe is refit on the train rows with leakage guardrails matching `ModelPipeline.run()`:

1. Encoder fit on train rows only, then transform both train and held-out.
2. Selector, if present, fit on encoded train rows only.
3. Each preprocessor fit on selected train rows only.
4. Model fit on fully transformed train rows.
5. Evaluate on held-out rows.

If `recipe.tuning` is set, tuning is skipped inside CV folds. The recipe's model params are used as-is. This is intentional because the report measures stability of the configured recipe and avoids nested-CV cost.

### Benchmark comparison

If `benchmark_col` is set:

- Read benchmark predictions from the original feature frame before dropping the column.
- Keep benchmark predictions aligned to each held-out fold by applying the same fold mask or row indices used for held-out data.
- Compute the same metrics for the GBM and benchmark.
- Emit both under `model = "gbm"` and `model = "benchmark"` in `fold_metrics`.
- Emit one row per `(model, metric)` pair in `summary`.

### Metrics computed

Compute the same metric set as `EvaluationReport._single_metrics()`:

- `poisson_deviance` for Poisson objectives, or `gamma_deviance` for Gamma objectives
- `gini`
- `rmse`
- `mae`

Implementation suggestion: extract a shared helper, for example:

```python
def compute_metrics(
    *,
    objective: Objective,
    actual: pl.Series,
    predicted: pl.Series,
    exposure: pl.Series | None = None,
    weight: pl.Series | None = None,
) -> pl.DataFrame: ...
```

Then use it from both `EvaluationReport._single_metrics()` and `CrossValidationReport`. This avoids fabricating temporary `FittedModel` objects and reduces the chance that CV metrics drift from test-set metrics.

Metric direction should be represented by a small shared constant, not hidden inside `CVResult`:

```python
METRIC_DIRECTIONS = {
    "gini": "higher",
    "poisson_deviance": "lower",
    "gamma_deviance": "lower",
    "rmse": "lower",
    "mae": "lower",
}
```

### Summary aggregation

`summary` groups by `model` and `metric`:

- `mean`: arithmetic mean of per-fold metric values.
- `std`: sample standard deviation of per-fold metric values.

Because validation requires at least two folds, `std` should be numeric rather than null.

---

## Section 3: `compare_reports()`

### Location

`src/ins_gbm/evaluation/comparison.py`, exported from `ins_gbm.evaluation`.

### Signature

```python
def compare_reports(
    reports: dict[str, EvaluationReport | CVResult],
) -> pl.DataFrame
```

### Supported inputs

- A `CVResult` from `CrossValidationReport.run()`.
- A single-model `EvaluationReport`.

Existing comparison-mode `EvaluationReport` objects, where `_comparison_models is not None`, should raise `ValueError` with a clear message. This avoids ambiguous nesting such as one report key expanding into multiple internal model names.

### Output schema

| column | type | description |
|---|---|---|
| `metric` | Utf8 | metric name |
| `<name>` | Utf8 | one column per report key; value is `"0.412 +/- 0.018"` for CV or `"0.412"` for a single test-set report |
| `preferred` | Utf8 | key of the report with the best value; `"tie"` if equal within `1e-6` |

### Preferred logic

Metric direction is fixed:

- Higher is better: `gini`
- Lower is better: `poisson_deviance`, `gamma_deviance`, `rmse`, `mae`

For `CVResult` inputs, comparison uses the mean value for the GBM rows by default. Benchmark rows inside a `CVResult` should not become separate report candidates unless the caller passes them as separate named results in a future API.

For `EvaluationReport` inputs, comparison uses the point estimate returned by `report.metrics()`.

Mixed inputs are valid; each is rendered in its natural format.

If a metric is missing from some reports because objectives differ, include the metric row and render missing cells as `None`; choose `preferred` only among reports that contain that metric.

---

## Files Changed

| File | Change |
|---|---|
| `src/ins_gbm/pipeline.py` | Add `FittedPipeline.predict()` and `predict_raw()` |
| `src/ins_gbm/persistence/io.py` | Fix `load_pipeline` to use `cloudpickle.load` |
| `src/ins_gbm/evaluation/metrics.py` or new helper module | Add shared metric-table helper and metric direction constant |
| `src/ins_gbm/evaluation/report.py` | Use shared metric helper in `EvaluationReport._single_metrics()` |
| `src/ins_gbm/evaluation/cv_report.py` | New file: `CrossValidationReport`, `CVResult` |
| `src/ins_gbm/evaluation/comparison.py` | New file: `compare_reports()` |
| `src/ins_gbm/evaluation/__init__.py` | Export new symbols |
| `tests/evaluation/test_cv_report.py` | New test file |
| `tests/evaluation/test_comparison.py` | New test file |
| `tests/test_pipeline.py` | Tests for `predict()` and `predict_raw()` |
| `tests/persistence/test_persistence.py` | Test that `load_pipeline` roundtrips correctly |

---

## Test Coverage

### Pipeline scoring

- `FittedPipeline.predict(test_like_model_data)` matches direct `fitted_model.predict()` on the pipeline's transformed test data.
- `predict_raw(raw_features, exposure=...)` matches `predict()` on an equivalent raw `ModelData`.
- Selection is applied after encoding by testing a pipeline with both encoder and selector.
- Multiple preprocessors are applied in order.
- Loaded pipelines can score after `save_pipeline()` and `load_pipeline()`.
- Missing required columns raise clear `ValueError`s.

### CV report

- Random k-fold produces `n_folds * metric_count` rows for the GBM.
- Predefined fold column uses exact fold IDs and ignores `n_folds`.
- `fold_col` and `benchmark_col` are dropped before fitting but remain available for fold assignment and benchmark evaluation.
- Benchmark rows are aligned to held-out rows.
- Encoder, selector, and preprocessors are fit only on train folds.
- `summary` contains mean and std by `(model, metric)`.
- Invalid fold/benchmark columns raise clear errors.

### Comparison

- Compares two `EvaluationReport` instances.
- Compares two `CVResult` instances.
- Compares mixed `EvaluationReport` and `CVResult` inputs.
- Handles metric direction correctly.
- Emits `"tie"` within tolerance.
- Rejects comparison-mode `EvaluationReport` inputs.
- Handles objective-specific missing metrics without crashing.

---

## Out of Scope

- Tuning inside CV folds. Existing `HyperparameterTuner` handles its own internal CV, and nested tuning would make this report slow and harder to interpret.
- Parallelizing CV folds. This can be added later with `joblib`.
- Learning curves.
- SHAP or interpretability.
- Changing `ModelData.target` to optional. That is a broader API change and should be considered separately if deployment scoring becomes a larger first-class workflow.
