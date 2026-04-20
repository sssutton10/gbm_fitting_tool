# GBM Fitting Tool Implementation Plan - Complete

> This plan implements `docs/superpowers/specs/2026-04-20-gbm-fitting-tool-design-v2.md`.
> It is intentionally test-first and task-by-task. Use checkbox (`- [ ]`) syntax for tracking.

## Goal

Build a Polars-native Python library for policy-level insurance frequency and severity modeling.

The library supports:

- Poisson frequency models with policy-level claim count target and exposure offset.
- Gamma severity models with positive policy-level severity target.
- One-hot encoding for v1 categorical handling.
- Reproducible train/test splits.
- Leakage-safe cross-validation, preprocessing, selection, tuning, stacking, and blending.
- Actuarial metrics, plots, reports, persistence, and reproducibility metadata.

## Design Contracts

- Public APIs accept and return Polars objects where practical.
- Internal utilities may use numpy, scipy, sklearn, or model-native matrices.
- The final test set is untouched until final evaluation.
- Encoders, selectors, preprocessors, blend weights, and meta-learners are never fit on final test data.
- During CV, every fold fits its own encoder, selector, preprocessor, and model.
- Frequency predictions support expected claim count and claim rate.
- Severity predictions return expected severity; rate prediction is invalid for Gamma severity.
- Interpretability and reproducibility take priority over automation.

---

## Task 1: Project Setup

**Files:**

- Create: `pyproject.toml`
- Create: `src/gbm_fitting/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`
- Create: `README.md`

- [ ] Create package metadata with core dependencies:
  - `polars`
  - `numpy`
  - `scipy`
  - `scikit-learn`
  - `optuna`
  - `matplotlib`
  - `pyarrow`

- [ ] Add optional extras:
  - `lightgbm`
  - `xgboost`
  - `catboost`
  - `explain`
  - `umap`
  - `all`
  - `dev`

- [ ] Create synthetic test fixtures:
  - policy-level Poisson frequency data with `claim_count` and positive `exposure`;
  - policy-level Gamma severity data with strictly positive `severity` and optional `weight`;
  - parquet fixtures for both.

- [ ] In the frequency fixture, use clear naming:
  - `log_rate = linear_predictor`
  - `expected_count = exposure * exp(log_rate)`
  - `claim_count = poisson(expected_count)`

- [ ] Create README with install and test commands.

- [ ] Verify:

```bash
pip install -e ".[all,dev]"
pytest --collect-only
```

- [ ] Commit:

```bash
git add pyproject.toml src/ tests/ .gitignore README.md
git commit -m "chore: project scaffolding and fixtures"
```

---

## Task 2: FeatureSchema and ModelData

**Files:**

- Create: `src/gbm_fitting/data/__init__.py`
- Create: `src/gbm_fitting/data/schema.py`
- Create: `src/gbm_fitting/data/model_data.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_schema.py`
- Create: `tests/data/test_model_data.py`

- [ ] Write tests for `FeatureSchema` defaults and `infer_schema`.

- [ ] Implement `FeatureSchema`:
  - `numeric`
  - `categorical`
  - `ordinal`
  - `passthrough`
  - `all_features()`

- [ ] Implement `infer_schema(df, feature_cols)`.

- [ ] Write tests for valid Poisson `ModelData`.

- [ ] Write validation tests:
  - row count mismatch;
  - duplicate feature names;
  - missing feature columns;
  - target nulls;
  - weight nulls;
  - exposure nulls;
  - non-positive exposure;
  - Poisson missing exposure;
  - Poisson negative target;
  - Gamma zero target;
  - Gamma negative target.

- [ ] Implement `ModelData`:
  - fields from v2 spec;
  - `n_rows`;
  - `validate()`;
  - `with_features()`;
  - optional `with_rows(mask)` helper for splitting.

- [ ] Decide and document whether dataclass construction validates automatically:
  - preferred: explicit `.validate()` for cheap object construction and clear loader behavior;
  - tests should call `.validate()` when validation is expected.

- [ ] Verify:

```bash
pytest tests/data/test_schema.py tests/data/test_model_data.py -v
```

- [ ] Commit:

```bash
git add src/gbm_fitting/data/ tests/data/
git commit -m "feat(data): add FeatureSchema and ModelData validation"
```

---

## Task 3: Parquet Loader and Train/Test Splitter

**Files:**

- Create: `src/gbm_fitting/data/loader.py`
- Create: `src/gbm_fitting/data/splitter.py`
- Create: `tests/data/test_loader.py`
- Create: `tests/data/test_splitter.py`

- [ ] Implement `load_model_data(...)`.

- [ ] Loader tests:
  - loads Poisson parquet;
  - loads Gamma parquet;
  - infers feature columns when omitted;
  - infers schema when omitted;
  - preserves supplied schema;
  - validates before returning.

- [ ] Implement `TrainTestSplit`.

- [ ] Splitter tests:
  - default 70/30 split;
  - reproducible random split;
  - invalid train ratio;
  - group split;
  - optional stratification by binned target;
  - optional exposure-balanced split for Poisson frequency.

- [ ] Support group splitting by either:
  - a feature column; or
  - a separate group `pl.Series` supplied to `.split(...)`.

- [ ] Verify:

```bash
pytest tests/data/ -v
```

- [ ] Commit:

```bash
git add src/gbm_fitting/data/ tests/data/
git commit -m "feat(data): add parquet loader and train/test splitter"
```

---

## Task 4: One-Hot Encoder

**Files:**

- Create: `src/gbm_fitting/preprocessing/__init__.py`
- Create: `src/gbm_fitting/preprocessing/encoder.py`
- Create: `tests/preprocessing/__init__.py`
- Create: `tests/preprocessing/test_encoder.py`

- [ ] Write tests for `OneHotEncoder`.

- [ ] Required behavior:
  - fit category levels on training data only;
  - stable output column order;
  - numeric columns pass through unchanged;
  - missing categorical values become an explicit missing level by default;
  - unknown prediction-time categories produce all-zero indicators for that feature;
  - `output_feature_names()` returns encoded feature names;
  - transformed output contains no raw categorical columns.

- [ ] Implement:
  - `OneHotEncoder.fit(features, schema)`;
  - `FittedOneHotEncoder.transform(features)`;
  - `FittedOneHotEncoder.output_feature_names()`;
  - serializable fitted category mapping.

- [ ] Verify:

```bash
pytest tests/preprocessing/test_encoder.py -v
```

- [ ] Commit:

```bash
git add src/gbm_fitting/preprocessing/ tests/preprocessing/
git commit -m "feat(preprocessing): add one-hot encoder"
```

---

## Task 5: Model Base Types and Capability Contracts

**Files:**

- Create: `src/gbm_fitting/models/__init__.py`
- Create: `src/gbm_fitting/models/base.py`
- Create: `tests/models/__init__.py`
- Create: `tests/models/test_base.py`

- [ ] Write tests for `ModelCapabilities`.

- [ ] Implement:
  - `Objective = Literal["poisson", "gamma"]`;
  - `PredictionType = Literal["response", "rate", "link"]`;
  - `ModelCapabilities`;
  - `BaseModel` protocol;
  - `FittedModel` protocol or base dataclass;
  - common validation helpers for objective, feature names, exposure, and prediction type.

- [ ] Required behavior:
  - Poisson `rate` predictions require exposure;
  - Gamma `rate` predictions raise `ValueError`;
  - prediction output length must match input row count;
  - feature columns are aligned to training feature order.

- [ ] Verify:

```bash
pytest tests/models/test_base.py -v
```

- [ ] Commit:

```bash
git add src/gbm_fitting/models/ tests/models/
git commit -m "feat(models): add base contracts and capabilities"
```

---

## Task 6: Actuarial Metrics

**Files:**

- Create: `src/gbm_fitting/evaluation/__init__.py`
- Create: `src/gbm_fitting/evaluation/metrics.py`
- Create: `tests/evaluation/__init__.py`
- Create: `tests/evaluation/test_metrics.py`

- [ ] Write manual-calculation tests for:
  - Poisson deviance;
  - Gamma deviance;
  - RMSE;
  - MAE;
  - weighted RMSE;
  - weighted MAE;
  - normalized Gini.

- [ ] Implement explicit formula behavior:
  - zero actual counts are valid for Poisson deviance;
  - Poisson predictions must be positive;
  - Gamma actuals and predictions must be positive;
  - null values raise errors;
  - weights and exposure must be non-negative where used;
  - all metric inputs must have equal length.

- [ ] Implement:
  - `poisson_deviance(actual, predicted, weight=None)`;
  - `gamma_deviance(actual, predicted, weight=None)`;
  - `rmse(actual, predicted, weight=None)`;
  - `mae(actual, predicted, weight=None)`;
  - `normalized_gini(actual, predicted, weight=None)`;
  - `metric_table(...)`.

- [ ] Verify:

```bash
pytest tests/evaluation/test_metrics.py -v
```

- [ ] Commit:

```bash
git add src/gbm_fitting/evaluation/ tests/evaluation/
git commit -m "feat(evaluation): add actuarial metrics"
```

---

## Task 7: LightGBM Model Wrapper

**Files:**

- Create: `src/gbm_fitting/models/lightgbm.py`
- Create: `tests/models/test_lightgbm.py`

- [ ] Write tests guarded with `pytest.importorskip("lightgbm")`.

- [ ] Poisson tests:
  - fit encoded policy-level claim count data;
  - use `log(exposure)` as offset at train time;
  - use `log(exposure)` as offset at prediction time;
  - `response` returns expected claim count;
  - `rate` returns expected claim count divided by exposure;
  - predictions are positive.

- [ ] Gamma tests:
  - fit positive policy-level severity;
  - reject non-positive targets via `ModelData.validate()`;
  - predictions are positive;
  - `rate` prediction raises.

- [ ] Implement `LightGBMModel`:
  - `objective`;
  - `fit(data, params=None)`;
  - `default_search_space()`;
  - `capabilities()`;
  - fitted model wrapper with `predict` and `feature_importance`.

- [ ] Verify:

```bash
pytest tests/models/test_lightgbm.py -v
```

- [ ] Commit:

```bash
git add src/gbm_fitting/models/lightgbm.py tests/models/test_lightgbm.py
git commit -m "feat(models): add LightGBM wrapper"
```

---

## Task 8: XGBoost Model Wrapper

**Files:**

- Create: `src/gbm_fitting/models/xgboost.py`
- Create: `tests/models/test_xgboost.py`

- [ ] Write tests guarded with `pytest.importorskip("xgboost")`.

- [ ] Poisson tests:
  - use `count:poisson`;
  - use `base_margin=log(exposure)` at train and prediction time;
  - support response and rate predictions.

- [ ] Gamma tests:
  - use `reg:gamma`;
  - reject invalid severity targets;
  - produce positive predictions.

- [ ] Implement `XGBoostModel`.

- [ ] Verify:

```bash
pytest tests/models/test_xgboost.py -v
```

- [ ] Commit:

```bash
git add src/gbm_fitting/models/xgboost.py tests/models/test_xgboost.py
git commit -m "feat(models): add XGBoost wrapper"
```

---

## Task 9: CatBoost and Random Forest Wrappers

**Files:**

- Create: `src/gbm_fitting/models/catboost.py`
- Create: `src/gbm_fitting/models/random_forest.py`
- Create: `tests/models/test_catboost.py`
- Create: `tests/models/test_random_forest.py`

- [ ] CatBoost tests guarded with `pytest.importorskip("catboost")`.

- [ ] CatBoost requirements:
  - expose accurate capabilities for installed support;
  - do not treat sample weights as exposure offsets;
  - if no true Poisson offset support is available, raise a clear `NotImplementedError` for exposure-offset Poisson;
  - support Gamma only when the installed version supports the required loss.

- [ ] Random Forest requirements:
  - expose capabilities as benchmark model;
  - document lack of native offset;
  - do not silently claim full Poisson/Gamma equivalence;
  - support basic positive regression benchmark where explicitly allowed.

- [ ] Verify:

```bash
pytest tests/models/test_catboost.py tests/models/test_random_forest.py -v
```

- [ ] Commit:

```bash
git add src/gbm_fitting/models/catboost.py src/gbm_fitting/models/random_forest.py tests/models/
git commit -m "feat(models): add CatBoost and Random Forest wrappers"
```

---

## Task 10: Preprocessor Protocols and Reducers

**Files:**

- Create: `src/gbm_fitting/preprocessing/base.py`
- Create: `src/gbm_fitting/preprocessing/pca.py`
- Create: `src/gbm_fitting/preprocessing/pls.py`
- Create: `src/gbm_fitting/preprocessing/umap.py`
- Create: `tests/preprocessing/test_reducers.py`

- [ ] Implement `Preprocessor` and `FittedPreprocessor` protocols.

- [ ] PCA tests:
  - standardizes using training data only;
  - transforms train and new data to stable component columns;
  - exposes component mapping.

- [ ] PLS tests:
  - requires target at fit time;
  - transforms without target;
  - exposes component mapping.

- [ ] UMAP tests:
  - guarded with `pytest.importorskip("umap")`;
  - deterministic with seed where possible;
  - transforms to stable component columns.

- [ ] Ensure reducers are optional and not enabled by default.

- [ ] Verify:

```bash
pytest tests/preprocessing/ -v
```

- [ ] Commit:

```bash
git add src/gbm_fitting/preprocessing/ tests/preprocessing/
git commit -m "feat(preprocessing): add optional dimensionality reducers"
```

---

## Task 11: Variable Selection

**Files:**

- Create: `src/gbm_fitting/selection/__init__.py`
- Create: `src/gbm_fitting/selection/base.py`
- Create: `src/gbm_fitting/selection/importance.py`
- Create: `src/gbm_fitting/selection/boruta.py`
- Create: `tests/selection/__init__.py`
- Create: `tests/selection/test_importance.py`
- Create: `tests/selection/test_boruta.py`

- [ ] Implement selector protocols:
  - `fit(data)`;
  - `transform(data)`;
  - `selected_features`;
  - selection summary table.

- [ ] Implement `ImportancePruner`:
  - built-in importance mode;
  - permutation importance mode;
  - threshold and top-N behavior.

- [ ] Implement `BorutaSelector`:
  - LightGBM or Random Forest base estimator;
  - shadow features;
  - confirmed, tentative, rejected classifications;
  - seed support.

- [ ] Tests:
  - selected features are subset of input features;
  - transform preserves target, exposure, and weight;
  - deterministic with seed;
  - no test-set fitting behavior is exposed.

- [ ] Verify:

```bash
pytest tests/selection/ -v
```

- [ ] Commit:

```bash
git add src/gbm_fitting/selection/ tests/selection/
git commit -m "feat(selection): add feature selectors"
```

---

## Task 12: Recipe Fitting Utilities

**Files:**

- Create: `src/gbm_fitting/pipeline.py`
- Create: `tests/test_recipe.py`

- [ ] Implement `ModelRecipe`.

- [ ] Implement internal fit helper for one full recipe:
  - clone or copy recipe configuration;
  - fit encoder on supplied training data;
  - fit selector on encoded training data;
  - fit preprocessors on fold-training data only;
  - fit model;
  - return fitted artifacts and transformed data.

- [ ] Tests:
  - full recipe can fit with encoder only;
  - selected feature list is recorded when selector is present;
  - preprocessor artifacts are recorded when present;
  - original recipe remains reusable.

- [ ] Verify:

```bash
pytest tests/test_recipe.py -v
```

- [ ] Commit:

```bash
git add src/gbm_fitting/pipeline.py tests/test_recipe.py
git commit -m "feat(pipeline): add ModelRecipe and recipe fitting"
```

---

## Task 13: Hyperparameter Tuning

**Files:**

- Create: `src/gbm_fitting/tuning/__init__.py`
- Create: `src/gbm_fitting/tuning/search_spaces.py`
- Create: `src/gbm_fitting/tuning/tuner.py`
- Create: `tests/tuning/__init__.py`
- Create: `tests/tuning/test_tuner.py`

- [ ] Implement default search spaces for supported model wrappers.

- [ ] Implement `HyperparameterTuner`.

- [ ] Required behavior:
  - CV uses training data only;
  - each fold fits the full recipe independently;
  - fold metrics are reported to Optuna for pruning;
  - test set is never accepted or referenced by tuner;
  - returns best params and trial history as Polars DataFrame;
  - deterministic with seed where practical.

- [ ] Tests:
  - small synthetic Poisson tune run;
  - small synthetic Gamma tune run;
  - trial history schema;
  - custom search space override;
  - fake encoder/preprocessor counters prove fold-local fitting.

- [ ] Verify:

```bash
pytest tests/tuning/ -v
```

- [ ] Commit:

```bash
git add src/gbm_fitting/tuning/ tests/tuning/
git commit -m "feat(tuning): add leakage-safe Optuna tuner"
```

---

## Task 14: ModelPipeline and FittedPipeline

**Files:**

- Update: `src/gbm_fitting/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] Implement:
  - `ModelPipeline`;
  - `FittedPipeline`;
  - final train/test split handling;
  - tuning on train only;
  - final refit on full train only;
  - final evaluation on untouched test only.

- [ ] `FittedPipeline` contains:
  - `fitted_model`;
  - `recipe`;
  - `train_data`;
  - `test_data`;
  - `selected_features`;
  - `tuning_history`;
  - `report`;
  - `encoder`;
  - `preprocessor`;
  - `metadata`.

- [ ] Tests:
  - end-to-end Poisson pipeline with encoder and LightGBM or test stub model;
  - end-to-end Gamma pipeline;
  - no tuning path;
  - tuning path;
  - test data row count preserved;
  - report attached.

- [ ] Verify:

```bash
pytest tests/test_pipeline.py -v
```

- [ ] Commit:

```bash
git add src/gbm_fitting/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): add end-to-end model pipeline"
```

---

## Task 15: Evaluation Reports and Plots

**Files:**

- Create: `src/gbm_fitting/evaluation/plots.py`
- Create: `src/gbm_fitting/evaluation/report.py`
- Create: `tests/evaluation/test_plots.py`
- Create: `tests/evaluation/test_report.py`

- [ ] Implement plots:
  - ordered lift chart;
  - double lift chart;
  - Actual vs Expected plot;
  - calibration curve;
  - feature importance bar chart;
  - SHAP summary where supported and installed;
  - loss ratio by decile only when actual loss and premium fields are supplied.

- [ ] Implement `EvaluationReport`:
  - `metrics()`;
  - plot methods;
  - `export(output_dir)`;
  - comparison mode.

- [ ] Tests:
  - metrics table returns Polars DataFrame;
  - plots return matplotlib figures;
  - export writes metrics and PNG files;
  - comparison report creates side-by-side metrics;
  - SHAP gracefully skips or raises clear optional dependency error.

- [ ] Verify:

```bash
pytest tests/evaluation/ -v
```

- [ ] Commit:

```bash
git add src/gbm_fitting/evaluation/ tests/evaluation/
git commit -m "feat(evaluation): add reports and plot exports"
```

---

## Task 16: Ensembles

**Files:**

- Create: `src/gbm_fitting/ensemble/__init__.py`
- Create: `src/gbm_fitting/ensemble/blending.py`
- Create: `src/gbm_fitting/ensemble/stacking.py`
- Create: `src/gbm_fitting/ensemble/pipeline.py`
- Create: `tests/ensemble/__init__.py`
- Create: `tests/ensemble/test_blending.py`
- Create: `tests/ensemble/test_stacking.py`
- Create: `tests/ensemble/test_pipeline.py`

- [ ] Implement `BlendingEnsemble`.

- [ ] Blending modes:
  - `fixed`;
  - `validation`;
  - `oof`.

- [ ] Blending must reject final test data as weight-fitting data unless the user explicitly marks it as non-test validation data.

- [ ] Implement `StackingEnsemble`.

- [ ] Stacking requirements:
  - base pipelines must contain cloneable recipes;
  - each base recipe is refit inside each stacking fold;
  - OOF predictions train the meta-learner;
  - full-training base models produce final predictions;
  - meta-learner combines final predictions.

- [ ] Implement `EnsemblePipeline`.

- [ ] Tests:
  - fixed blend weights;
  - validation blend weights;
  - test-set blend fitting rejection;
  - stacking OOF shape;
  - stacking does not reuse full-train preprocessors inside OOF generation;
  - ensemble predictions are positive for Poisson and Gamma.

- [ ] Verify:

```bash
pytest tests/ensemble/ -v
```

- [ ] Commit:

```bash
git add src/gbm_fitting/ensemble/ tests/ensemble/
git commit -m "feat(ensemble): add leakage-safe blending and stacking"
```

---

## Task 17: Persistence and Reproducibility Metadata

**Files:**

- Create: `src/gbm_fitting/persistence/__init__.py`
- Create: `src/gbm_fitting/persistence/metadata.py`
- Create: `src/gbm_fitting/persistence/io.py`
- Create: `tests/persistence/__init__.py`
- Create: `tests/persistence/test_io.py`
- Create: `tests/persistence/test_metadata.py`

- [ ] Implement `ReproducibilityMetadata`.

- [ ] Metadata includes:
  - package versions;
  - random seeds;
  - model params;
  - feature names;
  - selected features;
  - objective;
  - prediction scale;
  - schema;
  - encoder mapping.

- [ ] Implement save/load:
  - save fitted pipeline;
  - save model-native artifacts where practical;
  - save metadata JSON;
  - save metrics CSV or parquet;
  - save tuning history;
  - save selected feature lists;
  - save one-hot feature mappings.

- [ ] Tests:
  - save/load round trip;
  - predictions identical after reload within tolerance;
  - metadata restored;
  - missing optional model dependency raises clear error on load.

- [ ] Verify:

```bash
pytest tests/persistence/ -v
```

- [ ] Commit:

```bash
git add src/gbm_fitting/persistence/ tests/persistence/
git commit -m "feat(persistence): add pipeline save and load"
```

---

## Task 18: Integration Tests

**Files:**

- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_frequency_pipeline.py`
- Create: `tests/integration/test_severity_pipeline.py`
- Create: `tests/integration/test_ensemble_pipeline.py`
- Create: `tests/integration/test_export_roundtrip.py`

- [ ] Frequency integration test:
  - parquet load;
  - random split;
  - one-hot encode;
  - tune small number of trials;
  - fit final model;
  - evaluate on untouched test;
  - export report.

- [ ] Severity integration test:
  - parquet load;
  - one-hot encode;
  - fit Gamma severity model;
  - evaluate;
  - export report.

- [ ] Ensemble integration test:
  - fit two base pipelines;
  - stack using OOF predictions;
  - blend with fixed or OOF weights;
  - evaluate ensemble on untouched test.

- [ ] Persistence integration test:
  - save fitted pipeline;
  - load fitted pipeline;
  - compare predictions.

- [ ] Mark slow or optional-dependency-heavy tests with pytest markers.

- [ ] Verify:

```bash
pytest tests/integration/ -v
```

- [ ] Commit:

```bash
git add tests/integration/
git commit -m "test: add end-to-end integration coverage"
```

---

## Task 19: Documentation and Examples

**Files:**

- Update: `README.md`
- Create: `examples/frequency_lightgbm.py`
- Create: `examples/severity_lightgbm.py`
- Create: `examples/compare_models.py`
- Create: `examples/ensemble.py`
- Create: `docs/modeling_contracts.md`

- [ ] Document modeling contracts:
  - frequency target;
  - exposure offset;
  - count vs rate prediction;
  - severity target requirements;
  - one-hot encoding behavior;
  - untouched test-set rule;
  - leakage-safe CV rule.

- [ ] Add examples:
  - frequency model;
  - severity model;
  - model comparison;
  - ensemble.

- [ ] Add optional dependency install examples.

- [ ] Add reproducibility notes.

- [ ] Verify examples run or are import-checked.

- [ ] Commit:

```bash
git add README.md examples/ docs/
git commit -m "docs: add modeling contracts and examples"
```

---

## Task 20: Quality Gate

**Files:**

- Update: `pyproject.toml`
- Optional create: `.github/workflows/tests.yml`

- [ ] Add or configure:
  - pytest markers for slow and optional dependency tests;
  - coverage command;
  - lint/type tools if desired.

- [ ] Run full local verification:

```bash
pytest
```

- [ ] Run optional full verification if all extras are installed:

```bash
pytest -m "not slow"
pytest tests/integration/ -v
```

- [ ] Review public API exports in `src/gbm_fitting/__init__.py`.

- [ ] Confirm no final-test-set leakage paths remain:
  - tuner;
  - selectors;
  - preprocessors;
  - blending;
  - stacking;
  - reports.

- [ ] Confirm prediction-scale behavior:
  - Poisson response is expected count;
  - Poisson rate is expected count divided by exposure;
  - Gamma response is expected severity;
  - Gamma rate raises.

- [ ] Commit:

```bash
git add .
git commit -m "chore: final quality gate for v1"
```

---

## Definition of Done

- [ ] All core tests pass.
- [ ] Optional dependency tests skip cleanly when extras are unavailable.
- [ ] End-to-end frequency and severity pipelines run from parquet to report export.
- [ ] Poisson exposure offset is applied at train and prediction time for supported wrappers.
- [ ] Gamma severity rejects non-positive targets.
- [ ] One-hot encoding is stable and reproducible.
- [ ] Tuning and stacking fit full recipes inside each fold.
- [ ] Blending does not optimize on final test data.
- [ ] Metrics have formula-level tests.
- [ ] Reports export metrics and PNG plots.
- [ ] Fitted pipelines can be saved and loaded.
- [ ] README and examples explain the modeling contracts clearly.
