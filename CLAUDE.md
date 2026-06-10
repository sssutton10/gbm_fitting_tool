# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

**Always activate conda before any Python/pip/pytest command:**

```bash
source /c/Users/sssut/anaconda3/etc/profile.d/conda.sh && conda activate base
```

> **Warning:** A separate `gbm_fitting` package (the "GBM Fitting Codex") is already installed in this environment. This library is named `ins_gbm` to avoid collision. Never rename it back to `gbm_fitting`.

## Commands

```bash
# Install (all model extras + dev)
pip install -e ".[all,dev]"

# Run all tests
pytest

# Run a single test
pytest tests/models/test_lightgbm.py::test_lgb_poisson_fit_predict -v

# Run a test module
pytest tests/test_pipeline.py -v
```

## Architecture

Layered library. Public API is Polars-native; numpy conversion happens only at the model boundary (inside each model's `fit()` and `predict()`).

```
src/ins_gbm/
  data/           — ModelData, FeatureSchema, loader, splitter
  preprocessing/  — OneHotEncoder, PCAReducer, PLSReducer, UMAPReducer
  selection/      — BorutaSelector, ImportancePruner
  models/         — base Protocol + LightGBM, XGBoost, CatBoost, RandomForest wrappers
  tuning/         — HyperparameterTuner (Optuna + MedianPruner)
  evaluation/     — metrics, plots, EvaluationReport
  pipeline.py     — ModelRecipe, ModelPipeline, FittedPipeline
  ensemble/       — BlendingEnsemble, StackingEnsemble, EnsemblePipeline
  persistence/    — ReproducibilityMetadata, save_pipeline, load_pipeline
```

## Key Type Contracts

- **`ModelData`** — central data container (`features`, `target`, `exposure`, `weight`, `feature_names`, `schema`, `objective`). Call `.validate()` after construction. `.with_features(df)` returns a copy with new features + updated feature_names.
- **`FittedModel`** — wraps a trained model. Fields `predict_fn` and `importance_fn` are callables (NOT `_predict_fn` — leading underscore breaks dataclass `__init__`). `.predict(data, prediction_type)` where `prediction_type ∈ {"response", "rate", "link"}`.
- **`ModelRecipe`** — unfitted config (`model`, `encoder`, `selection`, `preprocessing`, `tuning`). Cloneable; used by tuner and stacking for CV re-fits.
- **`FittedPipeline`** — result of `ModelPipeline.run()`. `train_data` and `test_data` are **transformed** (post-encoder/selector/preprocessor), ready for the model.

## Missing Value Convention

The library expects missing values to arrive **pre-filled** before `OneHotEncoder.fit()` is called — upstream data preparation is responsible for this, not the library.

| Column type | Expected sentinel | How the library handles it |
|---|---|---|
| Numeric / ordinal | `-999999999.0` | Passed through by `OneHotEncoder`; converted back to `NaN` before LightGBM/CatBoost training so those frameworks use their native missing-value branch logic; declared as `missing=` in XGBoost `DMatrix` for the same effect |
| Categorical | `"-999999999"` | Treated as an explicit level during `OneHotEncoder.fit()`; gets its own indicator column like any other category |

`_NUMERIC_FILL = -999_999_999.0` and `_MISSING_LEVEL = "-999999999"` are the authoritative constants in `src/ins_gbm/preprocessing/encoder.py`. Import them in model wrappers rather than hard-coding the value.

Random Forest (sklearn) has no native missing-value support, so it receives the filled sentinel as a real number — this is a known limitation documented in `RandomForestModel`.

## Objectives & Prediction

- `"poisson"`: frequency. Requires positive `exposure`. LightGBM uses `init_score=log(exposure)`, XGBoost uses `base_margin=log(exposure)`. `prediction_type="rate"` is invalid for gamma.
- `"gamma"`: severity. Target must be strictly positive. CatBoost Gamma uses `Tweedie:variance_power=1.99` (CatBoost rejects exactly 2.0).

## Leakage Guardrails

- Encoder, selector, and preprocessor must be fit only on the **training fold** inside CV loops — never on the full dataset before splitting.
- `PLSReducer` is supervised (requires target at fit time); never let it see validation target during CV.
- Blend weights and stacking meta-learner are fit only on training/OOF data; test set is evaluation-only.
- `ModelPipeline.run()` enforces this order: split → tune (CV only on train) → refit full train → evaluate once on test.

## Persistence

Uses `cloudpickle` (not standard `joblib`) to serialize `FittedPipeline` because `predict_fn` and `importance_fn` are locally-defined closures that `pickle` cannot handle.

```python
from ins_gbm.persistence.io import save_pipeline, load_pipeline
save_pipeline(result, "output/my_model")
loaded = load_pipeline("output/my_model")
```

## Test Fixtures

`tests/conftest.py` provides:
- `poisson_raw` / `poisson_parquet` — 400 rows, features `x1` (float), `x2` (str A/B/C), `x3` (float), `exposure`, `claim_count`
- `gamma_raw` / `gamma_parquet` — 300 rows, features `x1` (float), `x2` (str A/B), `severity`, `weight`

Boruta and ImportancePruner tests use only `["x1", "x3"]` (numeric). In the full pipeline, OHE encoding precedes variable selection.
