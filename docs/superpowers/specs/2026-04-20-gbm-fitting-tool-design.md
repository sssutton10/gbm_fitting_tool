# GBM Fitting Tool — Design Specification

## Purpose

Build a Python library for fitting insurance frequency and severity models using gradient boosted machines. The tool provides a unified pipeline for variable selection, dimensionality reduction, hyperparameter tuning, model fitting, ensemble construction, and actuarial evaluation — all Polars-native with parquet input.

## Core Decisions

- **Data library:** Polars (numpy only at model boundaries)
- **Input format:** Parquet files
- **Configuration:** Python API (code-first, no config files)
- **Models:** LightGBM, XGBoost, Random Forest, CatBoost
- **Objectives:** Poisson (frequency), Gamma (severity)
- **HP tuning:** Optuna with Bayesian optimization and pruning
- **Variable selection:** Boruta (LightGBM or Random Forest base), importance-based pruning
- **Dimensionality reduction:** PCA, PLS, UMAP
- **Ensembles:** Stacking and blending, accepting pre-fitted pipelines
- **Evaluation:** Full actuarial suite with plot export (PNG)
- **Train/test split:** 70/30 default, no holdout

---

## Architecture

```
gbm_fitting/
    __init__.py
    data/
        __init__.py
        loader.py          # Parquet loading into Polars
        model_data.py      # ModelData dataclass
        splitter.py        # Train/test splitting
    preprocessing/
        __init__.py
        pca.py             # PCA reducer
        pls.py             # PLS reducer
        umap.py            # UMAP reducer
        base.py            # Preprocessor protocol
    selection/
        __init__.py
        boruta.py          # Boruta variable selection
        importance.py      # Importance-based pruning
        base.py            # Selector protocol
    models/
        __init__.py
        base.py            # BaseModel protocol + FittedModel
        lightgbm.py        # LightGBM wrapper
        xgboost.py         # XGBoost wrapper
        random_forest.py   # Random Forest wrapper
        catboost.py        # CatBoost wrapper
    ensemble/
        __init__.py
        stacking.py        # Stacking ensemble
        blending.py        # Blending (weighted average) ensemble
        pipeline.py        # EnsemblePipeline (accepts pre-fitted models)
    tuning/
        __init__.py
        tuner.py           # Optuna hyperparameter tuner
        search_spaces.py   # Default search spaces per model
    evaluation/
        __init__.py
        metrics.py         # Deviance, Gini, RMSE, MAE
        plots.py           # Lift, double-lift, AvE, calibration, SHAP, importance
        report.py          # EvaluationReport class with export
    pipeline.py            # ModelPipeline orchestrator
```

---

## Component Designs

### 1. Data Layer

**`ModelData`** (dataclass):
```python
@dataclass
class ModelData:
    features: pl.DataFrame       # predictor columns
    target: pl.Series            # response variable
    exposure: pl.Series | None   # exposure (earned premium, policy years)
    weight: pl.Series | None     # observation weights
    feature_names: list[str]     # tracked through selection/preprocessing

    @classmethod
    def from_parquet(cls, path: str, target: str,
                     exposure: str | None = None,
                     weight: str | None = None,
                     feature_cols: list[str] | None = None) -> "ModelData": ...
```

**`TrainTestSplit`**:
- Default 70/30 split
- Stratified option for imbalanced targets
- Returns `(train: ModelData, test: ModelData)` tuple
- Reproducible via random seed

### 2. Model Wrappers

**`BaseModel` protocol:**
```python
class BaseModel(Protocol):
    objective: str  # "poisson" or "gamma"

    def fit(self, data: ModelData, params: dict | None = None) -> "FittedModel": ...
    def default_search_space(self) -> dict[str, optuna.distributions.BaseDistribution]: ...
```

**`FittedModel`** (returned by fit):
```python
@dataclass
class FittedModel:
    model: Any                    # underlying framework model
    predict: Callable[[pl.DataFrame], pl.Series]
    feature_importance: Callable[[], pl.DataFrame]  # cols: feature, importance
    params: dict                  # parameters used
    framework: str                # "lightgbm", "xgboost", "random_forest", "catboost"
```

**Framework-specific handling:**
- **LightGBM:** Exposure as `init_score` (log of exposure), `Dataset` API
- **XGBoost:** Exposure as `base_margin` (log of exposure), `DMatrix` API
- **CatBoost:** Exposure via `sample_weight` or offset parameter, `Pool` API
- **Random Forest:** sklearn `RandomForestRegressor`, exposure incorporated via sample weights (no native offset support)

All wrappers convert Polars to numpy at the boundary and return Polars predictions.

### 3. Variable Selection

**`BorutaSelector`**:
- `base_estimator`: `"lightgbm"` (default) or `"random_forest"`
- Creates shadow features (shuffled copies of all features)
- Runs multiple iterations, comparing real feature importance to max shadow importance
- Returns confirmed/tentative/rejected feature classifications
- Configurable: `max_iter`, `alpha` (significance level)

**`ImportancePruner`**:
- Takes a fitted model, computes permutation importance or uses built-in importance
- Drops features below a threshold (absolute or percentile)
- Option to keep top-N features

**Chaining:** Boruta first to get candidate set, then importance pruning for further refinement.

### 4. Dimensionality Reduction

**`Preprocessor` protocol:**
```python
class Preprocessor(Protocol):
    def fit(self, features: pl.DataFrame, target: pl.Series | None = None) -> "FittedPreprocessor": ...

class FittedPreprocessor(Protocol):
    def transform(self, features: pl.DataFrame) -> pl.DataFrame: ...
    def component_mapping(self) -> dict[str, list[str]]: ...  # component -> original features
```

**Implementations:**
- **`PCAReducer(n_components)`** — sklearn PCA on standardized numeric features
- **`PLSReducer(n_components)`** — sklearn PLSRegression, supervised (uses target)
- **`UMAPReducer(n_components, n_neighbors, min_dist)`** — umap-learn, unsupervised by default

All standardize numeric inputs before transformation.

### 5. Hyperparameter Tuning

**`HyperparameterTuner`**:
```python
tuner = HyperparameterTuner(
    model=LightGBMModel(objective="poisson"),
    n_trials=100,
    cv_folds=5,
    metric="poisson_deviance",  # objective for optimization
    pruner=optuna.pruners.MedianPruner(),
    seed=42,
)
best_params = tuner.tune(train_data)
```

- Each trial: sample params from search space, run k-fold CV on training data, return mean metric
- Pruning: stop unpromising trials early based on intermediate fold results
- Returns: best params dict + trial history as Polars DataFrame
- User can override the default search space with custom ranges

### 6. Ensemble Methods

**`BlendingEnsemble`**:
- Takes pre-fitted `FittedPipeline` objects
- Optimizes blend weights by minimizing deviance on test predictions
- Weights constrained to sum to 1 (scipy `minimize` with constraints)
- Final prediction: weighted average

**`StackingEnsemble`**:
- Takes pre-fitted `FittedPipeline` objects (which carry the original model config + training data)
- Re-fits each base model on CV folds of the training data to generate out-of-fold predictions
- Trains a meta-learner (Ridge, or another GBM) on OOF predictions
- Test predictions: uses the original pre-fitted base models (trained on full training set) to predict on test, meta-learner combines

**`EnsemblePipeline`**:
```python
ensemble = EnsemblePipeline(
    fitted_models=[freq_lgb_result, freq_xgb_result, freq_cat_result],
    method="stacking",      # or "blending"
    meta_learner=RidgeRegressor(),  # for stacking
    cv_folds=5,             # for generating OOF predictions in stacking
)
ensemble_result = ensemble.run()
```

The key design: ensembles consume `FittedPipeline` results, not raw models. This means each base model can have its own variable selection, preprocessing, and tuning — the ensemble combines their final predictions.

### 7. Evaluation & Reporting

**Metrics** (all exposure/weight-aware):
- Poisson deviance
- Gamma deviance
- Gini coefficient (normalized)
- RMSE, MAE

**Plots** (exported to PNG via matplotlib):
- Ordered lift chart (actual vs predicted by decile)
- Double lift chart (model A vs model B)
- Loss ratio by decile
- Actual vs Expected (AvE) plot
- Calibration curve
- SHAP summary plot (beeswarm)
- Feature importance bar chart

**`EvaluationReport`**:
```python
report = EvaluationReport(
    fitted_model=result.fitted_model,
    test_data=result.test_data,
    train_data=result.train_data,  # for SHAP
)
report.metrics()          # returns pl.DataFrame with all metrics
report.plot_lift()        # generates and returns figure
report.export("output/")  # saves all metrics CSVs + plot PNGs
```

**Comparison mode:**
```python
report = EvaluationReport.compare(
    models={"lgb": lgb_result, "xgb": xgb_result, "ensemble": ens_result},
    test_data=test_data,
)
report.export("output/comparison/")  # side-by-side metrics + double-lift charts
```

### 8. Pipeline Orchestrator

**`ModelPipeline`**:
```python
pipeline = ModelPipeline(
    data=ModelData.from_parquet("claims.parquet", target="claim_count", exposure="earned_premium"),
    split=TrainTestSplit(train_ratio=0.7, seed=42),
    selection=BorutaSelector(base_estimator="lightgbm"),       # optional
    preprocessing=[PCAReducer(n_components=10)],               # optional, list
    model=LightGBMModel(objective="poisson"),
    tuning=HyperparameterTuner(n_trials=100, cv_folds=5),     # optional
)

result = pipeline.run()
# result: FittedPipeline with .fitted_model, .train_data, .test_data,
#         .selected_features, .tuning_history, .report
```

**`FittedPipeline`** (returned by `pipeline.run()`):
```python
@dataclass
class FittedPipeline:
    fitted_model: FittedModel
    model_config: BaseModel           # unfitted model + params (for stacking re-fit)
    train_data: ModelData
    test_data: ModelData
    selected_features: list[str] | None
    tuning_history: pl.DataFrame | None
    report: EvaluationReport
    preprocessor: FittedPreprocessor | None
```

**Execution order:** split -> select variables -> preprocess -> tune -> fit -> evaluate

---

## Dependencies

- **polars** — data handling
- **lightgbm** — LightGBM models
- **xgboost** — XGBoost models
- **catboost** — CatBoost models
- **scikit-learn** — Random Forest, PCA, PLS, Ridge (meta-learner)
- **optuna** — hyperparameter tuning
- **umap-learn** — UMAP dimensionality reduction
- **shap** — SHAP explanations
- **matplotlib** — plot generation and export
- **scipy** — blend weight optimization
- **boruta** (or custom implementation) — Boruta variable selection

---

## Verification Plan

1. **Unit tests per component:** Each module (loader, splitter, model wrappers, selectors, preprocessors, tuner, ensembles, metrics, plots) gets its own test file with synthetic data
2. **Integration test:** End-to-end pipeline run with a small synthetic insurance dataset — verify the full chain from parquet load through report export
3. **Ensemble test:** Fit 2+ models independently, pass FittedPipelines to EnsemblePipeline, verify stacking and blending produce valid predictions
4. **Metric correctness:** Compare deviance and Gini calculations against known values from a reference implementation (e.g., manual calculation on a small dataset)
5. **Plot smoke test:** Verify plot export creates valid PNG files without errors
