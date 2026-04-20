# GBM Fitting Tool - Design Specification v2

## Purpose

Build a Python library for fitting policy-level insurance frequency and severity models using gradient boosted machines and related benchmark models. The tool provides a unified, reproducible pipeline for feature preparation, variable selection, dimensionality reduction, hyperparameter tuning, model fitting, ensemble construction, and actuarial evaluation.

The public API is Polars-native and parquet-first. Model libraries may convert to numpy or library-native matrix formats at their boundaries.

## Scope

### In Scope

- Policy-level frequency models.
- Policy-level severity models.
- Separate frequency and severity modeling workflows.
- One-hot encoding for categorical variables in the first version.
- Reproducible random train/test splits.
- Final untouched test-set evaluation.
- Interpretable reports, metrics, feature importance, and SHAP output where supported.

### Out of Scope for v1

- Pure premium modeling as a direct objective.
- Temporal validation.
- Native categorical handling.
- Automated end-to-end model selection without user review.
- Test-set-based blend weight optimization.

## Core Decisions

- **Data library:** Polars at public API boundaries.
- **Input format:** Parquet files.
- **Configuration:** Python API, code-first, no config files.
- **Primary models:** LightGBM, XGBoost, CatBoost.
- **Benchmark model:** Random Forest, with documented objective limitations.
- **Objectives:** Poisson frequency and Gamma severity.
- **Target grain:** Policy level.
- **Frequency target:** Claim count per policy row, with exposure supplied separately.
- **Frequency prediction:** Claim rate over exposure by default, with expected claim count available.
- **Severity target:** Policy-level average severity or policy-level severity amount for positive-loss policies.
- **HP tuning:** Optuna with Bayesian optimization and pruning.
- **Variable selection:** Boruta and importance-based pruning, fit only inside the training data used for a given model fit.
- **Dimensionality reduction:** PCA, PLS, UMAP, fit only inside the training data used for a given model fit.
- **Ensembles:** Stacking and blending using validation or out-of-fold predictions only.
- **Evaluation:** Actuarial metrics and plot export to PNG.
- **Train/test split:** 70/30 default random split, no temporal validation, test set remains untouched until final evaluation.
- **Design priority:** Interpretability and reproducibility over automation.

---

## Architecture

```text
gbm_fitting/
    __init__.py
    data/
        __init__.py
        loader.py
        model_data.py
        splitter.py
        schema.py
    preprocessing/
        __init__.py
        encoder.py
        pca.py
        pls.py
        umap.py
        base.py
    selection/
        __init__.py
        boruta.py
        importance.py
        base.py
    models/
        __init__.py
        base.py
        lightgbm.py
        xgboost.py
        random_forest.py
        catboost.py
    ensemble/
        __init__.py
        stacking.py
        blending.py
        pipeline.py
    tuning/
        __init__.py
        tuner.py
        search_spaces.py
    evaluation/
        __init__.py
        metrics.py
        plots.py
        report.py
    persistence/
        __init__.py
        io.py
        metadata.py
    pipeline.py
```

---

## Component Designs

### 1. Data Layer

**`ModelData`** represents a policy-level modeling table.

```python
@dataclass
class ModelData:
    features: pl.DataFrame
    target: pl.Series
    exposure: pl.Series | None
    weight: pl.Series | None
    feature_names: list[str]
    schema: FeatureSchema | None = None
    objective: Literal["poisson", "gamma"] | None = None

    @classmethod
    def from_parquet(
        cls,
        path: str,
        target: str,
        exposure: str | None = None,
        weight: str | None = None,
        feature_cols: list[str] | None = None,
        schema: "FeatureSchema | None" = None,
        objective: Literal["poisson", "gamma"] | None = None,
    ) -> "ModelData": ...
```

Validation rules:

- All inputs must have the same row count.
- Feature names must be unique.
- `exposure`, when supplied, must be positive and non-null.
- For Poisson frequency:
  - `target` is policy-level claim count.
  - `target` must be non-negative.
  - `exposure` is required.
- For Gamma severity:
  - `target` is policy-level severity.
  - `target` must be strictly positive.
  - `exposure` is not used as an offset.
  - `weight` may be used for claim-count weights, credibility weights, or other user-supplied observation weights.

**`FeatureSchema`** defines feature roles for reproducible preprocessing.

```python
@dataclass
class FeatureSchema:
    numeric: list[str]
    categorical: list[str]
    ordinal: list[str] = field(default_factory=list)
    passthrough: list[str] = field(default_factory=list)
```

When no schema is supplied, the loader infers:

- numeric columns from integer and floating Polars dtypes;
- categorical columns from string, categorical, enum, and boolean dtypes;
- no ordinal columns by default.

**`TrainTestSplit`**:

- Default random 70/30 split.
- Reproducible via random seed.
- Optional stratification by target bins.
- Optional exposure-balanced split for frequency models.
- Optional group split by a policy, account, or risk identifier.
- Returns `(train: ModelData, test: ModelData)`.

Temporal validation is intentionally excluded from v1.

### 2. Feature Preparation

**`OneHotEncoder`** is the default categorical encoder for v1.

```python
class OneHotEncoder:
    def fit(self, features: pl.DataFrame, schema: FeatureSchema) -> "FittedOneHotEncoder": ...

class FittedOneHotEncoder:
    def transform(self, features: pl.DataFrame) -> pl.DataFrame: ...
    def output_feature_names(self) -> list[str]: ...
```

Encoding rules:

- Fit category levels on training data only.
- Unknown categories at prediction time produce all-zero indicator columns for that feature unless configured otherwise.
- Missing categorical values are treated as an explicit missing level by default.
- Numeric columns are passed through unchanged.
- Output column order is stable and stored with the fitted encoder.

One-hot encoding occurs inside each training fold during cross-validation and stacking to avoid leakage.

### 3. Model Wrappers

**`BaseModel` protocol:**

```python
class BaseModel(Protocol):
    objective: Literal["poisson", "gamma"]

    def fit(self, data: ModelData, params: dict | None = None) -> "FittedModel": ...
    def default_search_space(self) -> dict[str, optuna.distributions.BaseDistribution]: ...
    def capabilities(self) -> "ModelCapabilities": ...
```

**`ModelCapabilities`:**

```python
@dataclass(frozen=True)
class ModelCapabilities:
    supports_poisson: bool
    supports_gamma: bool
    supports_offset: bool
    supports_sample_weight: bool
    supports_feature_importance: bool
```

**`FittedModel`:**

```python
@dataclass
class FittedModel:
    model: Any
    params: dict
    framework: str
    objective: Literal["poisson", "gamma"]
    feature_names: list[str]

    def predict(
        self,
        data: ModelData,
        prediction_type: Literal["response", "rate", "link"] = "response",
    ) -> pl.Series: ...

    def feature_importance(self) -> pl.DataFrame: ...
```

Prediction-scale rules:

- For Poisson frequency:
  - model target is policy-level claim count;
  - exposure is modeled as a log offset where supported;
  - `prediction_type="response"` returns expected claim count for each policy row;
  - `prediction_type="rate"` returns expected claim count divided by exposure;
  - `prediction_type="link"` returns the log expected claim count where supported.
- For Gamma severity:
  - model target is policy-level severity;
  - exposure is not used as an offset;
  - `prediction_type="response"` returns expected severity;
  - `prediction_type="rate"` is invalid.

Framework-specific handling:

- **LightGBM:**
  - Poisson uses `objective="poisson"`.
  - Frequency exposure is supplied as `init_score=log(exposure)` at train and prediction time.
  - Gamma uses `objective="gamma"` with positive severity target.
- **XGBoost:**
  - Poisson uses `count:poisson`.
  - Frequency exposure is supplied as `base_margin=log(exposure)` at train and prediction time.
  - Gamma uses `reg:gamma` with positive severity target.
- **CatBoost:**
  - Gamma support is allowed where the installed CatBoost version supports the required loss.
  - Poisson may be supported without a true offset; if no native offset is available, the wrapper must either reject exposure-offset Poisson or use a documented approximation.
  - Sample weights are not treated as equivalent to exposure offsets.
- **Random Forest:**
  - Treated as an interpretable benchmark rather than a fully equivalent GLM-style objective wrapper.
  - May support nonnegative count regression with documented limitations.
  - Does not support Gamma objective or native exposure offsets unless explicitly implemented with a documented target transformation.

All wrappers convert Polars to numpy or framework-native matrix formats at the boundary and return Polars predictions.

### 4. Variable Selection

**`BorutaSelector`**:

- `base_estimator`: `"lightgbm"` by default, `"random_forest"` optionally.
- Creates shadow features by shuffling copies of all candidate features.
- Runs multiple iterations, comparing real feature importance to max shadow importance.
- Returns confirmed, tentative, and rejected feature classifications.
- Configurable: `max_iter`, `alpha`, `seed`.

**`ImportancePruner`**:

- Takes a fitted model and computes permutation importance or uses built-in importance.
- Drops features below a threshold, either absolute or percentile.
- Optionally keeps top-N features.

Leakage rule:

- Selectors are fit only on the training partition available to the current model fit.
- During hyperparameter tuning and stacking, selectors are fit independently inside each fold.
- The final fitted pipeline refits selectors on the full training set after tuning is complete.

### 5. Dimensionality Reduction

**`Preprocessor` protocol:**

```python
class Preprocessor(Protocol):
    def fit(self, features: pl.DataFrame, target: pl.Series | None = None) -> "FittedPreprocessor": ...

class FittedPreprocessor(Protocol):
    def transform(self, features: pl.DataFrame) -> pl.DataFrame: ...
    def component_mapping(self) -> dict[str, list[str]]: ...
```

Implementations:

- **`PCAReducer(n_components)`**: sklearn PCA on standardized numeric or encoded features.
- **`PLSReducer(n_components)`**: sklearn PLSRegression, supervised and therefore leakage-sensitive.
- **`UMAPReducer(n_components, n_neighbors, min_dist)`**: umap-learn, unsupervised by default.

Rules:

- Standardization parameters are learned on training data only.
- Preprocessors are fit independently inside each CV fold during tuning and stacking.
- Component mappings are stored for interpretability.
- PCA/PLS/UMAP are optional and should not be enabled by default for interpretability-first workflows.

### 6. Hyperparameter Tuning

**`HyperparameterTuner`**:

```python
tuner = HyperparameterTuner(
    model=LightGBMModel(objective="poisson"),
    n_trials=100,
    cv_folds=5,
    metric="poisson_deviance",
    pruner=optuna.pruners.MedianPruner(),
    seed=42,
)
best_params, trial_history = tuner.tune(train_data, recipe=recipe)
```

Each trial:

- Samples parameters from the model search space.
- Runs k-fold CV on training data only.
- For each fold, fits the full recipe inside the fold:
  - one-hot encoder;
  - variable selectors;
  - dimensionality reducers;
  - model.
- Evaluates on that fold's validation partition.
- Reports intermediate fold results for pruning.

Returns:

- best params dict;
- trial history as a Polars DataFrame;
- metric direction and selected objective metadata.

Users may override default search spaces with custom ranges.

### 7. Ensemble Methods

**`BlendingEnsemble`**:

- Takes pre-fitted `FittedPipeline` objects.
- Requires validation or out-of-fold predictions for weight fitting.
- Does not optimize blend weights on the final test set.
- Weights are constrained to sum to 1.
- Final prediction is the weighted average on response scale.

Supported blending modes:

- `validation`: user supplies a separate blend dataset.
- `oof`: blend weights are fit from out-of-fold predictions produced on the training data.
- `fixed`: user supplies fixed weights.

**`StackingEnsemble`**:

- Takes fitted pipelines that include cloneable, unfitted pipeline recipes.
- Re-fits each full base pipeline independently inside CV folds to generate out-of-fold predictions.
- Trains a meta-learner on OOF predictions.
- Uses base models fitted on the full training data to generate test predictions.
- Applies the meta-learner to those test predictions.

```python
ensemble = EnsemblePipeline(
    fitted_models=[freq_lgb_result, freq_xgb_result, freq_cat_result],
    method="stacking",
    meta_learner=RidgeRegressor(),
    cv_folds=5,
    seed=42,
)
ensemble_result = ensemble.run()
```

Leakage rule:

- The test set is evaluation-only.
- No selector, preprocessor, blend weight, or meta-learner may be fit using test data.

### 8. Evaluation & Reporting

Metrics are exposure-aware and weight-aware where applicable.

Frequency metrics:

- Poisson deviance on expected claim count.
- Poisson deviance on claim rate, weighted by exposure, where requested.
- RMSE and MAE for expected count or rate.
- Normalized Gini ordered by predicted frequency, weighted by exposure by default.

Severity metrics:

- Gamma deviance on positive severity.
- RMSE and MAE for severity.
- Normalized Gini ordered by predicted severity, weighted by supplied weight where present.

Plots exported to PNG:

- Ordered lift chart.
- Double lift chart.
- Loss ratio by decile when actual loss and premium fields are supplied.
- Actual vs Expected plot.
- Calibration curve.
- SHAP summary plot where supported.
- Feature importance bar chart.

**`EvaluationReport`**:

```python
report = EvaluationReport(
    fitted_model=result.fitted_model,
    test_data=result.test_data,
    train_data=result.train_data,
)
report.metrics()
report.plot_lift()
report.export("output/")
```

**Comparison mode:**

```python
report = EvaluationReport.compare(
    models={"lgb": lgb_result, "xgb": xgb_result, "ensemble": ens_result},
    test_data=test_data,
)
report.export("output/comparison/")
```

Metric formulas and edge-case behavior must be implemented and tested explicitly:

- Zero actual counts are valid for Poisson deviance.
- Zero or negative severity targets are invalid for Gamma deviance.
- Predictions must be positive for Poisson and Gamma deviance.
- Null targets, exposures, weights, or predictions raise validation errors unless an explicit missing-data policy is configured.

### 9. Pipeline Orchestrator

**`ModelRecipe`** is the cloneable, unfitted configuration used for tuning, stacking, and reproducibility.

```python
@dataclass
class ModelRecipe:
    encoder: OneHotEncoder | None
    selection: Selector | None
    preprocessing: list[Preprocessor]
    model: BaseModel
    tuning: HyperparameterTuner | None
```

**`ModelPipeline`**:

```python
pipeline = ModelPipeline(
    data=ModelData.from_parquet(
        "claims.parquet",
        target="claim_count",
        exposure="earned_exposure",
        objective="poisson",
    ),
    split=TrainTestSplit(train_ratio=0.7, seed=42),
    recipe=ModelRecipe(
        encoder=OneHotEncoder(),
        selection=BorutaSelector(base_estimator="lightgbm", seed=42),
        preprocessing=[],
        model=LightGBMModel(objective="poisson"),
        tuning=HyperparameterTuner(n_trials=100, cv_folds=5, seed=42),
    ),
)

result = pipeline.run()
```

**`FittedPipeline`**:

```python
@dataclass
class FittedPipeline:
    fitted_model: FittedModel
    recipe: ModelRecipe
    train_data: ModelData
    test_data: ModelData
    selected_features: list[str] | None
    tuning_history: pl.DataFrame | None
    report: EvaluationReport
    encoder: FittedOneHotEncoder | None
    preprocessor: FittedPreprocessor | None
    metadata: ReproducibilityMetadata
```

Execution order:

1. Split into train and untouched test.
2. Tune on training data only, fitting the full recipe inside each CV fold.
3. Refit encoder, selectors, preprocessors, and model on the full training data using selected params.
4. Evaluate once on the untouched test set.
5. Export metrics, plots, parameters, feature mappings, and reproducibility metadata.

### 10. Persistence and Reproducibility

**`ReproducibilityMetadata`**:

```python
@dataclass
class ReproducibilityMetadata:
    package_versions: dict[str, str]
    random_seeds: dict[str, int]
    model_params: dict
    feature_names: list[str]
    selected_features: list[str] | None
    objective: Literal["poisson", "gamma"]
    prediction_scale: str
```

Persistence requirements:

- Save and load fitted pipelines.
- Save model-native artifacts where possible.
- Save metadata as JSON.
- Save metrics and tuning history as parquet or CSV.
- Save selected feature lists and one-hot feature mappings.

---

## Dependencies

Core:

- **polars**: data handling.
- **scikit-learn**: Random Forest, preprocessing utilities, PCA, PLS, Ridge.
- **optuna**: hyperparameter tuning.
- **matplotlib**: plot generation and export.
- **scipy**: blend weight optimization.

Model extras:

- **lightgbm**: LightGBM models.
- **xgboost**: XGBoost models.
- **catboost**: CatBoost models.

Optional interpretability and dimensionality extras:

- **shap**: SHAP explanations.
- **umap-learn**: UMAP dimensionality reduction.
- **boruta** or custom Boruta implementation.

Recommended package extras:

- `gbm_fitting[lightgbm]`
- `gbm_fitting[xgboost]`
- `gbm_fitting[catboost]`
- `gbm_fitting[explain]`
- `gbm_fitting[umap]`
- `gbm_fitting[all]`

---

## Verification Plan

1. **Data validation tests**
   - Poisson requires positive exposure and non-negative target.
   - Gamma requires strictly positive target.
   - Row counts and feature names are validated.

2. **Feature preparation tests**
   - One-hot encoder stores stable output columns.
   - Unknown categories transform deterministically.
   - Missing categorical values are handled as configured.

3. **Model wrapper tests**
   - LightGBM and XGBoost use log exposure at train and prediction time for Poisson.
   - Gamma wrappers reject non-positive targets.
   - Random Forest exposes documented capability limitations.
   - `predict(..., prediction_type="response")` and `predict(..., prediction_type="rate")` behave correctly for Poisson.

4. **Leakage tests**
   - CV folds fit encoders, selectors, and preprocessors using fold-training data only.
   - PLS cannot see validation targets during CV fitting.
   - Stacking OOF predictions are produced from fold-fitted full recipes.
   - Blending raises an error if asked to optimize weights on test data.

5. **Metric correctness tests**
   - Compare Poisson deviance, Gamma deviance, RMSE, MAE, and normalized Gini against known manual calculations.
   - Validate exposure and weight handling.

6. **Integration test**
   - End-to-end frequency pipeline on a small synthetic policy-level dataset.
   - End-to-end severity pipeline on a small positive-severity policy-level dataset.
   - Verify parquet load, split, tune, fit, evaluate, and report export.

7. **Ensemble test**
   - Fit two or more base pipelines independently.
   - Verify stacking uses OOF predictions.
   - Verify blending supports fixed or validation/OOF weights.
   - Verify ensemble test evaluation does not fit on test data.

8. **Persistence test**
   - Save and load a fitted pipeline.
   - Verify predictions are identical after reload.
   - Verify metadata, selected features, and feature mappings are restored.

9. **Plot smoke test**
   - Verify all configured plot exports create valid PNG files without errors.
