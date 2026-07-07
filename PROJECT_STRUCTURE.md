# GBM Fitting Project Structure and Main Flows

This document describes the `GBM Fitting` project in this directory. It covers
the `ins_gbm` package only.

The project is a Python library for policy-level insurance modeling. It is
designed around Polars data structures at the public API boundary, parquet input
files, leakage-conscious modeling workflows, and wrappers around common machine
learning libraries used for frequency and severity models.

The library is not a command line application. Most usage happens through Python
imports and object construction.

## Quick Orientation

At a high level, the tool does this:

1. Load a policy-level parquet file into a `ModelData` object.
2. Validate basic target, exposure, weight, offset, fold, and comparison fields.
3. Split data into train and test sets.
4. Optionally tune model hyperparameters with cross-validation on training data.
5. Fit feature preparation steps on training data only.
6. Fit a model on the transformed training data.
7. Transform the test data with the fitted feature preparation artifacts.
8. Evaluate the fitted model once on the transformed test data.
9. Optionally export reports, persist the fitted pipeline, or combine fitted
   pipelines with blending or stacking.

The main package is:

```text
src/ins_gbm/
```

The package name is `ins_gbm`, not `gbm_fitting`.

The root README is intentionally minimal. The deeper implementation details are
in the source tree, tests, and this document.

## Top-Level Repository Layout

```text
GBM Fitting/
    README.md
    CLAUDE.md
    pyproject.toml
    uv.lock
    PROJECT_STRUCTURE.md
    examples/
        example_usage.ipynb
    docs/
        superpowers/
            specs/
            plans/
    src/
        ins_gbm/
    tests/
```

Important root files:

- `pyproject.toml`: package metadata, dependencies, optional extras, pytest
  configuration.
- `README.md`: install and test commands.
- `CLAUDE.md`: local development notes, including a warning that this package is
  named `ins_gbm` to avoid collisions with another package.
- `uv.lock`: lockfile for the local environment.
- `examples/example_usage.ipynb`: notebook-style usage example.
- `docs/superpowers/`: design specs and implementation plans. These are useful
  for intent, but the source code is the authority for current behavior.

## Installation and Test Commands

From the `GBM Fitting` directory:

```bash
pip install -e ".[all,dev]"
pytest
```

Optional extras declared in `pyproject.toml`:

- `lightgbm`: installs LightGBM support.
- `xgboost`: installs XGBoost support.
- `catboost`: installs CatBoost support.
- `explain`: installs SHAP.
- `umap`: installs UMAP.
- `all`: installs all optional modeling and explainability dependencies.
- `dev`: installs pytest-related development dependencies.

Common local pitfall: the path contains a space. Quote paths in shell commands
when needed.

## Package Map

```text
src/ins_gbm/
    __init__.py
    progress.py
    pipeline.py
    data/
        __init__.py
        loader.py
        model_data.py
        schema.py
        splitter.py
    preprocessing/
        __init__.py
        encoder.py
        pca.py
        pls.py
        umap.py
    selection/
        __init__.py
        boruta.py
        importance.py
    models/
        __init__.py
        base.py
        lightgbm.py
        xgboost.py
        catboost.py
        random_forest.py
    tuning/
        __init__.py
        tuner.py
        search_spaces.py
    evaluation/
        __init__.py
        metrics.py
        plots.py
        report.py
        cv_report.py
        comparison.py
    ensemble/
        __init__.py
        _utils.py
        blending.py
        stacking.py
        pipeline.py
    persistence/
        __init__.py
        io.py
        metadata.py
```

Many package `__init__.py` files are empty or expose only a small subset of the
implementation. Prefer explicit module imports such as:

```python
from ins_gbm.data.loader import load_model_data
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import ModelPipeline, ModelRecipe
```

Do not assume that `from ins_gbm.models import LightGBMModel` works.

## Main Concepts

### Objective Types

The library supports two modeling objectives:

- `poisson`: policy-level frequency modeling. The target is claim count and
  `exposure` is required.
- `gamma`: policy-level severity modeling. The target must be strictly positive.
  `weight` may be supplied, but exposure is not used as a severity offset.

Prediction scales use the `PredictionType` literal from `models/base.py`:

- `response`: expected claim count for Poisson, expected severity for Gamma.
- `rate`: expected claim count divided by exposure for Poisson. Invalid for
  Gamma and raises in `FittedModel.predict()`.
- `link`: model-specific link scale where implemented.

### Polars Public API

The public data containers use Polars:

- feature matrices are `pl.DataFrame`
- target, exposure, weight, offset, and fold columns are `pl.Series`
- metrics and feature importance outputs are usually `pl.DataFrame`

Model wrappers convert to NumPy or framework-native matrix types at model
boundaries.

## Data Layer

The data layer lives in `src/ins_gbm/data/`.

### `FeatureSchema`

Defined in `data/schema.py`.

```python
FeatureSchema(
    numeric=["x1", "x3"],
    categorical=["territory"],
    ordinal=[],
    passthrough=[],
)
```

Fields:

- `numeric`: continuous or integer predictors.
- `categorical`: unordered predictors that should be one-hot encoded.
- `ordinal`: ordered numeric-like predictors that should pass through the
  one-hot encoder with numeric missing handling.
- `passthrough`: columns copied as-is by the encoder.

`FeatureSchema.all_features()` returns all lists concatenated in this order:

```text
numeric, categorical, ordinal, passthrough
```

`infer_schema(df, feature_cols)` classifies feature columns from Polars dtypes:

- numeric: integer and floating dtypes.
- categorical: string, categorical, enum, and boolean dtypes.

If a dtype is unsupported, `infer_schema()` raises and the caller should supply
an explicit schema.

### `ModelData`

Defined in `data/model_data.py`.

`ModelData` is the central data container passed through almost every layer.

```python
ModelData(
    features=pl.DataFrame(...),
    target=pl.Series(...),
    exposure=pl.Series(...) or None,
    weight=pl.Series(...) or None,
    feature_names=["x1", "x2"],
    schema=FeatureSchema(...) or None,
    objective="poisson" or "gamma" or None,
    offset=pl.Series(...) or None,
    cv_fold=pl.Series(...) or None,
    comparisons=pl.DataFrame(...) or None,
)
```

Core fields:

- `features`: feature frame used by encoders, selectors, preprocessors, and
  models.
- `target`: target series.
- `exposure`: required for Poisson objective.
- `weight`: optional observation weights.
- `feature_names`: the ordered feature columns that should be used by models.
- `schema`: optional feature role metadata, mainly used by `OneHotEncoder`.
- `objective`: optional objective marker, usually `poisson` or `gamma`.

Additional fields:

- `offset`: optional numeric link-scale offset. LightGBM currently honors this
  field directly. See model-specific caveats below.
- `cv_fold`: optional integer fold ID series used by `HyperparameterTuner` when
  `use_data_folds=True`.
- `comparisons`: optional DataFrame of external model predictions used for
  report comparisons.

Important methods:

- `n_rows`: number of feature rows.
- `validate()`: checks structural and objective-specific rules and returns
  `self`.
- `with_features(features)`: returns a copy with new features and updated
  `feature_names`.
- `with_offset(offset)`: returns a copy with a new offset series.
- `slice_model_data(data, indices)`: returns a row-sliced `ModelData`, including
  optional `offset`, `cv_fold`, and `comparisons` fields.

Validation rules currently implemented:

- `target`, `exposure`, and `weight` row counts must match `features`.
- `feature_names` must be unique.
- Every `feature_name` must exist in `features`.
- If `exposure` is supplied, it must be non-null and positive.
- Poisson objective requires exposure and non-negative target values.
- Gamma objective requires strictly positive target values.
- `offset`, if supplied, must match row count, be numeric, non-null, and finite.
- `cv_fold`, if supplied, must match row count, be integer, non-null, and have
  at least two unique values.
- `comparisons`, if supplied, must match row count, contain numeric columns, and
  every value must be strictly positive.

Important validation pitfall: the current `validate()` implementation does not
explicitly reject null `target` or null `weight` values. Objective checks may
still catch some bad target data, but callers should clean these fields before
constructing `ModelData`.

### `load_model_data`

Defined in `data/loader.py`.

This is the standard parquet loader:

```python
data = load_model_data(
    path="frequency.parquet",
    target="claim_count",
    exposure="exposure",
    feature_cols=["x1", "x2", "x3"],
    objective="poisson",
)
```

Parameters:

- `path`: parquet file path.
- `target`: target column name.
- `exposure`: optional exposure column name.
- `weight`: optional weight column name.
- `feature_cols`: optional explicit feature columns.
- `schema`: optional `FeatureSchema`.
- `objective`: optional objective.
- `cv_fold`: optional column name to store in `ModelData.cv_fold`.
- `comparison_cols`: optional external prediction columns to store in
  `ModelData.comparisons`.

If `feature_cols` is omitted, the loader uses all columns except:

- target
- exposure
- weight
- cv_fold
- comparison columns

`offset` is intentionally not a load-time parameter. Offsets are expected to be
computed after loading and added with `ModelData.with_offset()`.

### `TrainTestSplit`

Defined in `data/splitter.py`.

```python
split = TrainTestSplit(train_ratio=0.7, seed=42)
train, test = split.split(data)
```

Supported split modes:

- random row split, controlled by `seed`.
- group split, if `group_col` is supplied and exists in `data.features`.

The splitter propagates all optional `ModelData` fields:

- `offset`
- `cv_fold`
- `comparisons`

Pitfalls:

- `train_ratio` must be strictly between 0 and 1.
- `group_col` must be present in `features`.
- Group splitting does not automatically remove the group column from model
  features. If the group identifier should not be used as a predictor, remove it
  or keep it out of `feature_cols`.
- This splitter does not implement stratified or exposure-balanced splits in
  the current `ins_gbm` code.

## Preprocessing

Preprocessing lives in `src/ins_gbm/preprocessing/`.

### One-Hot Encoding

Defined in `preprocessing/encoder.py`.

Classes:

- `OneHotEncoder`
- `FittedOneHotEncoder`

The pipeline uses the unfitted `OneHotEncoder` from `ModelRecipe`. It is fit on
training data and then used to transform both train and test.

Output order:

1. numeric columns
2. ordinal columns
3. passthrough columns
4. one-hot indicator columns for each categorical level

Missing value constants:

```python
_NUMERIC_FILL = -999_999_999.0
_MISSING_LEVEL = "-999999999"
```

Current missing handling:

- Numeric and ordinal nulls are filled with `_NUMERIC_FILL`.
- Categorical nulls are filled with `_MISSING_LEVEL`.
- Categorical levels are learned from training data.
- Unknown categories at transform time produce all-zero indicators for that raw
  categorical column because no fitted level matches them.
- Floating `NaN` values are not the same as Polars nulls. If upstream data can
  contain `NaN`, normalize it before fitting or make sure downstream model
  behavior is what you expect.

Framework-specific missing handling later:

- LightGBM converts `_NUMERIC_FILL` back to `np.nan`.
- CatBoost converts `_NUMERIC_FILL` back to `np.nan`.
- XGBoost passes `_NUMERIC_FILL` as the `missing` value in `DMatrix`.
- Random Forest receives `_NUMERIC_FILL` as an ordinary numeric value.

Pitfall: `OneHotEncoder.fit(features, schema)` requires a schema. If the
pipeline recipe includes an encoder, make sure `ModelData.schema` is present.

### PCA Reducer

Defined in `preprocessing/pca.py`.

`PCAReducer`:

- scales all input features with `StandardScaler`.
- fits `sklearn.decomposition.PCA`.
- returns components named `pca_1`, `pca_2`, and so on.
- stores the original input feature names so transform can select the same
  columns in the same order.

The transformed feature frame replaces the original feature frame. Original
features are not appended.

### PLS Reducer

Defined in `preprocessing/pls.py`.

`PLSReducer`:

- is supervised.
- requires `target` at fit time.
- scales features with `StandardScaler`.
- fits `sklearn.cross_decomposition.PLSRegression`.
- returns components named `pls_1`, `pls_2`, and so on.

Pitfall: because PLS is supervised, it must only be fit on training data inside
each split or fold. `ModelPipeline.run()` and ensemble fold helpers pass the
target. `HyperparameterTuner` currently calls `preprocessor.fit(features)`
without the target, so `PLSReducer` is not compatible with tuning as currently
implemented.

### UMAP Reducer

Defined in `preprocessing/umap.py`.

`UMAPReducer`:

- requires optional dependency `umap-learn`.
- scales features with `StandardScaler`.
- fits `umap.UMAP`.
- returns components named `umap_1`, `umap_2`, and so on.

Pitfall: reducers expect numeric input. Categorical columns should usually be
encoded first.

## Feature Selection

Feature selection lives in `src/ins_gbm/selection/`.

### Boruta Selector

Defined in `selection/boruta.py`.

`BorutaSelector` implements a shadow-feature selection process:

1. For each iteration, shuffle every original feature to create shadow features.
2. Fit a base model on original plus shadow features.
3. Compare original feature importance to the maximum shadow importance.
4. Count "hits" across iterations.
5. Use a binomial test to classify features as:
   - `confirmed`
   - `tentative`
   - `rejected`

Supported base estimators:

- `lightgbm`
- `random_forest`

The fitted selector exposes:

- `selected_features()`: confirmed plus tentative features.
- `confirmed_features()`: confirmed features only.
- `classification()`: DataFrame with feature status.

This selector fits the pipeline selector contract because `fit(data)` returns an
object with `selected_features()`.

### Importance Pruner

Defined in `selection/importance.py`.

`ImportancePruner` prunes features from a previously fitted model's feature
importance output.

Selection modes:

- `threshold`: keep features with importance greater than or equal to threshold.
- `percentile`: keep features at or above a percentile cutoff.
- `top_n`: keep the top N features.

Exactly one mode should be set. If none is set, the default threshold is `0.0`.

Important pitfall: `ImportancePruner.fit()` has this signature:

```python
fit(data: ModelData, fitted_model: FittedModel)
```

The main pipeline and tuner selector hooks call:

```python
selector.fit(current_train)
```

That means `ImportancePruner` is useful as a standalone post-fit utility, but it
does not currently fit the pipeline selector contract without an adapter or code
change.

## Model Wrappers

Model wrappers live in `src/ins_gbm/models/`.

### Base Contracts

Defined in `models/base.py`.

Important types:

- `PredictionType = Literal["response", "rate", "link"]`
- `Objective = Literal["poisson", "gamma"]`
- `ModelCapabilities`
- `FittedModel`
- `BaseModel` protocol

Every model wrapper should provide:

- `objective`
- `fit(data, params=None)`
- `default_search_space()`
- `capabilities()`

`fit()` returns a `FittedModel`.

`FittedModel` stores:

- the native model object.
- the params used for fitting.
- framework name.
- objective.
- feature names.
- a `predict_fn` closure.
- an `importance_fn` closure.

The closures are why persistence uses `cloudpickle`.

### LightGBMModel

Defined in `models/lightgbm.py`.

Supports:

- Poisson
- Gamma
- sample weights
- feature importance
- exposure offset for Poisson
- custom `ModelData.offset`

Training behavior:

- Converts feature data to `float64` NumPy.
- Converts `_NUMERIC_FILL` to `np.nan`.
- For Poisson with exposure, uses `log(exposure)` as an initial score.
- If `data.offset` is present, adds it to the initial score.
- Uses `data.weight` as sample weight if supplied.
- Pops `n_estimators` from params and passes it as `num_boost_round`.

Prediction behavior:

- Converts `_NUMERIC_FILL` back to `np.nan`.
- Applies prediction-time `data.offset` if present.
- For Poisson:
  - `response` returns expected claim count.
  - `rate` returns expected rate.
  - `link` returns link-scale predictions.
- For Gamma:
  - `response` returns expected severity.
  - `link` returns log response plus offset if present.
  - `rate` is rejected by `FittedModel.predict()` before wrapper logic.

### XGBoostModel

Defined in `models/xgboost.py`.

Supports:

- Poisson
- Gamma
- sample weights
- feature importance
- exposure base margin for Poisson

Training behavior:

- Converts feature data to `float64` NumPy.
- Passes `_NUMERIC_FILL` as `missing` to `xgb.DMatrix`.
- For Poisson with exposure, uses `log(exposure)` as `base_margin`.
- Uses `data.weight` as sample weight if supplied.
- Pops `n_estimators` from params and passes it as `num_boost_round`.

Prediction behavior:

- For Poisson with exposure, supplies prediction-time `base_margin`.
- `response` returns the model response.
- `rate` divides response by exposure if exposure is present.
- `link` returns `log(response)`.
- For Gamma, returns the raw model response.

Pitfall: the optional `ModelData.offset` field is not currently added by the
XGBoost wrapper, even though the capability object says offset support is true.
The implemented offset-like behavior is exposure base margin for Poisson.

### CatBoostModel

Defined in `models/catboost.py`.

Supports:

- Poisson
- Gamma-like Tweedie objective with variance power `1.99`
- sample weights
- feature importance
- exposure baseline for Poisson if the installed CatBoost version exposes a
  `baseline` fit parameter

Training behavior:

- Converts feature data to `float64` NumPy.
- Converts `_NUMERIC_FILL` to `np.nan`.
- Sets `loss_function` from objective unless caller overrides it.
- Uses `allow_writing_files=False` by default.
- For Poisson with exposure, uses `log(exposure)` as CatBoost baseline only if
  the installed CatBoost supports the `baseline` parameter.
- Uses `data.weight` as sample weight if supplied.

Pitfalls:

- CatBoost offset support depends on the installed CatBoost version.
- The optional `ModelData.offset` field is not currently added by this wrapper.
- The Gamma implementation uses Tweedie power `1.99` because CatBoost requires
  the power to be strictly between 1 and 2.

### RandomForestModel

Defined in `models/random_forest.py`.

This is a benchmark model, not a true GLM-style frequency or severity wrapper.

Poisson behavior:

- If exposure is present, fits on claim rate: `target / exposure`.
- Uses exposure as sample weight.
- `response` multiplies predicted rate by exposure.
- `rate` returns predicted rate.
- `link` returns `log(rate clipped above zero)`.

Gamma behavior:

- Fits directly on target with optional `data.weight`.
- Returns predictions clipped to be positive.

Pitfalls:

- No native exposure offset.
- No native missing-value handling. Numeric sentinel values are treated as real
  numbers.
- Useful as a benchmark, but not as a replacement for a Poisson likelihood model.

## ModelPipeline

The pipeline orchestration lives in `src/ins_gbm/pipeline.py`.

### ModelRecipe

`ModelRecipe` is the unfitted configuration:

```python
ModelRecipe(
    model=LightGBMModel(objective="poisson"),
    encoder=OneHotEncoder(),
    selection=BorutaSelector(...),
    preprocessing=[PCAReducer(n_components=3)],
    tuning=HyperparameterTuner(...),
    params={"n_estimators": 100},
)
```

Fields:

- `model`: required model wrapper.
- `encoder`: optional encoder.
- `selection`: optional selector.
- `preprocessing`: optional list of preprocessors.
- `tuning`: optional `HyperparameterTuner`.
- `params`: optional manual params used when tuning is not enabled.

Pitfall: if `tuning` is present, tuned best params take precedence over
`recipe.params`.

### Run Order

`ModelPipeline.run()` executes in this order:

1. Split data with `TrainTestSplit`.
2. Optionally tune hyperparameters on the training split only.
3. Fit encoder on the full training split and transform train and test.
4. Fit selector on transformed training data and select the same columns from
   train and test.
5. Fit each preprocessor on training features and target, then transform train
   and test.
6. Fit model on transformed training data.
7. Build `EvaluationReport` on transformed test data.
8. Build reproducibility metadata.
9. Return `FittedPipeline`.

The final test set is not passed into tuning, selector fitting, preprocessor
fitting, or model fitting.

### Tuning Inside Pipeline

If `recipe.tuning` is supplied, pipeline tuning calls:

```python
self.recipe.tuning.tune(
    train_data,
    self.recipe.model,
    encoder=self.recipe.encoder,
    selector=self.recipe.selection,
    preprocessor=single_prep,
    schema=train_data.schema,
    progress=self.progress,
    should_stop=self.should_stop,
)
```

Only the first preprocessor is passed into tuning. The full list of
preprocessors is still used during the final full-train refit.

Pitfalls:

- Multiple-preprocessor chains are not fully represented during tuning.
- PLS preprocessing is not compatible with tuning in current code because the
  tuner does not pass the target into `preprocessor.fit()`.
- `ImportancePruner` does not match the tuner selector signature.

### FittedPipeline

`FittedPipeline` is the result object returned by `ModelPipeline.run()`.

Important fields:

- `fitted_model`: the final `FittedModel`.
- `recipe`: the original `ModelRecipe` object.
- `train_data`: transformed training data as seen by the fitted model.
- `test_data`: transformed test data as seen by the fitted model.
- `selected_features`: selected feature names, if selection was used.
- `tuning_history`: Optuna history DataFrame, if tuning was used.
- `report`: `EvaluationReport`.
- `encoder`: fitted encoder, if used.
- `preprocessors`: fitted preprocessors.
- `metadata`: reproducibility metadata.

Important methods:

- `predict(data, prediction_type="response")`
- `predict_raw(features, exposure=None, weight=None, prediction_type="response")`

Use `fitted_model.predict(result.test_data)` when you already have transformed
model-ready data.

Use `result.predict(raw_model_data)` when you have a raw `ModelData` shaped like
the original pre-transform data.

Use `result.predict_raw(raw_features, exposure=...)` when scoring a feature
DataFrame without a real target column.

Pitfalls:

- `FittedPipeline.predict()` applies the fitted transform chain. If an encoder
  was used, pass raw feature columns, not already encoded features.
- `FittedPipeline.predict_raw()` creates a placeholder target. It does not
  replace the need to provide exposure for Poisson expected claim count scoring.
- `train_data` and `test_data` are transformed data. The raw split rows are not
  stored on `FittedPipeline`.

## Hyperparameter Tuning

Tuning lives in `src/ins_gbm/tuning/`.

### HyperparameterTuner

Defined in `tuning/tuner.py`.

```python
tuner = HyperparameterTuner(
    n_trials=20,
    cv_folds=5,
    metric="poisson_deviance",
    seed=42,
    use_data_folds=False,
)
```

Supported metrics:

- `poisson_deviance`
- `gamma_deviance`
- `rmse`
- `mae`

For each Optuna trial:

1. Draw params from `model.default_search_space()`.
2. Build fold splits:
   - KFold if `use_data_folds=False`.
   - `ModelData.cv_fold` if `use_data_folds=True`.
3. Slice training and validation `ModelData`.
4. Fit encoder on fold training data and transform fold validation data.
5. Fit selector on fold training data and apply selected columns to validation.
6. Fit preprocessor on fold training features and transform validation.
7. Fit model on fold training data.
8. Predict validation response.
9. Score validation predictions.
10. Report intermediate score to Optuna for pruning.

The tuner returns:

```python
best_params, history = tuner.tune(...)
```

`history` has one row per completed trial and includes:

- `trial`
- `value`
- one column per tuned hyperparameter

### Search Spaces

Every model wrapper exposes `default_search_space()` using Optuna distribution
objects.

`tuning/search_spaces.py` provides:

```python
narrow_search_space(space, **overrides)
```

This returns a copied search space with selected distributions replaced.

Pitfall: there is no constructor parameter on `HyperparameterTuner` for a custom
search space. To use a custom search space, wrap or subclass the model, or adjust
the model method.

## Evaluation and Reporting

Evaluation lives in `src/ins_gbm/evaluation/`.

### Metrics

Defined in `evaluation/metrics.py`.

Metric functions:

- `poisson_deviance(actual, predicted, weights=None)`
- `gamma_deviance(actual, predicted, weights=None)`
- `normalized_gini(actual, predicted, weights=None)`
- `rmse(actual, predicted, weights=None)`
- `mae(actual, predicted, weights=None)`
- `compute_metrics(...)`

`compute_metrics()` returns a DataFrame with:

- objective-specific deviance
- gini
- rmse
- mae

For Poisson, exposure is used as the deviance and Gini weight. For Gamma, weight
is used.

Pitfalls:

- Poisson and Gamma deviance require positive predictions.
- Gamma deviance also requires positive actual values.
- `rmse` and `mae` in `compute_metrics()` are currently unweighted.

### EvaluationReport

Defined in `evaluation/report.py`.

An `EvaluationReport` can operate in three modes:

1. Single-model report.
2. Single GBM model plus external comparison predictions.
3. Multi-model comparison mode from `EvaluationReport.compare()`.

Primary methods:

- `metrics()`
- `plot_lift(output_path=None)`
- `plot_ave(output_path=None)`
- `plot_calibration(output_path=None)`
- `plot_feature_importance(output_path=None)`
- `plot_double_lift(name, output_path=None)`
- `export(output_dir)`

`export(output_dir)` writes:

- `metrics.csv`
- `lift.png`
- `ave.png`
- `calibration.png`
- `feature_importance.png`
- double-lift plots when comparison predictions exist

External comparison predictions can come from `ModelData.comparisons`. The main
pipeline passes test-set comparison columns into `EvaluationReport` as named
prediction series.

### CrossValidationReport

Defined in `evaluation/cv_report.py`.

`CrossValidationReport` runs repeated fold fits and reports fold stability:

```python
result = CrossValidationReport(
    recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    data=data,
    n_folds=5,
    benchmark_col=None,
    fold_col=None,
    seed=42,
).run()
```

Returns `CVResult`:

- `fold_metrics`: columns `fold`, `model`, `metric`, `value`.
- `summary`: columns `model`, `metric`, `mean`, `std`.
- `fold_col`: name of fold column if predefined feature folds were used.

Fold modes:

- random folds if `fold_col=None`.
- predefined folds if `fold_col` is a column in `data.features`.

Benchmark mode:

- `benchmark_col` can name a positive prediction column in `data.features`.
- The benchmark column is dropped before fitting the GBM recipe.
- Fold metrics include both `gbm` and `benchmark` rows.

Important distinction: `HyperparameterTuner(use_data_folds=True)` uses
`ModelData.cv_fold`. `CrossValidationReport(fold_col=...)` expects the fold
column to be present in `data.features`.

### Report Comparison

Defined in `evaluation/comparison.py`.

`compare_reports()` compares multiple `EvaluationReport` or `CVResult` objects:

```python
table = compare_reports({
    "lightgbm": result_lgb.report,
    "xgboost": result_xgb.report,
})
```

The output has one row per metric, one column per report name, and a `preferred`
column. Direction comes from `METRIC_DIRECTIONS`:

- higher is better for `gini`.
- lower is better for deviance, RMSE, and MAE.

Pitfall: comparison-mode `EvaluationReport` objects from
`EvaluationReport.compare()` cannot be passed into `compare_reports()`. Pass
individual single-model reports or `CVResult` objects instead.

## Ensembles

Ensemble code lives in `src/ins_gbm/ensemble/`.

The ensemble layer combines already fitted `FittedPipeline` objects.

### Shared Helpers

`ensemble/_utils.py` contains:

- `_apply_pipeline_transforms(pipeline, data)`: applies a fitted pipeline's
  encoder, selected feature subset, and preprocessors to raw data.
- `_predict_from_pipeline(pipeline, data)`: transforms data and gets response
  predictions.
- `_apply_recipe_fold_transforms(recipe, fold_train, fold_val)`: fits recipe
  transformations on fold training data and applies them to validation data.

These helpers are central to leakage control in OOF blending and stacking.

### BlendingEnsemble

Defined in `ensemble/blending.py`.

Blend modes:

- `fixed`: user supplies weights.
- `validation`: optimize weights on user-supplied validation data.
- `oof`: generate out-of-fold predictions from base pipeline training data and
  optimize weights.

`FittedBlendingEnsemble.predict(data)` returns the weighted average of base
pipeline predictions.

Pitfalls:

- Fixed weights must sum to 1.
- Validation data must not be the final test set. The code trusts the caller.
- OOF mode refits each base pipeline recipe inside folds.

### StackingEnsemble

Defined in `ensemble/stacking.py`.

Stacking flow:

1. For each fitted base pipeline, refit the pipeline recipe inside KFold splits
   of that pipeline's training data.
2. Collect out-of-fold predictions into a matrix.
3. Fit a meta-learner on that matrix and the training target.
4. For prediction, get base predictions from the original fitted pipelines and
   run the meta-learner.

Default meta-learner:

- `sklearn.linear_model.Ridge`

Predictions are clipped to be positive.

Pitfalls:

- Base pipelines should have the same objective, compatible train/test rows, and
  compatible row order. The code does not perform deep compatibility checks.
- OOF refits call `pipeline.recipe.model.fit(current_train)` without explicitly
  passing tuned best params or `recipe.params`.
- The final report uses the first base pipeline's test data.

### EnsemblePipeline

Defined in `ensemble/pipeline.py`.

Unified wrapper for blending and stacking:

```python
ensemble_result = EnsemblePipeline(
    fitted_pipelines=[pipeline_a, pipeline_b],
    method="blending",
    blend_mode="fixed",
    blend_weights=[0.5, 0.5],
).run()
```

Returns `EnsembleResult`:

- `ensemble`: fitted blending or stacking ensemble.
- `report`: `EvaluationReport` using a proxy `FittedModel`.
- `base_pipelines`: base fitted pipelines.

The proxy model lets `EvaluationReport` call `predict()` on the ensemble.

## Persistence

Persistence lives in `src/ins_gbm/persistence/`.

### Saving

Defined in `persistence/io.py`.

```python
from ins_gbm.persistence.io import save_pipeline

save_pipeline(result, "output/my_model")
```

Artifacts written:

- `pipeline.pkl`: full fitted pipeline via `cloudpickle`.
- `metadata.json`: `ReproducibilityMetadata` as JSON.
- `metrics.csv`: report metrics, if they can be generated.
- `tuning_history.parquet`: only when tuning history exists.

### Loading

```python
from ins_gbm.persistence.io import load_pipeline

loaded = load_pipeline("output/my_model")
```

Pitfalls:

- Standard `pickle` and `joblib` are not used because `FittedModel` contains
  local prediction and importance closures.
- Loading is safest in an environment with compatible package versions.
- `metadata.json` is useful for auditing but does not recreate the pipeline by
  itself.

### Metadata

Defined in `persistence/metadata.py`.

`ReproducibilityMetadata` records:

- package versions
- random seeds
- model params
- fitted feature names
- selected features
- objective
- prediction scale

`build_metadata()` currently records package versions for:

- `ins_gbm`
- `polars`
- `numpy`
- `scikit-learn`
- `optuna`
- `lightgbm`
- `xgboost`
- `catboost`

## Progress and Cancellation

Defined in `progress.py`.

Types:

- `ProgressEvent`
- `ProgressCallback`
- `PipelineCancelled`

`ModelPipeline` can receive:

- `progress`: callback that accepts a `ProgressEvent`.
- `should_stop`: callback returning truthy when the run should be cancelled.

Pipeline stages include:

- `split`
- `tuning`
- `encode`
- `select`
- `preprocess`
- `fit`
- `evaluate`

The tuner also emits trial-level progress events when a progress callback is
provided.

## Common Usage Flows

### Basic Poisson Frequency Pipeline

```python
from ins_gbm.data.loader import load_model_data
from ins_gbm.data.splitter import TrainTestSplit
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import ModelPipeline, ModelRecipe

data = load_model_data(
    path="frequency.parquet",
    target="claim_count",
    exposure="exposure",
    feature_cols=["x1", "x3"],
    objective="poisson",
)

recipe = ModelRecipe(
    model=LightGBMModel(objective="poisson"),
    params={"n_estimators": 100},
)

result = ModelPipeline(
    data=data,
    split=TrainTestSplit(train_ratio=0.7, seed=42),
    recipe=recipe,
).run()

metrics = result.report.metrics()
preds = result.fitted_model.predict(result.test_data, prediction_type="response")
```

### Basic Gamma Severity Pipeline

```python
from ins_gbm.data.loader import load_model_data
from ins_gbm.data.splitter import TrainTestSplit
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import ModelPipeline, ModelRecipe

data = load_model_data(
    path="severity.parquet",
    target="severity",
    weight="weight",
    feature_cols=["x1"],
    objective="gamma",
)

result = ModelPipeline(
    data=data,
    split=TrainTestSplit(seed=42),
    recipe=ModelRecipe(model=LightGBMModel(objective="gamma")),
).run()
```

### Categorical Features with One-Hot Encoding

```python
from ins_gbm.data.schema import FeatureSchema
from ins_gbm.preprocessing.encoder import OneHotEncoder

schema = FeatureSchema(
    numeric=["x1", "x3"],
    categorical=["territory"],
)

data = load_model_data(
    path="frequency.parquet",
    target="claim_count",
    exposure="exposure",
    feature_cols=schema.all_features(),
    schema=schema,
    objective="poisson",
)

recipe = ModelRecipe(
    model=LightGBMModel(objective="poisson"),
    encoder=OneHotEncoder(),
)
```

### Optuna Tuning

```python
from ins_gbm.tuning.tuner import HyperparameterTuner

recipe = ModelRecipe(
    model=LightGBMModel(objective="poisson"),
    tuning=HyperparameterTuner(
        n_trials=25,
        cv_folds=5,
        metric="poisson_deviance",
        seed=42,
    ),
)

result = ModelPipeline(data=data, split=TrainTestSplit(seed=42), recipe=recipe).run()
history = result.tuning_history
```

### Predefined Folds for Tuning

```python
data = load_model_data(
    path="frequency_with_folds.parquet",
    target="claim_count",
    exposure="exposure",
    feature_cols=["x1", "x3"],
    objective="poisson",
    cv_fold="fold_id",
)

tuner = HyperparameterTuner(
    n_trials=20,
    use_data_folds=True,
    seed=42,
)
```

Remember that this uses `ModelData.cv_fold`, not a feature column.

### Cross-Validation Report

```python
from ins_gbm.evaluation.cv_report import CrossValidationReport

cv_result = CrossValidationReport(
    recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    data=data,
    n_folds=5,
    seed=42,
).run()

fold_metrics = cv_result.fold_metrics
summary = cv_result.summary
```

### Saving and Loading a Pipeline

```python
from ins_gbm.persistence.io import save_pipeline, load_pipeline

save_pipeline(result, "output/frequency_model")
loaded = load_pipeline("output/frequency_model")
```

### Fixed-Weight Blend

```python
from ins_gbm.ensemble.pipeline import EnsemblePipeline

ensemble_result = EnsemblePipeline(
    fitted_pipelines=[result_lgb, result_xgb],
    method="blending",
    blend_mode="fixed",
    blend_weights=[0.6, 0.4],
).run()

ensemble_preds = ensemble_result.predict(result_lgb.test_data)
```

### Stacking

```python
ensemble_result = EnsemblePipeline(
    fitted_pipelines=[result_lgb, result_rf],
    method="stacking",
    cv_folds=5,
    seed=42,
).run()
```

## Tests and What They Cover

Tests live under `tests/`.

Major areas:

- `tests/data/`: schema inference, model data validation, loader behavior,
  splitter behavior, optional fields.
- `tests/preprocessing/`: one-hot encoding and reducers.
- `tests/models/`: base contracts and model wrapper behavior.
- `tests/tuning/`: Optuna tuning and predefined folds.
- `tests/selection/`: Boruta and importance pruning.
- `tests/evaluation/`: metrics, plots, reports, CV reports, comparison helpers.
- `tests/ensemble/`: blending, stacking, and ensemble pipeline.
- `tests/persistence/`: save/load behavior.
- `tests/test_pipeline.py`: main pipeline behavior.
- `tests/test_integration.py`: end-to-end flows and leakage checks.
- `tests/test_progress.py`: progress callbacks and cancellation.

The fixtures in `tests/conftest.py` generate synthetic Poisson and Gamma data:

- `poisson_raw`: 400 rows with `x1`, `x2`, `x3`, `exposure`, and `claim_count`.
- `gamma_raw`: 300 rows with `x1`, `x2`, `severity`, and `weight`.
- parquet fixtures write those frames to temporary files.

## Extension Points

### Adding a Model Wrapper

Create a module under `src/ins_gbm/models/` that implements the `BaseModel`
protocol:

1. Add an unfitted dataclass with an `objective` field.
2. Implement `capabilities()`.
3. Implement `default_search_space()`.
4. Implement `fit(data, params=None)`.
5. Convert `data.features.select(data.feature_names)` to the model's expected
   matrix format.
6. Return a `FittedModel` with `predict_fn` and `importance_fn`.
7. Add model-specific tests under `tests/models/`.

Be explicit about:

- prediction scales.
- exposure handling.
- optional `ModelData.offset` handling.
- sample weights.
- missing-value behavior.
- feature importance units.

### Adding a Preprocessor

A preprocessor should fit this shape:

```python
fitted = preprocessor.fit(features, target=None)
transformed = fitted.transform(features)
```

For supervised preprocessors, require target and make sure every call site that
uses the preprocessor passes target.

Important call sites:

- `ModelPipeline.run()`
- `HyperparameterTuner.tune()`
- `ensemble/_utils.py`
- `CrossValidationReport.run()` through ensemble fold utilities

### Adding a Selector

Pipeline-compatible selectors should support:

```python
fitted = selector.fit(data)
selected = fitted.selected_features()
```

If the selector requires a fitted model, it currently needs an adapter or a
pipeline change.

## Common Pitfalls

### Importing the Wrong Package Name

The package in this directory is `ins_gbm`.

Use:

```python
from ins_gbm.pipeline import ModelPipeline
```

Do not use `gbm_fitting` for this project.

### Assuming Root Package Exports Everything

`src/ins_gbm/__init__.py` only exports progress-related types in the current
code. Import most classes from their concrete modules.

### Treating `FittedPipeline.train_data` as Raw Data

`FittedPipeline.train_data` and `FittedPipeline.test_data` are transformed
model-ready frames. If an encoder, selector, or reducer was used, these are not
the raw parquet features.

### Passing Transformed Data to `FittedPipeline.predict()`

`FittedPipeline.predict()` applies the transform chain. If the pipeline has an
encoder, pass raw columns. For already transformed data, call
`result.fitted_model.predict(transformed_data)` instead.

### Forgetting Exposure for Poisson

Poisson `ModelData.validate()` requires exposure. Prediction methods can
sometimes run without exposure and effectively behave as if exposure were 1, but
that is usually not the intended expected claim count workflow. Supply exposure
when scoring Poisson models.

### Using Gamma Rate Predictions

`prediction_type="rate"` is invalid for Gamma and raises.

### Confusing `cv_fold` and `fold_col`

There are two fold mechanisms:

- `ModelData.cv_fold` is used by `HyperparameterTuner(use_data_folds=True)`.
- `CrossValidationReport(fold_col=...)` expects a fold column inside
  `data.features`.

They are not the same API.

### Expecting Stratified Splits

The current `TrainTestSplit` supports random and group splits only.

### Leaving Group IDs in Model Features

If `group_col` is used for splitting and also remains in `feature_names`, the
model may train on that identifier. Remove it from the modeling features unless
that is intentional.

### Using `ImportancePruner` Directly in `ModelRecipe.selection`

`ImportancePruner` requires a fitted model. The pipeline selector hook does not
provide one. `BorutaSelector` matches the current hook; `ImportancePruner` is
best used manually after fitting a model.

### Tuning with PLS

`PLSReducer` requires target at fit time. The main pipeline passes target during
the final fit, but the tuner currently does not pass target to preprocessors.
Avoid `PLSReducer` inside tuned recipes unless the tuner is updated.

### Tuning with Multiple Preprocessors

`ModelPipeline.run()` passes only the first preprocessor into the tuner. The
final full-training refit still applies the full preprocessor list.

### Relying on Custom Offset Outside LightGBM

`ModelData.offset` is currently honored directly by LightGBM. XGBoost and
CatBoost implement exposure-related base margins or baselines in their wrappers,
but they do not currently add the optional `ModelData.offset` series.

### Assuming Optional Dependencies Are Installed

LightGBM, XGBoost, CatBoost, SHAP, and UMAP are optional extras. Install the
right extras before using wrappers or reducers that need them.

### Mixing NaN and Null Missing Values

The encoder fills Polars nulls. Floating `NaN` values are different and should
be handled explicitly upstream if present.

### Treating Random Forest as a True Poisson or Gamma Objective

`RandomForestModel` is a benchmark. Its Poisson behavior fits rates with exposure
weights; it does not optimize a Poisson likelihood with a native log exposure
offset.

### Optimizing Blends on the Final Test Set

The code supports validation-mode blending but cannot know whether the
validation data is actually the final test set. The caller is responsible for
keeping final test data untouched.

### Assuming Ensemble Inputs Are Validated Deeply

The ensemble code expects fitted pipelines to be compatible. It uses the first
pipeline's train and test data as the reference. Make sure base pipelines share
the same objective, row basis, and intended evaluation split.

### Expecting Tuned Params in Ensemble OOF Refits

Stacking and OOF blending refit base recipes inside folds. The current fold
refit path calls `recipe.model.fit(current_train)` without passing tuned best
params or manual recipe params.

### Persisting Across Incompatible Environments

Saved pipelines use `cloudpickle`. They are convenient for same-project
round-trips, but they are not a stable interchange format across major library
or Python version changes.

## Mental Model for New Contributors

The main invariant to preserve is leakage control:

- fit encoders on training data only.
- fit selectors on training data only.
- fit reducers on training data only.
- tune only on the training split.
- optimize blend weights or stacking meta-learners without final test data.
- evaluate once on the final test set.

The second invariant is data shape consistency:

- every `ModelData` row-level field must remain aligned when slicing or
  splitting.
- every model should use `data.features.select(data.feature_names)`.
- transform steps should update `feature_names` when they replace features.
- prediction data must have the columns required by the fitted transform chain
  and the fitted model.

The third invariant is objective consistency:

- Poisson means claim count target plus positive exposure.
- Gamma means strictly positive severity target.
- rate predictions are Poisson-only.
- deviance metrics require positive predictions.

If a change touches one of these invariants, add or update tests around the full
flow, not just the individual function.
