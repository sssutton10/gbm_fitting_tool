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
3. Optionally tune model hyperparameters with fold-local cross-validation.
4. Fit feature preparation steps and the model on all supplied training rows.
5. Evaluate an explicitly supplied holdout with the fitted artifacts.
6. Optionally export reports, persist the fitted pipeline, or combine fitted
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
    preprocessing/
        __init__.py
        encoder.py
        steps.py
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

### Explicit holdout data

The library does not create holdout partitions. Supply all training rows to
`ModelPipeline`, then construct and retain any final holdout in caller code.
Evaluate it explicitly with `fitted_pipeline.evaluate(holdout_data)`.

## Preprocessing

Preprocessing lives in `src/ins_gbm/preprocessing/`.

### One-Hot Encoding

Defined in `preprocessing/encoder.py`.

Classes:

- `OneHotEncoder`
- `FittedOneHotEncoder`

The pipeline uses the unfitted `OneHotEncoder` from `ModelRecipe`. It is fit on
all supplied training rows, or on each fold's training rows during tuning, and
the fitted encoder is reused for evaluation and prediction.

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
each split or fold. `ModelPipeline.run()`, `HyperparameterTuner`, and ensemble
fold helpers pass the fold-training target to the reducer.

### UMAP Reducer

Defined in `preprocessing/umap.py`.

`UMAPReducer`:

- requires optional dependency `umap-learn`.
- scales features with `StandardScaler`.
- fits `umap.UMAP`.
- returns components named `umap_1`, `umap_2`, and so on.

Pitfall: reducers expect numeric input. Categorical columns should usually be
encoded first.

### Multiple Preprocessing Steps

`ModelRecipe.preprocessing` accepts a list of preprocessors. The pipeline fits
and applies every item in that ordered chain after encoding and selection; the
same complete chain is refit on each training fold during tuning, cross-
validation, blending, and stacking. This preserves leakage isolation for every
step, including supervised reducers such as PLS.

`PreprocessingStep` in `preprocessing/steps.py` is the preferred wrapper when
a reducer should affect only selected columns. It replaces its input columns
with name-prefixed outputs while passing all other columns through. Multiple
targeted steps can be supplied in one recipe, provided their step names are
unique.

The current chain is sequential, not concurrently fit: a later preprocessor
receives the output of the preceding one. Use independent
`PreprocessingStep`s for separate feature groups; overlapping or dependent
steps should be ordered deliberately.

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

### Staged Importance Selection

Defined in `selection/importance.py` and exported from `ins_gbm.selection`.

`StagedImportanceSelector` accepts an ordered list of
`ImportanceSelectionStage` objects. Every stage declares:

- an unfitted importance-capable model;
- fixed model parameters for that stage;
- `max_features`, the maximum number of encoded model columns to retain;
- an optional framework-native `importance_type`; and
- an optional audit name.

Each stage fits on the columns retained by the prior stage, ranks importance in
descending order, and keeps at most `max_features` columns. Equal scores retain
incoming feature order. Selection runs after one-hot encoding, so feature caps
refer to encoded columns rather than original source fields.

The usual pattern is a shallow, fast screening learner followed by a more
realistic tree configuration for final pruning. Stage learner parameters remain
fixed during hyperparameter tuning; Optuna tunes only the final recipe model.
The selector is refit independently on fold-training data in tuning,
cross-validation, blending, and stacking.

`FittedStagedImportanceSelector` exposes every stage's ranking DataFrame with
`feature`, `importance`, `rank`, and `selected` columns. `FittedPipeline` keeps
this audit information in `selection_results`; its existing `selected_features`
field remains the final stage's retained columns. Reproducibility metadata
contains concise stage configuration and retained-column records.

Native scalar importance types supported by the wrappers are:

- LightGBM: `gain`, `split`.
- XGBoost: `weight`, `gain`, `cover`, `total_gain`, `total_cover`.
- CatBoost: `FeatureImportance`, `PredictionValuesChange`,
  `LossFunctionChange`.
- Random Forest: `impurity`.

Unsupported types, non-finite scores, and models without feature-importance
support raise `ValueError`; interaction and SHAP outputs are intentionally not
ranked as one score per feature.

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
    preprocessing=[
        PreprocessingStep(
            name="numeric_pca",
            preprocessor=PCAReducer(n_components=3),
            feature_names=["x1", "x3", "vehicle_age"],
        ),
    ],
    tuning=HyperparameterTuner(...),
    params={"n_estimators": 100},
)
```

Fields:

- `model`: required model wrapper.
- `encoder`: optional encoder.
- `selection`: optional selector.
- `preprocessing`: optional ordered list of preprocessors or
  `PreprocessingStep` wrappers.
- `tuning`: optional `HyperparameterTuner`.
- `params`: optional manual params used when tuning is not enabled.

Pitfall: if `tuning` is present, tuned best params take precedence over
`recipe.params`.

### Run Order

`ModelPipeline.run()` executes in this order:

1. Optionally tune hyperparameters with fold-local cross-validation fitting.
2. Fit encoder, feature selection, the full ordered preprocessing chain, and
   model on all supplied data.
3. Build reproducibility metadata and return `FittedPipeline`.

Call `FittedPipeline.evaluate(holdout_data)` to transform and evaluate a
caller-provided final holdout. That holdout is never used for tuning, selector
fitting, preprocessor fitting, or model fitting.

### Tuning Inside Pipeline

If `recipe.tuning` is supplied, pipeline tuning calls:

```python
self.recipe.tuning.tune(
    train_data,
    self.recipe.model,
    encoder=self.recipe.encoder,
    selector=self.recipe.selection,
    preprocessors=self.recipe.preprocessing,
    schema=train_data.schema,
    progress=self.progress,
    should_stop=self.should_stop,
)
```

The full preprocessing chain is fit and applied on each fold's training data,
then applied to that fold's validation data.

Pitfalls:

- Preprocessors are an ordered transform chain, not concurrent jobs.
- `ImportancePruner` requires an already-fitted model and does not match the
  pipeline selector signature. Use `StagedImportanceSelector` for in-pipeline
  importance pruning.

### FittedPipeline

`FittedPipeline` is the result object returned by `ModelPipeline.run()`.

Important fields:

- `fitted_model`: the final `FittedModel`.
- `recipe`: the original `ModelRecipe` object.
- `raw_train_data`: reusable raw training data retained for OOF ensemble fits.
- `train_data`: non-cached property that reconstructs the transformed training
  data only when explicitly accessed.
- `selected_features`: selected feature names, if selection was used.
- `selection_results`: per-stage importance rankings and selected columns for a
  staged importance selector.
- `tuning_history`: Optuna history DataFrame, if tuning was used.
- `report`: `EvaluationReport`.
- `encoder`: fitted encoder, if used.
- `preprocessors`: fitted preprocessors.
- `metadata`: reproducibility metadata.

Important methods:

- `predict(data, prediction_type="response")`
- `predict_raw(features, exposure=None, weight=None, prediction_type="response")`

Use `result.predict(holdout_data)` for raw holdout data, or
`result.evaluate(holdout_data)` to produce metrics and plots.

Use `result.predict(raw_model_data)` when you have a raw `ModelData` shaped like
the original pre-transform data.

Use `result.predict_raw(raw_features, exposure=...)` when scoring a feature
DataFrame without a real target column.

Pitfalls:

- `FittedPipeline.predict()` applies the fitted transform chain. If an encoder
  was used, pass raw feature columns, not already encoded features.
- `FittedPipeline.predict_raw()` creates a placeholder target. It does not
  replace the need to provide exposure for Poisson expected claim count scoring.
- Accessing `train_data` constructs transformed data; the result is not stored
  on `FittedPipeline`. Holdouts are also never stored on the pipeline itself.

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
6. Fit each preprocessor in the ordered chain on fold training features and
   transform both training and validation data.
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

External comparison predictions can come from `ModelData.comparisons`. The
caller-provided holdout supplies those columns to `EvaluationReport` as named
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
    "lightgbm": result_lgb.evaluate(lightgbm_holdout),
    "xgboost": result_xgb.evaluate(xgboost_holdout),
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
- Validation data must not be the final evaluation holdout. The code trusts the
  caller.
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

- Base pipelines should have the same objective, compatible raw input rows, and
  compatible row order. The code does not perform deep compatibility checks.
- OOF refits call `pipeline.recipe.model.fit(current_train)` without explicitly
  passing tuned best params or `recipe.params`.
- Ensemble evaluation uses the caller-provided holdout; the first base
  pipeline's transformed training data is retained as report context.

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
    recipe=recipe,
).run()

report = result.evaluate(holdout_data)
metrics = report.metrics()
preds = result.predict(holdout_data, prediction_type="response")
```

### Basic Gamma Severity Pipeline

```python
from ins_gbm.data.loader import load_model_data
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

result = ModelPipeline(data=data, recipe=recipe).run()
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

ensemble_preds = ensemble_result.predict(holdout_data)
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

- `tests/data/`: schema inference, model data validation, loader behavior, and
  optional fields.
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
pipeline change. `StagedImportanceSelector` is the built-in model-driven
selector: its stages fit their own declared learners and then return the final
selected columns through this same interface.

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

`FittedPipeline.train_data` reconstructs a transformed model-ready frame on
each access. If an encoder, selector, or reducer was used, it is not the raw
parquet frame and callers should avoid retaining it when memory is constrained.
Use `raw_train_data` for the shared raw reference.

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

### Managing Holdouts

The caller owns holdout construction. Keep final holdout rows separate from
the `ModelData` passed to `ModelPipeline.run()` and pass them only to
`FittedPipeline.evaluate()`.

### Leaving Group IDs in Model Features

If `group_col` is used for splitting and also remains in `feature_names`, the
model may train on that identifier. Remove it from the modeling features unless
that is intentional.

### Using `ImportancePruner` Directly in `ModelRecipe.selection`

`ImportancePruner` requires a fitted model. The pipeline selector hook does not
provide one. `BorutaSelector` and `StagedImportanceSelector` match the current
hook; `ImportancePruner` is best used manually after fitting a model.

### Tuning with PLS

`PLSReducer` requires target at fit time. The pipeline, tuner, and fold helpers
pass the fold-training target to every preprocessing step, so it is safe to use
in a tuned recipe. As with any supervised transformation, it is deliberately
refit independently on each fold.

### Tuning with Multiple Preprocessors

The full ordered preprocessor list participates in tuning and the final
full-training refit. Steps are sequential: each receives the transformed frame
from the preceding step, so they are not fitted concurrently.

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

### Optimizing Blends on the Final Holdout

The code supports validation-mode blending but cannot know whether the
validation data is actually the final evaluation holdout. The caller is
responsible for keeping that holdout untouched.

### Assuming Ensemble Inputs Are Validated Deeply

The ensemble code expects fitted pipelines to be compatible. It uses the first
pipeline's training data as report context. Make sure base pipelines share the
same objective, row basis, and intended evaluation holdout.

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

- fit encoders, selectors, and reducers only on the rows available to the
  current fit or CV training fold.
- tune only with fold-local training transformations.
- optimize blend weights or stacking meta-learners without the final holdout.
- evaluate once on a caller-provided final holdout.

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
