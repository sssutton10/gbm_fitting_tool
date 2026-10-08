# GBM Fitting Project Structure and Main Flows

This document describes the `GBM Fitting` project in this directory. It covers
the `ins_gbm` package only.

The project is a Python library for policy-level insurance modeling. It is
designed around Polars data structures at the public API boundary, parquet input
files, leakage-conscious modeling workflows, and wrappers around common machine
learning libraries used for frequency and severity models.

The library is not a command line application. Most usage happens through Python
imports and object construction.

## Current Public Workflow and Migration Notes

The preferred common workflow is now:

```python
from ins_gbm import CVConfig, LightGBMModel, ModelRecipe, load_model_data

data = load_model_data(
    "training.parquet", target="claim_count", exposure="exposure",
    objective="poisson", cv_fold="fold_id",
)
recipe = ModelRecipe(model=LightGBMModel())
cv_result = recipe.cross_validate(data, cv=CVConfig(folds="auto"))
fitted = recipe.fit(data)
predictions = fitted.predict(data.features, exposure=data.exposure)
fitted.save("output/model")
cv_result.save("output/model/cv_report")
```

The original `ModelPipeline(data, recipe).run()`, `predict_raw()`,
`save_pipeline()`, `load_pipeline()`, and deep module imports remain supported.
The shorter API delegates to those same implementations.

Behavioral changes made during the 2026 correctness review:

- LightGBM predictions no longer exponentiate an already transformed response.
- `link` consistently means `log(response)` for Poisson and Gamma. For Poisson
  it therefore includes exposure; use `log(rate)` when a log rate is required.
- Explicit offsets are additive on the link scale in LightGBM, XGBoost, and
  supported CatBoost versions. Random Forest rejects offsets.
- Targets, exposures, weights, offsets, and comparison predictions receive
  finite/non-null validation before fitting or evaluation.
- `CVConfig(folds="auto")` uses `ModelData.cv_fold` when present and otherwise
  creates shuffled folds. Fold IDs remain row metadata rather than predictors.
- Encoding and learned feature selection run before tuning by default.
  Preprocessing is fold-local during tuning. `ModelRecipe(selection_scope="fold")`
  refits encoding and selection in each tuning fold.
- Ensemble OOF refits preserve the fitted base model's effective parameters.
- `HyperparameterTuner(metric=None)` infers objective-specific deviance and merges
  trial suggestions over `ModelRecipe.params`.
- LightGBM enables row sampling when a sampling fraction is below one unless
  the caller supplies a bagging frequency. XGBoost keeps prediction-time offsets
  consistent with the intercept convention used during training.
- Tuning pools validation scores by effective metric weight, preserves Optuna
  distribution steps, and excludes pruned trials from completed history.
- `HyperparameterTuner(search_space=..., cache_transforms=True)` supports custom
  distributions and optional reuse of transforms fitted within training folds.
- Boruta adjusts its final binomial tests across features with Bonferroni by
  default; `multiple_testing="none"` restores unadjusted tests.
- Model wrappers reject conflicting native objectives and validate targets
  against the resolved objective even when `ModelData.objective` is absent.

Artifacts created before these fixes contain cloudpickled prediction closures and
may retain their original behavior. Refit old models when corrected numerical
semantics matter; loading cannot safely rewrite those closures in place.

## Quick Orientation

At a high level, the tool does this:

1. Load a policy-level parquet file into a `ModelData` object.
2. Validate basic target, exposure, weight, offset, fold, and comparison fields.
3. Optionally choose a raw-feature subset and configure encoding, selection, and
   preprocessing.
4. Fit encoding and selection, optionally tune on folds with the selected
   features, then fit preprocessing and the model on all training rows.
5. Evaluate an explicitly supplied holdout with the fitted artifacts.
6. Optionally export reports, persist the fitted pipeline, or combine fitted
   pipelines with blending or stacking.

The main package is:

```text
src/ins_gbm/
```

The package name is `ins_gbm`, not `gbm_fitting`.

The root README is a quick-start guide. Deeper implementation details are in
the source tree, tests, and this document.

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

The `explain` extra currently installs SHAP for downstream use; this package
does not currently expose a SHAP-specific API.

Common local pitfall: the path contains a space. Quote paths in shell commands
when needed.

The core requirements include Polars >= 1.0 and Optuna >= 4.0 for the categorical
mapping and process journal APIs. NumPy >= 1.26 remains supported; Gini integration
uses SciPy's `trapezoid` rather than the NumPy 2.0-only API.

## Package Map

```text
src/ins_gbm/
    __init__.py
    progress.py
    pipeline.py
    data/
        folds.py
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
        cv_importance.py
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
        cv_io.py
        io.py
        metadata.py
```

Common workflow objects are exported at package level:

```python
from ins_gbm import load_model_data, LightGBMModel, ModelRecipe
```

Deep imports remain available for specialized and internal-facing objects.

## Main Concepts

### Objective Types

The library supports two modeling objectives:

- `poisson`: policy-level frequency modeling. The target is a non-negative
  claim count or frequency response. `exposure` is optional; when omitted, no
  exposure offset, base margin, baseline, or exposure-derived sample weight is
  passed to the model.
- `gamma`: policy-level severity modeling. The target must be strictly positive.
  `weight` may be supplied, but exposure is not used as a severity offset.

Prediction scales use the `PredictionType` literal from `models/base.py`:

- `response`: expected claim count for Poisson when exposure is supplied;
  otherwise the model's unadjusted Poisson response. For Gamma it is expected
  severity.
- `rate`: expected claim count divided by exposure when exposure is supplied;
  otherwise the same unadjusted Poisson response. Invalid for Gamma and raises
  in `FittedModel.predict()`.
- `link`: `log(response)` for both objectives. For Poisson this includes
  `log(exposure)` when exposure is supplied.

### Polars Public API

The public data containers use Polars:

- feature matrices are `pl.DataFrame`
- target, exposure, weight, offset, and fold columns are `pl.Series`
- metrics and feature importance outputs are usually `pl.DataFrame`

Model wrappers convert to NumPy or framework-native matrix types at model
boundaries. Floating-point `ModelData` columns and model-fitting buffers use
`float32`; evaluation metrics use `float64` accumulation.

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
- `categorical`: unordered predictors available to an encoder or a model with
  native categorical support.
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

`ModelData` calls `infer_schema()` during construction whenever `schema=None`,
using its `features` and ordered `feature_names`. The parquet loader also
infers the schema when the caller does not provide one.

### `ModelData`

Defined in `data/model_data.py`.

`ModelData` is the central data container passed through almost every layer.
During construction, Float64 feature and row-level columns are downcast to
Float32. Integer and categorical columns retain their original dtypes.

```python
ModelData(
    features=pl.DataFrame(...),
    target=pl.Series(...),
    feature_names=["x1", "x2"],
    exposure=pl.Series(...) or None,
    weight=pl.Series(...) or None,
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
- `feature_names`: the ordered feature columns that should be used by models.
- `exposure`: optional Poisson exposure, defaulting to `None`. When absent, it
  remains `None` throughout model fitting and prediction.
- `weight`: optional observation weights, defaulting to `None`.
- `schema`: feature role metadata, mainly used by `OneHotEncoder`. If omitted,
  it is inferred automatically from supported Polars dtypes.
- `objective`: optional objective marker, usually `poisson` or `gamma`.

Additional fields:

- `offset`: optional numeric link-scale offset honored by the native GBM wrappers.
- `cv_fold`: optional integer fold ID series used by shared CV resolution.
- `comparisons`: optional DataFrame of external model predictions used for
  report comparisons.

Important methods:

- `n_rows`: number of feature rows.
- `validate()`: checks structural and objective-specific rules and returns
  `self`.
- `with_features(features)`: returns a copy with new features and updated
  `feature_names`.
- `select_features(feature_names)`: returns an ordered raw-feature subset,
  filters the schema to the same names, and preserves row-level fields.
- `with_offset(offset)`: returns a copy with a new offset series.
- `slice_model_data(data, indices)`: returns a row-sliced `ModelData`, including
  optional `offset`, `cv_fold`, and `comparisons` fields.

Validation rules currently implemented:

- `target`, `exposure`, and `weight` row counts must match `features`.
- `feature_names` must be unique.
- Every `feature_name` must exist in `features`.
- Target, exposure, weight, offset, and comparison values must be numeric where
  applicable, non-null, and finite.
- If `exposure` is supplied, it must be positive.
- Weights must be nonnegative and have positive total weight.
- Poisson objective requires non-negative target values; exposure is optional.
- Gamma objective requires strictly positive target values.
- `offset`, if supplied, must match row count.
- `cv_fold`, if supplied, must match row count, be integer, non-null, and have
  at least two unique values.
- `comparisons`, if supplied, must match row count, contain numeric columns, and
  every value must be finite and strictly positive.

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

The main pipeline uses the unfitted `OneHotEncoder` from `ModelRecipe`. It is
fit once on all supplied training rows before feature selection and tuning, and
the fitted encoder is reused for evaluation and prediction. A direct
`HyperparameterTuner.tune(...)` call may instead receive an encoder to refit on
each fold's training rows.

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

`OneHotEncoder.fit(features, schema)` requires a schema. Normally
`ModelData` supplies one through automatic inference. Provide an explicit
`FeatureSchema` when feature dtypes are unsupported or need ordinal or
passthrough roles that dtype inference cannot determine.

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
each split or fold. `ModelPipeline.run()` passes the supplied full-training
target during its final fit; `HyperparameterTuner` and ensemble fold helpers
pass only the fold-training target.

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
validation, OOF blending, and stacking. This preserves leakage isolation for
every step, including supervised reducers such as PLS.

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

### Cross-Validated Feature Importance

Defined in `selection/cv_importance.py` and exported as
`ins_gbm.cv_feature_importance`. This standalone screen fits one shallow model
on **the training rows** of each CV fold. The held-out rows choose the training
partition but are not used for fitting or importance calculation. The result is
a Polars DataFrame; it does not select a threshold or fit the final model.

The function accepts a `ModelData`, an optional model and `encoder`, optional
raw `feature_names`, `CVConfig`, framework-native `importance_types`, and
parameter overrides. `CVConfig(folds="auto")` uses `ModelData.cv_fold` if present
and otherwise creates shuffled folds. `folds="predefined"` requires stored fold
IDs; `CVConfig(folds="random", n_splits=5, seed=42)` forces random splits even
when `cv_fold` exists. The `feature_names` argument restricts the raw inputs
*before* encoding. When no encoder is supplied, these columns must already be
fit-ready numeric features and the result uses their names directly.

When `encoder=OneHotEncoder()` is supplied, a fresh encoder is fitted on each
fold's training rows, then its output is passed to the shallow model. A source
column such as `territory` is represented by rows such as
`territory__north` and `territory__south`; there is no separate `territory`
importance row. Numeric columns pass through with their original names. The
result contains the union of encoded columns seen across fold training sets.
If a level is absent from a particular training set, it contributes zero for
that fold. The final recipe fits its own encoder on all supplied training rows,
so use the same encoder configuration and input features for the final fit.

The result has `feature`, `n_folds_selected`, and one `mean_<type>` column for
each requested importance type. `n_folds_selected` counts a fold when **any**
requested type has importance strictly greater than zero. Each mean uses every
fold as its denominator; unused or absent columns contribute zero. Zero-score
columns remain in the result. Without an encoder, row order follows the input
columns; with an encoder, rows follow the order their output columns first
appear across fold fits. The default model is XGBoost and the default types are
`("weight", "gain", "cover")`, producing `mean_weight`, `mean_gain`, and
`mean_cover`. LightGBM supports `("split", "gain")`; CatBoost supports
`("FeatureImportance", "PredictionValuesChange", "LossFunctionChange")`;
Random Forest supports `("impurity",)`.

Built-in screening fits default to 50 trees of depth two (CatBoost uses 50
iterations; LightGBM also sets four leaves). `params` overrides these values
for the screening models only. It does not alter the final recipe's parameters.

To use the result, filter the `feature` column to an ordered, nonempty list.
Pass **raw names** to `recipe.fit(data, feature_names=raw_names)`, which uses
`feature_stage="raw"` by default and restricts the inputs before encoding.
Pass **encoded names** to the same recipe with `feature_stage="encoded"`:

```python
ranking = cv_feature_importance(
    training,
    model=LightGBMModel(),
    encoder=OneHotEncoder(),
    importance_types=("split", "gain"),
    cv=CVConfig(folds="predefined"),
)
encoded_names = ranking.filter(pl.col("n_folds_selected") >= 3)["feature"].to_list()
recipe = ModelRecipe(model=LightGBMModel(), encoder=OneHotEncoder())
fitted = recipe.fit(training, feature_names=encoded_names, feature_stage="encoded")
```

At `feature_stage="encoded"`, the recipe fits its encoder, checks that every
requested name exists, keeps exactly those encoded columns in the requested
order, and then fits preprocessing and the final model. A missing name raises
`ValueError`. This stage cannot be combined with `recipe.selection` because
the supplied names are already the fixed selection. Prediction still accepts
raw input data; the fitted pipeline replays the encoder and column selection.
With recipe tuning, this fixed encoded subset makes the tuning results
conditional on that selection. To estimate the performance of the *selection
procedure*, repeat the screen inside each outer CV training partition or use
a separate holdout that was excluded from screening and tuning.

### Boruta Selector

Defined in `selection/boruta.py`.

`BorutaSelector` implements a shadow-feature selection process:

1. For each iteration, shuffle every original feature to create shadow features.
2. Fit a base model on original plus shadow features.
3. Compare original feature importance to the configured shadow importance
   percentile (the maximum by default).
4. Count "hits" across iterations.
5. Use a final two-sided binomial test against hit probability 0.5, with
   Bonferroni adjustment across candidate columns by default, to classify features as:
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

The default screen uses 30 trees per iteration and the strongest shadow
(`base_n_estimators=30`, `shadow_percentile=100`). For a broader screen, set
`shadow_percentile=95` or `90`, and optionally increase
`base_n_estimators` so the base learner can use more features. A lower shadow
percentile admits more hits and can retain more features, including more noise.
`classification()` includes each feature's `hits` count for inspection. Changing
`alpha` alone is not a feature-count control: `selected_features()` already
includes tentative features, and a larger alpha can reject more of them.

`multiple_testing="none"` restores the previous unadjusted tests. The adjusted
tests can leave more features tentative; since tentative features are retained,
correction does not necessarily reduce the selected feature count. This is a
shadow-importance screening heuristic, especially with a reduced shadow
percentile, and does not guarantee an optimal subset or a false discovery rate.
Validate the complete selection recipe with outer CV or an independent holdout.
Invalid estimator names, iteration counts, significance levels, and tree counts
are rejected before fitting.

This selector fits the pipeline selector contract because `fit(data)` returns an
object with `selected_features()`.

Set `candidate_features=[...]` on `BorutaSelector` to limit its starting search
space. `candidate_stage="raw"` (the default) names raw inputs and restricts them
before encoding; a categorical name contributes all its encoded levels.
`candidate_stage="encoded"` names individual post-encoding columns and restricts
the selector after encoding. Features outside the candidate set do not enter the
final model. The `ModelData` passed to `recipe.fit()` or
`recipe.cross_validate()` is unchanged, so its folds and comparison predictions
remain available for CV and double-lift comparisons. Direct selector fits
expect data already at the named stage.

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

`StagedImportanceSelector` accepts the same `candidate_features` and
`candidate_stage` settings as Boruta. The candidate set applies before the
first importance stage; later stages operate on its survivors. Candidate lists
must be nonempty, unique, and present at the requested stage.

The usual pattern is a shallow, fast screening learner followed by a more
realistic tree configuration for final pruning. Stage learner parameters remain
fixed, and the main pipeline completes every stage before Optuna tunes only the
final recipe model on the fixed retained columns. OOF blending and stacking
continue to refit their recipe transform chains within their own folds.

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
- `percentile`: keep the top specified percentage of scores, including ties;
  for example, 25 retains the top quarter.
- `top_n`: keep the top N features.

Exactly one mode should be set. If none is set, the default threshold is `0.0`.
`top_n` must be a positive integer, `percentile` must be in [0, 100], and
`threshold` must be finite. Equal scores in top-N pruning retain their original
fitted feature order.

`ImportancePruner.fit()` accepts a fitted model or pipeline and ranks its fitted
model columns, including one-hot indicators and preprocessing outputs:

```python
selection = ImportancePruner(top_n=20).fit(fitted_model)
refit = recipe.fit(data, feature_names=selection)
```

The two-argument `fit(data, fitted_model)` call remains supported, but the
fitted model's column names determine the result. A fitted pruner result passed
as `feature_names` is applied after preprocessing. A plain list still defaults
to raw feature names; use `feature_stage="encoded"` or `"model"` to place a
plain list at a later stage.

Pipeline-compatible selector hooks call:

```python
selector.fit(current_train)
```

`ImportancePruner` remains a post-fit utility. For selection learned during
pipeline fitting, use `StagedImportanceSelector`.

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

- optional `objective`
- `fit(data, params=None, *, feature_names=None, encoder=None, preprocessing=None)`
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
- an optional fitted transform chain for direct model fits that supplied
  feature selection, encoding, or preprocessing.

The closures are why persistence uses `cloudpickle`.

Unfitted wrappers default `objective` to `None`. At fit time, the objective is
resolved in this order:

1. an objective explicitly supplied to the model wrapper;
2. `ModelData.objective`;
3. `poisson` as the legacy fallback when both are absent.

The resolved value is stored on `FittedModel`. Conflicting model/data objectives
are rejected, and the resolved objective validates the target even when the data
has no objective marker. Native `objective`/`loss_function` overrides must match
the wrapper's objective so its exponential response and log-link remain valid.

### LightGBMModel

Defined in `models/lightgbm.py`.

Supports:

- Poisson
- Gamma
- sample weights
- feature importance
- exposure offset for Poisson
- custom `ModelData.offset`
- native categorical features

Training behavior:

- Converts feature and row-level fitting data to `float32` NumPy.
- With `categorical_features="auto"`, uses surviving names from
  `ModelData.schema.categorical` and passes them to the native `Dataset`.
- Learns stable, consecutive category codes from the current training rows.
  Explicit names can be supplied for integer-coded categorical columns.
- Converts `_NUMERIC_FILL` to `np.nan`.
- For Poisson with exposure, uses `log(exposure)` as an initial score.
- If exposure and `data.offset` are both absent, omits `init_score` from the
  native `Dataset` rather than synthesizing unit exposure.
- If `data.offset` is present, adds it to the initial score.
- Uses `data.weight` as sample weight if supplied.
- Pops `n_estimators` from params and passes it as `num_boost_round`.
- If a row-sampling fraction is below one, defaults `bagging_freq` to 1 so
  sampling is effective. Explicit `bagging_freq` or `subsample_freq`, including
  zero, takes precedence.

Prediction behavior:

- Reuses the fitted category codes; missing and unseen levels use LightGBM's
  missing-value path.
- Converts `_NUMERIC_FILL` back to `np.nan`.
- Applies prediction-time `data.offset` if present.
- For Poisson:
  - `response` returns expected claim count.
  - `rate` returns expected rate.
  - `link` returns link-scale predictions.
  - when exposure is absent, no exposure adjustment is applied and `response`
    and `rate` return the same model-scale value.
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
- custom `ModelData.offset` for either objective

Training behavior:

- Converts feature and row-level fitting data to `float32` NumPy.
- Passes `_NUMERIC_FILL` as `missing` to `xgb.DMatrix`.
- For Poisson with exposure, uses `log(exposure)` as `base_margin`.
- Adds `data.offset` to the margin when supplied. If both exposure and offset
  are absent, omits training `base_margin` and lets XGBoost learn its intercept.
- Uses `data.weight` as sample weight if supplied.
- Pops `n_estimators` from params and passes it as `num_boost_round`.

Prediction behavior:

- If training used a margin, prediction supplies the new exposure/offset margin,
  or zero when neither is supplied. This prevents XGBoost's default intercept
  from being reintroduced when scoring without exposure.
- If training used no margin, prediction retains the learned intercept and adds
  the new exposure/offset on the link scale rather than replacing the intercept.
- `response` exponentiates the raw link prediction.
- `rate` divides response by exposure if exposure is present.
- `link` returns `log(response)`.
- For Gamma, `response` returns severity, `link` returns its log, and `rate`
  is rejected.

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

- Converts numeric features and row-level fitting data to `float32` NumPy.
- With `categorical_features="auto"`, passes surviving schema categorical
  columns as strings and specifies `cat_features` on training and prediction
  Pools. Explicit categorical feature names can include numeric-coded columns.
- Converts `_NUMERIC_FILL` to `np.nan` in numeric features; missing categorical
  values use the categorical missing-level string.
- Sets `loss_function` from the resolved objective and rejects conflicting
  overrides. A matching `objective` alias is normalized to `loss_function`.
- Uses `allow_writing_files=False` by default.
- For Poisson with exposure, uses `log(exposure)` as CatBoost baseline only if
  the installed CatBoost supports the `baseline` parameter.
- Adds `data.offset` to the baseline for either objective. If both exposure
  and offset are absent, omits `baseline` from the corresponding Pool.
- Uses `data.weight` as sample weight if supplied.
- Defaults to `LossFunctionChange`, defers it until requested, caches its scores,
  and releases the training Pool afterward. Serialization also materializes those scores to
  preserve data-free importance access in compact artifacts. The Pool is retained
  in memory until either operation. CV scoring and tuning do not request it.
- CatBoost's own `fit()` still computes `PredictionValuesChange` internally.

Pitfalls:

- CatBoost offset support depends on the installed CatBoost version.
- Exposure or explicit offsets raise an error if baseline support is unavailable.
- The Gamma implementation uses Tweedie power `1.99` because CatBoost requires
  the power to be strictly between 1 and 2.

### RandomForestModel

Defined in `models/random_forest.py`.

This is a benchmark model, not a true GLM-style frequency or severity wrapper.

Poisson behavior:

- If exposure is present, fits on claim rate: `target / exposure`.
- Uses exposure as sample weight, multiplied by `data.weight` when supplied.
- `response` multiplies predicted rate by exposure.
- `rate` returns predicted rate.
- `link` returns the log of the clipped expected response, including exposure.
- If exposure is absent, fits directly on the target and does not pass an
  exposure-derived `sample_weight`; a separate `data.weight` is still honored.

Gamma behavior:

- Fits directly on the untransformed target with MSE and optional `data.weight`.
- Returns predictions clipped to be positive.

Pitfalls:

- No native exposure offset.
- The wrapper passes numeric sentinel values through as real numbers rather
  than converting them to NaN; missing-value behavior depends on sklearn's
  version and the supplied criterion.
- Explicit `ModelData.offset` is rejected in fitting and prediction.
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
- `params`: optional base params; tuning suggestions override overlapping keys.
- `selection_scope`: `"fixed"` by default; `"fold"` refits encoding and learned
  selection inside each tuning fold.

Tuned suggestions override overlapping base params. To keep a parameter fixed,
omit it from the tuner's replacement `search_space` and supply it in `params`.

### Run Order

`recipe.fit(data)` delegates to `ModelPipeline.run()`, which executes in this order:

1. Optionally restrict the raw input to `feature_names`, or defer a fixed
   `feature_names` subset until after encoding with `feature_stage="encoded"`.
2. Fit the encoder and selector on all supplied training rows.
3. If tuning, fit preprocessors independently inside each tuning fold using the
   selected features. Merge trial suggestions over recipe params.
4. Fit preprocessing and the model on all supplied data.
5. Build reproducibility metadata and return `FittedPipeline`.

Call `FittedPipeline.evaluate(holdout_data)` to transform and evaluate a
caller-provided final holdout. That holdout is never used for tuning, selector
fitting, preprocessor fitting, or model fitting.

### Tuning Inside Pipeline

With `selection_scope="fold"`, pipeline tuning calls:

```python
self.recipe.tuning.tune(
    raw_train_data,
    self.recipe.model,
    encoder=self.recipe.encoder,
    selector=self.recipe.selection,
    preprocessors=self.recipe.preprocessing,
    base_params=self.recipe.params,
    progress=self.progress,
    should_stop=self.should_stop,
)
```

With the default `selection_scope="fixed"`, encoding and selection are
completed on all supplied training rows before tuning. The tuner receives the
selected features and refits preprocessing within each fold. Set
`selection_scope="fold"` to refit encoding and selection within each tuning fold.

When fitting with `recipe.fit(data, feature_names=...)`, that
ordered raw-feature subset is applied before encoding and selection. Tuning
then receives the encoded, final selected columns derived from that subset, and
the final full-data model uses the same selected columns. Direct tuner calls can
instead pass `feature_names` to `HyperparameterTuner.tune(...)`.

Use `feature_stage="encoded"` when `feature_names` already contains one-hot or
otherwise encoded column names. The encoder is fit first, the ordered subset is
stored as the pipeline's fixed final selection, and tuning and prediction reuse
that selection. This mode cannot be combined with `recipe.selection`.

```python
fixed_fit = recipe_without_selector.fit(
    data,
    feature_names=["age", "territory__urban", "territory__rural"],
    feature_stage="encoded",
)
```

Pitfalls:

- Preprocessors are an ordered transform chain, not concurrent jobs.
- `ImportancePruner` requires an already-fitted model and does not match the
  pipeline selector signature. Pass its fitted result as `feature_names` for a
  later fit, or use `StagedImportanceSelector` for in-pipeline importance pruning.

### FittedPipeline

`FittedPipeline` is the result object returned by `recipe.fit(data)`.

Important fields:

- `fitted_model`: the final `FittedModel`.
- `recipe`: the original `ModelRecipe` object, or a copy containing the new
  tuner when returned by `retune()`.
- `input_feature_names`: ordered raw inputs selected for this run.
- `raw_train_data`: optional selected raw training data retained in memory for
  OOF ensemble fits and omitted from compact persisted artifacts.
- `input_schema`: compact raw-feature schema retained for scoring.
- `train_data`: non-cached property that reconstructs the transformed training
  data only when explicitly accessed.
- `selected_features`: learned or fixed post-encoding feature names, if
  selection was used or `feature_stage="encoded"` was requested.
- `selection_results`: per-stage importance rankings and selected columns for a
  staged importance selector.
- `tuning_history`: Optuna history DataFrame, if tuning was used.
- `encoder`: fitted encoder, if used.
- `preprocessors`: fitted preprocessors.
- `metadata`: reproducibility metadata.

Important methods:

- `predict(data, prediction_type="response")`
- `predict_raw(features, exposure=None, weight=None, prediction_type="response")`
- `evaluate(holdout_data)`
- `retune(tuner, progress=None, should_stop=None)`

Use `result.predict(holdout_data)` for raw holdout data, or
`result.evaluate(holdout_data)` to produce metrics and plots.

`result.retune(tuner)` returns a new pipeline while leaving `result` unchanged.
It freezes the fitted encoder and exact selected columns, tunes with fold-local
preprocessing, then refits preprocessing and the model on all attached training
rows. Compact persisted pipelines must have their original training data
reattached through `load_pipeline(..., training_data=...)` before retuning.

```python
retuned = result.retune(
    HyperparameterTuner(n_trials=100, cv_folds=5, seed=42)
)
assert result.tuning_history is None
history = retuned.tuning_history
```

Use `result.predict(raw_model_data)` when you have a raw `ModelData` shaped like
the original pre-transform data.

Use `result.predict_raw(raw_features, exposure=...)` when scoring a feature
DataFrame without a real target column and an exposure offset is wanted. Omit
the argument to leave exposure absent.

Pitfalls:

- `FittedPipeline.predict()` applies the fitted transform chain. If an encoder
  was used, pass raw feature columns, not already encoded features.
- `FittedPipeline.predict_raw()` creates a placeholder target. When exposure is
  omitted, it remains `None` and the fitted wrapper receives no exposure
  offset, margin, baseline, or exposure-derived weight.
- Accessing `train_data` constructs transformed data; the result is not stored
  on `FittedPipeline`. A compactly loaded pipeline must have its original
  training data reattached first. Holdouts are never stored on the pipeline.

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
    n_jobs=4,
    backend="process",
    journal_path=None,
    show_progress_bar=True,
    search_space=None,       # use the model's default distributions
    cache_transforms=False, # opt in when all fold matrices fit in memory
)
```

Supported metrics:

- `poisson_deviance`
- `gamma_deviance`
- `rmse`
- `mae`

For a direct `HyperparameterTuner.tune(...)` call, each Optuna trial:

1. Draw params from `search_space` or `model.default_search_space()`, preserving
   distribution steps, and merge them over base params.
2. Use fold splits resolved once before the search: `cv=CVConfig(...)` controls
   the policy; otherwise `use_data_folds=None` uses stored folds when available,
   `False` forces shuffled KFold, and `True` requires `ModelData.cv_fold`.
3. Slice training and validation `ModelData`.
4. Fit encoder on fold training data and transform fold validation data.
5. Fit selector on fold training data and apply selected columns to validation.
6. Fit each preprocessor in the ordered chain on fold training features and
   transform both training and validation data.
7. Fit model on fold training data.
8. Predict validation response.
9. Score validation predictions and pool scores by validation row count or
   effective metric weight. Poisson deviance uses rate-scale inputs and
   exposure times observation weight. RMSE pools squared errors before the root.
10. Report the pooled score so far to Optuna's MedianPruner.

With the default `selection_scope="fixed"`, `ModelPipeline.run()` has
already fitted the encoder and completed feature selection, so it passes the
fixed selected `ModelData` without an encoder or selector. In that path, steps
4 and 5 above are already complete, and only the preprocessing chain is refit
inside each tuning fold.

Supervised fixed selection has seen inner validation targets, so that tuning
score can be optimistic. Use `selection_scope="fold"` to refit encoding and
selection within each inner training fold. Outer CV refits the complete recipe
on each outer training partition. Lists derived from all rows outside the recipe
also require separate evaluation data.

With `cache_transforms=True`, steps 3–6 run once per fold before search and the
prepared matrices are reused across trials. This freezes stochastic transforms
and preserves training/validation separation. Caching is disabled by default
because all fold matrices stay in memory and each process receives its own copy.

The tuner returns:

```python
best_params, history = tuner.tune(
    data,
    model,
    feature_names=["x1", "x3"],
)
```

`feature_names` is an ordered raw-feature subset applied before any fold-local
encoder, selector, or preprocessor. An explicit encoder schema is restricted
to the same subset.

`n_jobs` controls concurrent Optuna trials. The default is `1`; use `-1` for
all available CPUs. `backend="thread"` is the backward-compatible default and
passes `n_jobs` directly to Optuna. `backend="process"` launches independent
Python processes coordinated through
`JournalStorage(JournalFileBackend(...))`, bypassing the Python GIL for
CPU-bound Python work.

The process backend uses an internal temporary journal unless `journal_path`
is supplied. An explicit path preserves the journal for inspection; each
`tune()` call still creates a new uniquely named study rather than resuming a
previous one. Journal file locking supports multiple workers on the same host
and local filesystem.

On Windows, journal files automatically use `JournalFileOpenLock` rather than
Optuna's default symbolic-link lock. This avoids the Windows "A required
privilege is not held by the client" error without requiring Developer Mode or
administrator privileges. Other platforms retain Optuna's default lock.

Process tuning can be called directly from a Jupyter notebook cell. Workers
start through an importable `ins_gbm` module, and the tuning payload is
serialized with cloudpickle, so the notebook does not need an
`if __name__ == "__main__"` guard. User-defined models and transforms must be
cloudpickle-serializable.

`show_progress_bar` displays completed trials and defaults to `True`. Set it to
`False` for quiet batch runs.

`history` has one row per successfully completed trial and includes:

- `trial`
- `value`
- one column per tuned hyperparameter

Pruned and failed trials are excluded. Trial counts must be positive integers;
metrics must return finite scores. Gini is an evaluation metric, not a supported
tuning metric: the tuner minimizes the four error/deviance metrics listed above.

Pitfalls:

- Parallel TPE trial scheduling can produce different suggestions between
  runs even with the same seed.
- Avoid CPU oversubscription: if `n_jobs` runs several trials concurrently,
  set `num_threads` (LightGBM), `nthread` (XGBoost), `thread_count` (CatBoost),
  or `n_jobs` (Random Forest) in the model's base params.
- Each process holds its own copy of the tuning data, and each concurrent trial
  holds its fold data and fitted transforms, so increase `n_jobs`
  conservatively on very large datasets.

### Search Spaces

Every model wrapper exposes `default_search_space()` using Optuna distribution
objects.

`tuning/search_spaces.py` provides:

```python
narrow_search_space(space, **overrides)
```

This returns a copied search space with selected distributions replaced.

Pass the returned dictionary as `HyperparameterTuner(search_space=space)`.
This replaces the complete default space; omitted parameters may be fixed in
`ModelRecipe.params`. An empty space evaluates only those base params.

```python
from optuna.distributions import IntDistribution
from ins_gbm.tuning.search_spaces import narrow_search_space

space = narrow_search_space(
    LightGBMModel().default_search_space(),
    n_estimators=IntDistribution(100, 400, step=50),
)
tuner = HyperparameterTuner(search_space=space, cache_transforms=True)
```

The default spaces and trial budgets are starting points rather than guarantees
of optimal settings. GPU modes and special bootstrap/boosting types may need a
different space. Compare complete recipes on outer CV or a separate holdout.

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
- `double_lift_table(actual, predicted_a, predicted_b, weights=None, n_bins=10)`
- `double_lift_score(dl_table, deviation="absolute")`
- `compute_metrics(...)`

`compute_metrics()` returns a DataFrame with:

- objective-specific deviance
- gini
- rmse
- mae

For Poisson with exposure, exposure is used as the deviance and Gini weight;
when a separate model weight is present, the effective weight is
`exposure * weight`. Without exposure, metrics use the supplied actual and
predicted values directly and use only `weight` when present. For Gamma,
`weight` is used directly.

Report-level double-lift calculations compare Poisson rates using
`exposure * weight` as the effective weight when both are present. Without
exposure, they compare the supplied values directly and use only the separate
model weight when present. A positive `double_lift_score` favors model 2; a
negative score favors model 1.

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
- `double_lift_score(name, other_name=None, n_bins=10, deviation="absolute")`
- `export(output_dir)`

`plot_double_lift()` also works in named-model comparison mode. Pass
`name`/`other_name`, or omit both when the report contains exactly two models.
Comparison metrics include `double_lift_score`; positive values favor the
second model and negative values favor the first.

For a single-model report, `export(output_dir)` writes:

- `metrics.csv`
- `lift.png`
- `ave.png`
- `calibration.png`
- `feature_importance.png`
- double-lift plots when comparison predictions exist

Named-model comparison reports write `metrics.csv` plus one double-lift image
for every model pair; they do not write the single-model lift, calibration, or
feature-importance images.

Named-model double lift requires aligned evaluation targets, exposures, and
weights with the same objective.

External comparison predictions can come from `ModelData.comparisons`. The
caller-provided holdout supplies those columns to `EvaluationReport` as named
prediction series.

### CrossValidationReport

Defined in `evaluation/cv_report.py`.

`recipe.cross_validate()` runs repeated fold fits and reports fold stability:

```python
from ins_gbm import CVConfig, LightGBMModel, ModelRecipe

recipe = ModelRecipe(model=LightGBMModel(objective="poisson"))
result = recipe.cross_validate(
    data, cv=CVConfig(n_splits=5, seed=42, folds="auto"),
    feature_names=["x1", "x3"],
)
```

Returns `CVResult`:

- `fold_metrics`: columns `fold`, `model`, `metric`, `value`.
- `summary`: columns `model`, `metric`, `mean`, `std`.
- `fold_col`: name of the predefined fold field when one was used.
- `predictions`: out-of-fold predictions used for comparison charts.
- `row_folds`: the fold assignment for each prediction row.
- `actual`, `exposure`, `weight`, and `objective`: aligned row-level context for
  in-memory out-of-fold predictions. Target, exposure, and weight are omitted
  when the result is saved.
- `feature_names`: the ordered runtime feature subset.
- `cv_config`, `data_signature`, and `fold_params`: effective CV settings,
  an ordered evaluation-input fingerprint, and fitted parameters by fold.

`cross_validate(..., feature_names=...)` selects raw predictors without removing a
`benchmark_col` or `fold_col` from its special role. Pass
`feature_stage="encoded"` to select named columns after each fold fits its
encoder. In benchmark mode,
`CVResult.plot_double_lift()` plots the OOF GBM and benchmark predictions,
`CVResult.double_lift_score()` returns their signed score, and fold/summary
metrics include `double_lift_score`.

Use `CrossValidationReport(...)` directly when `benchmark_col`, `fold_col`, or
`show_progress_bar` needs explicit configuration.

Fold modes are configured with `CVConfig`: `auto` uses `ModelData.cv_fold` when
present, `random` forces shuffled folds, and `predefined` requires `cv_fold`.
The legacy `fold_col` adapter remains supported.

Benchmark mode:

- `benchmark_col` can name a positive prediction column in `data.features`.
- The benchmark column is dropped before fitting the GBM recipe.
- Fold metrics include both `gbm` and `benchmark` rows.

`CrossValidationReport` fits the complete recipe. When tuning is configured,
each outer training partition performs inner tuning.

Save a CV report explicitly with `result.save("output/model/cv_report")` and
reload it with `load_cv_result(...)`. The report stores metrics, aligned OOF
predictions, fold assignments, and provenance, but no targets, weights, or
features. A saved fitted model alone cannot reconstruct its CV predictions.

### Report Comparison

Defined in `evaluation/comparison.py`.

`compare_reports()` compares multiple `EvaluationReport` or `CVResult` objects:

```python
table = compare_reports({
    "lightgbm": result_lgb.evaluate(lightgbm_holdout),
    "xgboost": result_xgb.evaluate(xgboost_holdout),
})
```

The output has one row per metric, one column per report name, and a
`preferred` column. Direction for standalone metrics comes from
`METRIC_DIRECTIONS`:

- higher is better for `gini`.
- lower is better for deviance, RMSE, and MAE.

Single evaluation values are formatted to four decimals. CV values are
formatted as `mean +/- std`.

When a single-model `EvaluationReport` also contains external comparison
predictions, `compare_reports()` compares the fitted model's metrics rather
than the benchmark rows.

For exactly two aligned CV results, `compare_reports()` adds a pooled
`double_lift_score` row. The signed value appears under the second report,
with the first report blank; positive favors the second report, negative favors
the first, and zero is a tie. Results must come from the same rows in the same
order and use the same folds:

```python
from ins_gbm import compare_cv_double_lift, compare_reports, load_cv_result

saved = load_cv_result("output/baseline/cv_report")
candidate = candidate_recipe.cross_validate(data, cv=cv)
metrics = compare_reports({"baseline": saved, "candidate": candidate})
fold_details = compare_cv_double_lift(saved, candidate)
```

`fold_details` includes the pooled score and one score per fold. If both reports
were loaded from disk, supply the original rows with
`compare_reports({"baseline": saved, "candidate": candidate}, data=data)` or
`compare_cv_double_lift(saved, candidate, data=data)`. Without evaluation data,
the standard metrics still appear but the double-lift row is omitted. The row
is also omitted for unaligned CV reports or more than two reports. The pairwise
calculation checks a fingerprint of ordered targets, exposure, weights, and
fold assignments. The pooled score uses all OOF rows, not an average of folds.

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
- Fixed OOF refits preserve each base pipeline's effective fitted parameters;
  `refit="retune"` performs tuning inside each fold instead.
- Ensemble evaluation uses only the caller-provided holdout and does not retain
  training data on the resulting `EvaluationReport`.

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
- `base_pipelines`: base fitted pipelines.

`EnsembleResult.predict(data)` scores raw data. Call
`EnsembleResult.evaluate(holdout_data)` to create an `EvaluationReport`; the
method builds a proxy `FittedModel` so the standard report machinery can call
the ensemble.

## Persistence

Persistence lives in `src/ins_gbm/persistence/`.

### Saving

Model artifacts are implemented in `persistence/io.py`.

```python
from ins_gbm import load_model

result.save("output/my_model")
loaded = load_model("output/my_model")
```

Artifacts written:

- `pipeline.pkl`: fitted pipeline via `cloudpickle`, without raw training rows.
- `metadata.json`: `ReproducibilityMetadata` as JSON.
- `tuning_history.parquet`: only when tuning history exists.

CV reports are independent artifacts implemented in `persistence/cv_io.py`:

```python
from ins_gbm import load_cv_result

cv_result.save("output/my_model/cv_report")
saved_cv = load_cv_result("output/my_model/cv_report")
```

The CV directory contains `fold_metrics.parquet`, `summary.parquet`,
`predictions.parquet`, and `metadata.json`. The predictions file has a row per
original training row, with OOF predictions and its fold ID. The metadata holds
the objective, features, fold settings, fold parameters, row count, and an
evaluation-input fingerprint. Training features, targets, exposure, and weights
remain outside the CV artifact.

### Loading

```python
from ins_gbm import load_model

# Reattach the original rows only for train_data or OOF ensemble fitting.
loaded_for_oof = load_model(
    "output/my_model",
    training_data=original_training_data,
)
```

Pitfalls:

- Standard `pickle` and `joblib` are not used because `FittedModel` contains
  local prediction and importance closures.
- Prediction, evaluation, and feature importance do not require training data.
  Reattached rows must be the original training dataset in its original order.
- A fitted UMAP reducer may retain training-derived matrices that its transform
  implementation needs even though `raw_train_data` itself is omitted.
- Loading model pickles is safest in an environment with compatible package versions.
- `metadata.json` is useful for auditing but does not recreate the pipeline by
  itself.

### Metadata

Defined in `persistence/metadata.py`.

`ReproducibilityMetadata` records:

- package versions
- supplied split/tuning seeds
- model params
- fitted feature names
- raw input feature names
- selected features
- staged-selection summaries, when present
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

Pipeline-emitted stages are:

- `encode`
- `select`
- `tuning`
- `preprocess`
- `fit`

The tuner emits `tuning` events when a progress callback is provided. The
pipeline does not currently emit `split` or `evaluate` events.

## Common Usage Flows

### Basic Poisson Frequency Pipeline

```python
from ins_gbm.data.loader import load_model_data
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm import ModelRecipe

data = load_model_data(
    path="frequency.parquet",
    target="claim_count",
    exposure="exposure",
    feature_cols=["x1", "x3"],
    objective="poisson",
)

recipe = ModelRecipe(
    model=LightGBMModel(),  # inherits objective="poisson" from data
    params={"n_estimators": 100},
)

result = recipe.fit(data)

report = result.evaluate(holdout_data)
metrics = report.metrics()
preds = result.predict(holdout_data, prediction_type="response")
```

### Basic Gamma Severity Pipeline

```python
from ins_gbm.data.loader import load_model_data
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm import ModelRecipe

data = load_model_data(
    path="severity.parquet",
    target="severity",
    weight="weight",
    feature_cols=["x1"],
    objective="gamma",
)

recipe = ModelRecipe(model=LightGBMModel())  # inherits "gamma" from data
result = recipe.fit(data)
```

### Native Categorical Features in LightGBM

```python
from ins_gbm.data.schema import FeatureSchema
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
)
```

The schema is inferred automatically for string, categorical, enum, and boolean
columns when it is omitted. For integer-coded categories, use
`LightGBMModel(categorical_features=["territory_code"])`. Add
`encoder=OneHotEncoder()` when individual levels must become model columns or
when sharing an encoded matrix with a model that lacks native categorical
support.

### Optuna Tuning

```python
from ins_gbm.preprocessing.encoder import OneHotEncoder
from ins_gbm.selection.boruta import BorutaSelector
from ins_gbm.tuning.tuner import HyperparameterTuner

recipe = ModelRecipe(
    model=LightGBMModel(objective="poisson"),
    encoder=OneHotEncoder(),
    selection=BorutaSelector(
        base_estimator="lightgbm",
        max_iter=50,
        seed=42,
        candidate_features=["x1", "x3"],
    ),
    selection_scope="fold",
    params={"num_threads": 1},
    tuning=HyperparameterTuner(
        n_trials=25,
        cv_folds=5,
        metric="poisson_deviance",
        seed=42,
        n_jobs=4,
        cache_transforms=True,
    ),
)

result = recipe.fit(data)
history = result.tuning_history
```

For individual one-hot levels, use
`candidate_features=["x1", "territory__north"]` with
`candidate_stage="encoded"`. Pass the same full `data` to CV reporting; the
selector applies its candidate limit within each fit.

This fit relearns the encoder, selector, and preprocessing inside tuning folds.
It then fits them on all training rows for the returned model.

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
from ins_gbm import CVConfig, LightGBMModel, ModelRecipe

recipe = ModelRecipe(model=LightGBMModel(objective="poisson"))
cv_result = recipe.cross_validate(
    data, cv=CVConfig(n_splits=5, seed=42, folds="auto"),
    feature_names=["x1", "x3"],
)

fold_metrics = cv_result.fold_metrics
summary = cv_result.summary
cv_result.save("output/frequency_model/cv_report")
```

### Saving and Loading a Pipeline

```python
from ins_gbm import load_model

result.save("output/frequency_model")
loaded = load_model("output/frequency_model")
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
- `tests/selection/`: Boruta, CV importance screening, standalone importance
  pruning, and staged importance selection.
- `tests/evaluation/`: metrics, plots, reports, CV reports, comparison helpers.
- `tests/ensemble/`: blending, stacking, and ensemble pipeline.
- `tests/persistence/`: save/load behavior.
- `tests/test_pipeline.py`: main pipeline behavior.
- `tests/test_integration.py`: end-to-end pipeline and explicit-evaluation
  flows.
- `tests/test_progress.py`: progress callbacks and cancellation.

The fixtures in `tests/conftest.py` generate synthetic Poisson and Gamma data:

- `poisson_raw`: 400 rows with `x1`, `x2`, `x3`, `exposure`, and `claim_count`.
- `gamma_raw`: 300 rows with `x1`, `x2`, `severity`, and `weight`.
- parquet fixtures write those frames to temporary files.

## Extension Points

### Adding a Model Wrapper

Create a module under `src/ins_gbm/models/` that implements the `BaseModel`
protocol:

1. Add an unfitted dataclass with
   `objective: Optional[Objective] = None`, and resolve it from the model
   override, `ModelData.objective`, then the legacy Poisson fallback.
2. Implement `capabilities()`.
3. Implement `default_search_space()`.
4. Implement the `fit(...)` signature from `BaseModel`, including optional
   runtime features and fit-time transforms.
5. Convert `data.features.select(data.feature_names)` to the model's expected
   matrix format.
6. Return a `FittedModel` with `predict_fn`, `importance_fn`, and any fitted
   transform chain.
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

### Finding Public Imports

The root `ins_gbm` package exports the common fitting, evaluation, CV report,
and persistence functions used in the examples. More specialized helpers remain
available through their concrete modules.

### Treating `FittedPipeline.train_data` as Raw Data

`FittedPipeline.train_data` reconstructs a transformed model-ready frame on
each access. If an encoder, selector, or reducer was used, it is not the raw
parquet frame and callers should avoid retaining it when memory is constrained.
Use `raw_train_data` for the shared raw reference.

### Passing Transformed Data to `FittedPipeline.predict()`

`FittedPipeline.predict()` applies the transform chain. If the pipeline has an
encoder, pass raw columns. For already transformed data, call
`result.fitted_model.predict(transformed_data)` instead.

### Omitting Exposure for Poisson

Poisson `ModelData.validate()` allows `exposure=None`. The wrappers keep it
absent: LightGBM receives no exposure-derived initial score, XGBoost no base
margin, CatBoost no baseline, and Random Forest no exposure-derived sample
weight. No vector or scalar of ones is synthesized. Supply exposure only when
the model should include that exposure adjustment.

### Using Gamma Rate Predictions

`prediction_type="rate"` is invalid for Gamma and raises.

### Fold metadata

Store predefined folds in `ModelData.cv_fold` and use `CVConfig`. The older
feature-column `fold_col` argument is a compatibility adapter only.

### Managing Holdouts

The caller owns holdout construction. Keep final holdout rows separate from
the `ModelData` passed to `recipe.fit()` and pass them only to
`FittedPipeline.evaluate()`.

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

### Offsets and Random Forest

LightGBM, XGBoost, and compatible CatBoost versions apply `ModelData.offset` on
the link scale. Random Forest rejects offsets because it cannot honor that model.

### Assuming Optional Dependencies Are Installed

LightGBM, XGBoost, CatBoost, and UMAP are optional extras. Install the right
extras before using wrappers or reducers that need them. SHAP is also declared
as an optional dependency for downstream explainability work, but there is no
package-level SHAP workflow at present.

### Mixing NaN and Null Missing Values

The encoder fills Polars nulls. Floating `NaN` values are different and should
be handled explicitly upstream if present.

### Treating Random Forest as a True Poisson or Gamma Objective

`RandomForestModel` is a benchmark. With exposure, its Poisson behavior fits
rates with exposure weights; without exposure, it fits the target directly. It
does not optimize a Poisson likelihood with a native log exposure offset.

### Optimizing Blends on the Final Holdout

The code supports validation-mode blending but cannot know whether the
validation data is actually the final evaluation holdout. The caller is
responsible for keeping that holdout untouched.

### Assuming Ensemble Inputs Are Validated Deeply

The ensemble code expects fitted pipelines to be compatible. Make sure base
pipelines share the same objective, row basis, and intended evaluation
holdout.

### Ensemble OOF Refits

Stacking and OOF blending refit learned transforms inside folds and pass the
base pipeline's effective fitted parameters to each fold model by default.
Set `refit="retune"` to run a recipe's tuner inside every outer ensemble fold;
this is substantially more expensive but evaluates tuning without global params.

### Persisting Across Incompatible Environments

Saved pipelines use `cloudpickle`. They are convenient for same-project
round-trips, but they are not a stable interchange format across major library
or Python version changes.

## Mental Model for New Contributors

The main invariant to preserve is final-holdout isolation:

- fit the main pipeline encoder and selector only on the supplied training
  data, never on the final holdout.
- outer CV fits encoding, learned selection, and reducers on each outer training
  split. Inner tuning refits reducers within training folds; encoding and
  selection are fixed by default, or fold-local with `selection_scope="fold"`.
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

- Poisson means a non-negative target and optional positive exposure.
- Gamma means strictly positive severity target.
- rate predictions are Poisson-only.
- deviance metrics require positive predictions.

If a change touches one of these invariants, add or update tests around the full
flow, not just the individual function.
