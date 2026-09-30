# GBM Fitting

Polars-native Python tools for insurance frequency and severity GBM modeling.

## Install and test

```bash
uv sync --all-extras --dev
uv run python -m pytest
```

Recreate `.venv` after a system Python upgrade rather than reusing packages from
an older interpreter. The package is named `ins_gbm`; do not rename it to
`gbm_fitting` because that name conflicts with another installed package.

## Fit, evaluate, predict, and save

Keep the final holdout separate from every fitting, selection, and tuning step.

```python
from ins_gbm import (
    LightGBMModel, ModelRecipe, OneHotEncoder, load_model, load_model_data,
)

training = load_model_data(
    "training.parquet", target="claim_count", exposure="exposure",
    objective="poisson",
)
holdout = load_model_data(
    "holdout.parquet", target="claim_count", exposure="exposure",
    objective="poisson",
)

recipe = ModelRecipe(model=LightGBMModel(), encoder=OneHotEncoder())
fitted = recipe.fit(training)
metrics = fitted.evaluate(holdout).metrics()
counts = fitted.predict(holdout)
rates = fitted.predict(
    holdout.features, exposure=holdout.exposure, prediction_type="rate",
)

fitted.save("output/frequency_model")
restored = load_model("output/frequency_model")
```

`ModelPipeline(data, recipe).run()` and the original persistence functions remain
supported. The shorter methods delegate to the same implementation.

## Prune fitted model columns

`ImportancePruner` ranks the columns actually fitted by a model, including
one-hot indicators and preprocessing outputs. Pass a fitted model or pipeline;
the raw training data is not needed for pruning. Pass the fitted pruning result
as `feature_names` to apply it after the new recipe's transforms:

```python
from ins_gbm.selection import ImportancePruner

selection = ImportancePruner(top_n=20).fit(fitted)
pruned = recipe.fit(training, feature_names=selection)
```

The new recipe must produce the selected model columns. Plain `feature_names`
lists still refer to raw inputs unless `feature_stage="encoded"` or
`feature_stage="model"` is specified.

## Start selection from a smaller candidate set

Set candidates on `BorutaSelector` or `StagedImportanceSelector` while passing
the original `ModelData` to fitting and CV. Raw candidates limit encoding and
selection; each categorical raw feature includes all its one-hot levels. Encoded
candidates limit selection after encoding and can name individual levels.
Features outside the candidate set are excluded from the final model.

```python
from ins_gbm import BorutaSelector

raw_recipe = ModelRecipe(
    model=LightGBMModel(), encoder=OneHotEncoder(),
    selection=BorutaSelector(candidate_features=["driver_age", "territory"]),
)
encoded_recipe = ModelRecipe(
    model=LightGBMModel(), encoder=OneHotEncoder(),
    selection=BorutaSelector(
        candidate_features=["driver_age", "territory__north"],
        candidate_stage="encoded",
    ),
)
raw_fit = raw_recipe.fit(training)
encoded_cv = encoded_recipe.cross_validate(training)
```

The same settings work on `StagedImportanceSelector`. Direct selector fits
expect data at the stage named by `candidate_stage`. Candidates must be present
in that stage. The supplied `ModelData`, including its comparison predictions
and fold assignments, remains available for CV and double-lift reporting.

Boruta's default threshold is the strongest shadow feature in each iteration.
For a less strict screen, set `shadow_percentile=95` (or `90` for a broader
screen). `base_n_estimators` controls the trees in each Boruta fit; its default
is 30. Inspect `BorutaSelector(...).fit(encoded_train).classification()` for
feature statuses and hit counts. Increasing `alpha` can reject more features;
it does not make `selected_features()` longer because that method already
includes tentative features.

## Screen features across CV folds

`cv_feature_importance` fits a shallow XGBoost model on the training rows of
each fold and returns every input feature, including those with zero importance.
Its default columns are `feature`, `n_folds_selected`, `mean_weight`,
`mean_gain`, and `mean_cover`. The count is the number of folds where at least
one requested importance measure is positive; averages include zero scores.

```python
import polars as pl
from ins_gbm import CVConfig, cv_feature_importance

# Uses training.cv_fold if present, otherwise five shuffled folds.
ranking = cv_feature_importance(training)
ranking = cv_feature_importance(
    training,
    cv=CVConfig(folds="random", n_splits=10, seed=42),
    feature_names=["driver_age", "vehicle_age", "annual_miles"],
    importance_types=("weight", "gain", "cover", "total_gain"),
)
candidates = ranking.filter(pl.col("n_folds_selected") >= 7)["feature"].to_list()
final_fit = recipe.fit(training, feature_names=candidates)
```

Without an encoder, the supplied features must be fit-ready numeric columns.
For another model wrapper, pass its supported `importance_types`; for example,
LightGBM supports `("split", "gain")`. `params` can override the shallow
fitting defaults.

To rank one-hot levels separately, supply the same encoder used by the final
recipe. It is fitted within each CV training fold. The returned names then
include levels such as `territory__north`:

```python
ranking = cv_feature_importance(
    training,
    model=LightGBMModel(),
    encoder=OneHotEncoder(),
    importance_types=("split", "gain"),
)
selected = ranking.filter(pl.col("n_folds_selected") >= 3)["feature"].to_list()
fitted = recipe.fit(training, feature_names=selected, feature_stage="encoded")
```

Plain raw feature names use `recipe.fit(training, feature_names=selected)`.
Encoded names require `feature_stage="encoded"`; the final recipe must produce
every selected encoded column.

## Tune and cross-validate

```python
from ins_gbm import (
    CVConfig, HyperparameterTuner, compare_cv_double_lift,
    compare_reports, load_cv_result,
)

recipe.tuning = HyperparameterTuner(n_trials=30, cv_folds=5, seed=42)
fitted = recipe.fit(training)
cv_result = recipe.cross_validate(
    training, cv=CVConfig(n_splits=5, seed=42, folds="auto"),
)
cv_result.save("output/frequency_model/cv_report")

# Later, run a candidate on the same rows, in the same order and folds.
saved_cv = load_cv_result("output/frequency_model/cv_report")
candidate_recipe = ModelRecipe(
    model=LightGBMModel(), encoder=OneHotEncoder(), params={"num_leaves": 16},
)
candidate_cv = candidate_recipe.cross_validate(
    training, cv=CVConfig(n_splits=5, seed=42, folds="auto"),
)
metric_comparison = compare_reports({"saved": saved_cv, "candidate": candidate_cv})
# metric_comparison includes a pooled double_lift_score row.
double_lift = compare_cv_double_lift(saved_cv, candidate_cv)
# double_lift also shows each fold; positive scores favor the candidate.
```

`auto` uses `ModelData.cv_fold` when present and otherwise uses shuffled K-fold
splits. By default, encoding and feature selection run once on the supplied
training rows before tuning. Tuning evaluates the selected features across CV
folds, refitting preprocessing within each fold. Set `selection_scope="fold"`
to refit encoding and selection within each tuning fold. Outer CV of a tuned
recipe is nested CV.

The CV artifact stores fold metrics, out-of-fold predictions, fold assignments,
and provenance. It does not store targets, exposure, weights, or features. If both
CV results are loaded from disk, pass the original training data to
`compare_reports({"saved": saved_cv, "candidate": candidate_cv}, data=training)`
or pass `data=training` to `compare_cv_double_lift()`. Results must use the same
ordered rows and fold assignments. With two aligned CV reports, `compare_reports()`
adds a pooled `double_lift_score` row under the second report's column; positive
values favor that report. It omits the row for unaligned reports or when both
reports are loaded without evaluation data. `compare_cv_double_lift()` also
returns scores for each fold. The pooled score uses all out-of-fold rows rather
than averaging fold scores.

`HyperparameterTuner(metric=None)` infers Poisson or Gamma deviance. Recipe params
are base parameters; trial suggestions override overlapping keys.

## Prediction contract

- Poisson `response` is expected count; `rate` is count per unit exposure.
- Gamma `response` is expected severity and does not support `rate`.
- For both objectives, `link` is `log(response)`.
- Explicit offsets add on the link scale. Poisson combines offset and
  `log(exposure)`.
- Random Forest is an approximation benchmark and rejects explicit offsets.

Pass `feature_names=[...]` to `recipe.fit()` or `recipe.cross_validate()` to reuse
one loaded candidate pool across fits. To select encoded columns such as
`"territory__urban"`, pass `feature_stage="encoded"` with `feature_names`.
Each cross-validation fold fits its own encoder before selecting those columns.

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for architecture and migration
details. The runnable workflow is `examples/example_usage.py`, with a companion
notebook at `examples/example_usage.ipynb`.
