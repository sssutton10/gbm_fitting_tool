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

Use a rate target and leave `exposure` empty (`None`, or `null` in JSON) by
default, across model fitting, tuning, CV, evaluation, and ensembles. The tool
fits the supplied rates directly; it does not infer exposure or convert them
to counts. Supply observation `weight` separately when wanted.

Keep the final holdout separate from every fitting, selection, and tuning step.

```python
from ins_gbm import (
    LightGBMModel, ModelRecipe, OneHotEncoder, load_model, load_model_data,
)

training = load_model_data(
    "training.parquet", target="claim_rate",
    objective="poisson",
)
holdout = load_model_data(
    "holdout.parquet", target="claim_rate",
    objective="poisson",
)

recipe = ModelRecipe(model=LightGBMModel())
fitted = recipe.fit(training)
metrics = fitted.evaluate(holdout).metrics()
rates = fitted.predict(holdout)
raw_rates = fitted.predict(holdout.features)

fitted.save("output/frequency_model")
restored = load_model("output/frequency_model")
```

`ModelPipeline(data, recipe).run()` and the original persistence functions remain
supported. The shorter methods delegate to the same implementation.

## Native categorical features in LightGBM

`LightGBMModel` uses LightGBM's native categorical splits by default. With
`categorical_features="auto"`, it takes categorical names from
`ModelData.schema`. String, categorical, enum, and boolean columns are assigned
to that schema automatically. Category codes are learned only from each fit's
training rows, including within cross-validation, and are reused for prediction.
Missing and previously unseen levels follow LightGBM's missing-value path.

For an integer-coded category that schema inference would otherwise classify as
numeric, name it explicitly:

```python
recipe = ModelRecipe(
    model=LightGBMModel(categorical_features=["territory_code", "vehicle_code"])
)
```

Parameters such as `cat_smooth`, `cat_l2`, `max_cat_threshold`, and
`max_cat_to_onehot` can be supplied through `ModelRecipe.params` or directly to
`LightGBMModel.fit(..., params=...)`. Set categorical names on
`LightGBMModel(categorical_features=...)`, rather than in `params`.

`CatBoostModel` also handles schema categorical columns natively by default.
It passes string, categorical, enum, and boolean features to CatBoost as
categorical values in both training and prediction Pools. To treat a numeric
code as a category, set `CatBoostModel(categorical_features=["territory_code"])`.
As with LightGBM, an explicit list replaces automatic selection; include every
column you want treated as categorical. Set names on the model rather than
passing `cat_features` in `params`.

`OneHotEncoder` remains available when the same feature matrix must work across
several model families, or when selection and importance should operate on
individual category levels:

```python
one_hot_recipe = ModelRecipe(
    model=LightGBMModel(),
    encoder=OneHotEncoder(),
)
```

## CatBoost tuning on wide feature sets

`CatBoostModel.default_search_space()` is intended for CPU tuning. It searches
200–1,200 iterations in steps of 100, learning rates of 0.03–0.2 (log scale),
depths of 3–8, `l2_leaf_reg` of 0.1–100 (log scale), `subsample` of 0.5–1.0,
and `colsample_bylevel` of 0.1–1.0 (log scale). Lower feature-sampling fractions
help control the cost of fitting hundreds of predictors; the depth cap avoids
the largest symmetric trees. These are starting ranges, not a guarantee that
CatBoost will outperform another model family. Custom tuner search spaces and
the model-search example's tree/depth overrides take precedence.

For an initial native-categorical CPU search, limit categorical statistics to
individual features and run one tuning trial at a time:

```python
from ins_gbm import CatBoostModel, HyperparameterTuner, ModelRecipe

catboost_recipe = ModelRecipe(
    model=CatBoostModel(objective="poisson"),
    params={
        "random_seed": 42,
        "thread_count": 4,
        "bootstrap_type": "Bernoulli",  # Compatible with tuned subsample.
        "max_ctr_complexity": 1,
    },
    tuning=HyperparameterTuner(n_trials=20, cv_folds=3, seed=42, n_jobs=1),
)
catboost_fit = catboost_recipe.fit(training)
```

`max_ctr_complexity=1` disables combinations in CatBoost's categorical
statistics, which can reduce runtime but can also lose useful interactions.
After pruning, consider comparing it with `max_ctr_complexity=2` on the smaller
feature set. These fixed parameters are optional and are not imposed by the
wrapper. See CatBoost's [training-speed guidance](https://catboost.ai/docs/en/concepts/speed-up-training).

The wrapper does not supply a validation Pool for early stopping: each fit runs
its full requested iteration count. If the best trials reach 1,200 iterations,
consider extending that range after reducing the feature pool. Assess runtime
on your data before launching a large search. The default feature-sampling
range is not suitable for ordinary GPU Poisson training, where CatBoost does
not support `rsm`; supply a custom search space without `colsample_bylevel`
when using that configuration.

`ImportancePruner` uses the importance of the already fitted model and does not
perform a separate shallow screen. Retune the reduced feature set, and keep a
holdout separate from both discovery and tuning. For an outer CV estimate of
the complete selection procedure, selection must be relearned inside each
outer training fold; passing a selection learned on all rows freezes it.

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

The final binomial tests use Bonferroni correction across candidate features.
Set `multiple_testing="none"` to reproduce the previous unadjusted behavior.
This implementation is a shadow-importance screen with a final binomial test;
lowering the shadow percentile changes the heuristic and does not provide a
guaranteed false discovery rate. Importance rankings and Boruta screens do not
establish an optimal feature subset; compare complete recipes on outer CV or a
separate holdout.

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

Without an encoder, the model wrapper must support the supplied feature types;
LightGBM can consume schema categorical columns natively. For another model
wrapper, pass its supported `importance_types`; for example, LightGBM supports
`("split", "gain")`. `params` can override the shallow fitting defaults.

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

To screen preprocessing outputs, pass `preprocessing` with the same ordered
steps used by the final recipe. Encoding runs first, followed by preprocessing;
both are fitted independently on each fold's training rows and targets.
`feature_names`, when supplied to the screen, still names raw input columns.

```python
from ins_gbm import PCAReducer, PreprocessingStep

preprocessing = [PreprocessingStep(
    name="ages",
    preprocessor=PCAReducer(n_components=1),
    feature_names=["driver_age", "vehicle_age"],
)]
recipe = ModelRecipe(
    model=LightGBMModel(), encoder=OneHotEncoder(),
    preprocessing=preprocessing,
)
ranking = cv_feature_importance(
    training, model=recipe.model, encoder=recipe.encoder,
    preprocessing=recipe.preprocessing, importance_types=("split", "gain"),
)
selected = ranking.filter(pl.col("n_folds_selected") >= 3)["feature"].to_list()
fitted = recipe.fit(training, feature_names=selected, feature_stage="model")
```

The ranking includes components such as `ages__pca_1` and any columns passed
through by the steps. Select these outputs with `feature_stage="model"` so the
recipe retains their inputs until preprocessing is complete. Scores aggregate
by output name; independently fitted PCA/PLS components can represent different
directions across folds. This screen does not align components or attribute
their importance back to raw inputs.

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

Use `selection_scope="fold"` when tuning should evaluate the complete learned
selection workflow. With the default `"fixed"`, supervised selection has seen
the inner validation targets, so the tuning score can be optimistic. Outer CV
still refits the entire recipe on each outer training split. A feature list
chosen using all rows outside the recipe also needs a separate evaluation set.

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

Tuning minimizes the pooled validation metric: folds contribute in proportion
to their row count or effective metric weight, and RMSE pools squared errors
before taking the square root. History contains completed trials only.

Pass `search_space={...}` to replace the model's default Optuna distributions;
parameters omitted from that space can be fixed in `recipe.params`. Integer and
float distribution steps are preserved. The default spaces are starting points;
their ranges and trial budgets cannot guarantee the best settings for every
dataset. GPU modes and special bootstrap/boosting types may need a custom space.

Set `cache_transforms=True` on the tuner to fit encoding, selection, and
preprocessing once per training fold and reuse the transformed matrices across
trials. This freezes stochastic transforms and holds all fold matrices in
memory (with copies in each process worker), so caching is disabled by default.
When running concurrent trials, set each model's thread budget explicitly:
`num_threads` for LightGBM, `nthread` for XGBoost, `thread_count` for CatBoost,
or `n_jobs` for Random Forest. This avoids excessive nested concurrency.

LightGBM automatically enables `bagging_freq=1` when a supplied row-sampling
fraction is below one, unless a bagging frequency is explicitly supplied.

| Model | Poisson fitting | Gamma fitting | Categories without an encoder |
| --- | --- | --- | --- |
| LightGBM | Native Poisson with exposure/offset | Native Gamma with offset | Native |
| XGBoost | `count:poisson` with exposure/offset | `reg:gamma` with offset | Numeric inputs required |
| CatBoost | Native Poisson with baseline | Tweedie power 1.99, a Gamma approximation | Native |
| Random Forest | Exposure-weighted rate regression with Poisson deviance splits | Severity regression with MSE | Numeric inputs required |

CatBoost does not expose an exact native Gamma objective. Random Forest's
objectives are approximation benchmarks, and it rejects explicit offsets.
Native objective overrides that conflict with a wrapper's resolved objective
are rejected to keep the log-link prediction contract valid.

Random Forest defaults follow the resolved model/data objective: `criterion="poisson"`
with `min_samples_leaf=10` for frequency, and `criterion="squared_error"` with
`min_samples_leaf=5` for Gamma severity (sklearn has no Gamma criterion). Both use
`n_estimators=200`, `max_depth=12`, `max_features=0.8`, `n_jobs=1`, and
`random_state=42`. These are starting points for tuning; explicit `params`
override any of these defaults, including `criterion`. The Poisson criterion
requires at least one positive training target; all-zero targets raise `ValueError`.

## Prediction contract

- With a Poisson rate target and no exposure, `response` and `rate` both return
  predicted rates. Fractional nonnegative targets are supported.
- For optional count modeling, supply count targets and positive exposure;
  `response` then returns expected counts and `rate` divides by exposure.
  Do not supply an exposure offset with an already divided rate target.
- Gamma `response` is expected severity and does not support `rate`.
- For both objectives, `link` is `log(response)`.
- Explicit offsets add on the link scale. Poisson combines offset and
  `log(exposure)`.
- Random Forest is an approximation benchmark and rejects explicit offsets.

Pass `feature_names=[...]` to `recipe.fit()` or `recipe.cross_validate()` to reuse
one loaded candidate pool across fits. To select encoded columns such as
`"territory__urban"`, pass `feature_stage="encoded"` with `feature_names`.
Each cross-validation fold fits its own encoder before selecting those columns.

Model wrappers calculate feature importance on request. CatBoost's
default importance is `LossFunctionChange`, deferred until importance is
requested or the fitted model is saved. Its compact scores are cached so saved
models need no training rows. Until then, the fitted CatBoost wrapper retains
its training Pool in memory.
CV scoring and tuning do not request importance; Boruta, staged importance
selection, and `cv_feature_importance()` calculate it as part of selection.
CatBoost itself still calculates `PredictionValuesChange` inside its native
`fit()`.

## Ensemble methods and prediction scales

`EnsemblePipeline` supports `"blending"`, `"stacking"`, `"seed_average"`, and
`"bagging"`. Seed averaging and bagging start from one fitted pipeline, reuse
its model parameters, and independently refit transforms for each member:

```python
bagged = EnsemblePipeline(
    [fitted], method="bagging", n_members=5, seed=42,
    sample_fraction=0.8, bootstrap=True,
).run()
seed_average = EnsemblePipeline([fitted], method="seed_average", n_members=5).run()
bagged.save("output/bagged")  # load_ensemble("output/bagged") restores it
```

**Use Poisson rate targets with exposure left empty by default.** Every ensemble
`predict()` returns rates directly in this workflow. Blending, bagging, and seed
averaging average rates on the response scale. Gamma inputs and predictions
remain severity. Count targets with explicitly supplied exposure remain supported;
those predictions are counts and can be divided by scoring exposure for rates.

**Stacking now defaults to Poisson/Gamma regression**, using log normalized base
predictions. With rate targets and no exposure or offset, Poisson stacking fits
the supplied rates and observation weights, returning rates directly. When count
targets and exposure are explicitly supplied, it internally fits rates weighted
by exposure times observation weights and restores scoring exposure for counts.
Explicit offsets are also normalized out and restored once;
Gamma stacking normalizes only for offsets and uses observation weights.
Select `meta_learner="ridge"` to retain the previous response-scale approach,
or supply a custom estimator. `meta_alpha=1.0` controls built-in regularization.

See [the ensemble guide](docs/ensembles.md) for the exact count/rate and offset
formulas, severity examples, fold controls, sampling behavior, and save/load.

## Compare models and ensembles using training-data CV

Keep the final holdout reserved while comparing candidates:

```python
from ins_gbm import CVConfig, cross_validate_candidates, compare_reports

results = cross_validate_candidates(
    {"single": single_recipe, "blend": blend_candidate, "stack": stack_candidate},
    train, cv=CVConfig(n_splits=5, seed=42), base_refit="fixed",
    inner_cv=CVConfig(n_splits=3, seed=42, folds="random"),
)
print(compare_reports(results))
print(results["stack"].cv_config["execution"])
```

Candidates are `ModelRecipe` or `EnsemblePipeline` configurations. Matching
base fits and predictions are shared within outer folds; combination weights
and meta-models are fitted independently using only outer training rows.
Choose grouped inner folds when your data requires group isolation. All
ensemble methods support individual `candidate.cross_validate(train, cv=...)`
and `CrossValidationReport(recipe=candidate, ...)`, returning standard,
persistable `CVResult` objects. Poisson OOF outputs remain expected counts.
See [the ensemble CV guide](docs/ensembles.md#comparing-candidates-before-opening-the-final-holdout)
for tuning policies, manual feature subsets, internal validation splitting,
fit-count interpretation, and final holdout evaluation.

## Ensemble folds and weight optimization

OOF blending and stacking reuse the base pipelines' training `ModelData.cv_fold`
assignments when available, otherwise they generate random folds. Set
`use_data_folds=True` to require existing folds, or `False` to generate random
folds. Alternatively pass `cv=CVConfig(folds="predefined")`; `CVConfig` also
controls the random fold count and seed. All base pipelines must have aligned
training rows and fold assignments.

```python
from ins_gbm import EnsemblePipeline

ensemble = EnsemblePipeline(
    fitted_pipelines=[first, second],
    blend_mode="oof",
    use_data_folds=True,
    metric="auto",
).run()
```

For blending with `blend_mode="oof"` or `"validation"`, `metric="auto"` selects
Poisson or Gamma deviance from the fitted models' common objective. Explicit
`"poisson_deviance"` and `"gamma_deviance"` must match that objective. The default
remains `"mse"`. Optimization respects observation weights; Poisson deviance
uses exposure-weighted rates, matching evaluation. These options are also
available directly on `BlendingEnsemble`, and fold controls are available on
`StackingEnsemble`. Evaluate the result on a separate final holdout.

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for architecture and migration
details. The runnable workflow is `examples/example_usage.py`, with a companion
notebook at `examples/example_usage.ipynb`.
