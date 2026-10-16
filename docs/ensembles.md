# Ensembles: fitting, rates, counts, and persistence

`EnsemblePipeline` supports fixed/optimized blending, distribution-aware
stacking, seed averaging, and bagging. All methods return `EnsembleResult`
with `predict(data)`, `evaluate(holdout_data)`, and `save(directory)`.

## Rate targets by default

For **Poisson frequency**, supply nonnegative observed **rates** as
`ModelData.target` and leave `ModelData.exposure=None` (the default). Every base
pipeline and ensemble fits those rates directly. Fractional targets and zero
rates are supported. No exposure is inferred or applied. Optional
`ModelData.weight` supplies observation weights independently.

```python
predicted_rates = result.predict(scoring_data)  # exposure remains None
```

Blending, stacking, seed averaging, bagging, tuning, and outer CV all follow this
same convention. `response` and `rate` predictions coincide without exposure.
Keep training and scoring targets, weights, and exposure definitions consistent.

Count modeling remains available explicitly: supply count targets and positive
exposure separately. Response predictions then represent expected counts for
that scoring row; divide by scoring exposure to obtain rates. For example, count
`0.3` at exposure `0.5` means rate `0.6`. Do not supply an exposure offset for a
target already divided by exposure. Ensemble `predict()` returns the response
scale and has no `prediction_type` argument.

For **Gamma severity**, targets and response predictions are positive severity
amounts. There is no count-to-rate conversion. Exposure is neither a severity
multiplier nor a meta-training weight. `ModelData.weight` can supply observation
weights such as claim counts, depending on the meaning of the severity target.

An explicit `ModelData.offset` is a separate **log-scale** model baseline, not
an observation weight. Backends must support offsets; Random Forest does not.
Poisson base models incorporate both `log(exposure)` and the explicit offset,
while Gamma incorporates only the explicit offset.

## Seed averaging and bagging

Start from **one** fitted template pipeline with its original training rows
attached:

```python
from ins_gbm import EnsemblePipeline

seed_average = EnsemblePipeline(
    [frequency_fit], method="seed_average", n_members=5, seed=42,
).run()

bagged = EnsemblePipeline(
    [frequency_fit], method="bagging", n_members=5, seed=42,
    sample_fraction=1.0, bootstrap=True,
).run()

# The same interface works for severity.
severity_bagged = EnsemblePipeline(
    [severity_fit], method="bagging", n_members=5,
    sample_fraction=0.8, bootstrap=False,
).run()
```

Seed averaging refits each member on all rows with a distinct model seed.
Bagging samples rows uniformly, with replacement by default. The sample size is
`floor(sample_fraction * n_rows)` and must be at least one; the fraction must be
in `(0, 1]`. `bootstrap=False` draws without replacement. Sampled exposure,
weights, offsets, comparisons, and fold IDs remain aligned with their rows;
bootstrap duplicates retain all their original row fields. Exposure does not
change a row's sampling probability.

Both methods reuse the template's fitted hyperparameters without rerunning its
tuner. Each member independently refits encoding, configured feature selection,
and preprocessing. Manually fixed encoded/model feature subsets remain fixed.
If a sample lacks a required encoded feature, fitting fails with the member
index and seed rather than silently changing the selected feature set.

Members are fitted sequentially. The master seed determines model seeds and
sampling reproducibly; `result.configuration` records the member seeds and
sampling settings. Backend model seed settings are overridden for each member.
Other model settings remain unchanged. Different seeds need not produce
meaningfully different fits when the model has no active stochastic behavior.

Predictions are an equal-weight **arithmetic average on the response scale**.
For the default Poisson rate targets with no exposure, this averages rates
directly. For explicit count targets with exposure, it averages counts; dividing
by the common scoring exposure gives the average rate. Gamma averages severity.

Generated members are available through `result.base_pipelines`. Bagging
members have different training samples; do not pass them to stacking/OOF
blending, which require aligned training rows. They can be combined through
fixed or separate-validation blending. This version does not provide OOB
scores, group sampling, or per-member retuning.

## Distribution-aware stacking

```python
stacked = EnsemblePipeline(
    [first, second], method="stacking", meta_alpha=1.0,
    use_data_folds=True,
).run()

# Explicit compatibility option for the previous stacking approach:
ridge_stacked = EnsemblePipeline(
    [first, second], method="stacking", meta_learner="ridge",
).run()
```

**Default change:** `meta_learner=None` or `"auto"` selects `PoissonRegressor`
or `GammaRegressor` according to the base pipelines' common objective. These
have a log link and train on log normalized base predictions. `meta_alpha`
controls their regularization, defaulting to `1.0`; both fit an intercept and
allow up to 1,000 optimization iterations.

The meta-learner is trained on OOF predictions only. All pipelines must have
aligned original training rows, targets, exposure, weights, offsets, and fold
IDs. Predefined `ModelData.cv_fold` assignments are reused when available;
`use_data_folds=True` requires them, while `False` requests random folds.
Alternatively pass `cv=CVConfig(...)`. `refit="fixed"` reuses fitted model
parameters inside folds; `"retune"` repeats configured tuning in each training
fold. The supplied full-data fitted pipelines are used for final predictions.

### Poisson: direct rate fitting by default

With rate targets and no exposure or offset, meta-features are `log(mu_j)`,
the meta-target is the supplied rate `y`, and only optional observation weights
are passed. Predictions are rates directly. No exposure-derived weights or
count conversion are applied. Explicit offsets are normalized and restored as
described below.

### Optional counts with exposure

Let `y` be observed count, `mu_j` each base model's expected count, `e` exposure,
`o` explicit log offset, and `w` observation weight. Missing exposure/weight is
one; missing offset is zero. Define:

```text
b = e * exp(o)
meta features:      log(mu_j / b)
meta target:        y / b
meta sample weight: b * w
predicted count:    b_scoring * meta_model.predict(meta_features_scoring)
ordinary rate:      predicted count / e_scoring
```

Without an explicit offset, `b=e`: targets become observed rates and training
weights are exposure times observation weight. Zero claim counts remain valid.
Scoring multiplies the fitted rate by the **scoring row's** exposure, exactly
once. It does not use training exposure averages.

With an explicit offset, the internal mean is also adjusted for `exp(o)`.
It is not the ordinary claim rate: the ordinary rate retains the offset factor.
For example, if `e=0.5`, `exp(o)=2`, and the meta-model's normalized mean is
`0.3`, the returned expected count is `0.3` and the ordinary rate is `0.6`.
Changing exposure to `1.0` returns count `0.6` and rate `0.6`. The offset and
exposure already present in base response predictions are removed before
meta-fitting and restored once afterward, avoiding double application.

### Gamma: severity fitting

For Gamma, `b=exp(o)`. The features are `log(mu_j / b)`, target is `y / b`,
sample weight is `w`, and returned severity is `b_scoring` times the normalized
meta-prediction. With no offset, the target stays severity. Exposure has no
role in this calculation. Positive severity and positive finite base means
are required.

### Ridge and custom learners

`meta_learner="ridge"` retains response-scale OOF inputs and response targets,
with positive clipping of predictions. Custom estimators follow the same
response-scale contract, must implement `fit`/`predict`, and must accept
`sample_weight` when observation weights are supplied. Custom estimators are
copied before fitting. Observation weights are passed without multiplying them
by exposure. These response-scale learners do not enforce the exposure/offset
scaling guarantees of the built-in distribution learners.

## Blending loss and final evaluation

`metric="auto"` selects Poisson/Gamma deviance when optimizing validation or
OOF blend weights; explicit `"poisson_deviance"` / `"gamma_deviance"` must match
the model objective. The default remains `"mse"`.

Without exposure, Poisson deviance and MSE compare supplied rates and predicted
rates directly, using only optional observation weights.

With explicit count targets and exposure, Poisson deviance is evaluated on
`y/e` and `mu/e`, weighted by `e*w`, consistent
with the evaluation report. The explicit offset is already incorporated in
`mu` and is not a second metric weight. MSE measures **response-scale** errors for
Poisson, using observation weights without an additional exposure multiplier.
Gamma deviance uses severity and observation weights.

Always assess the completed ensemble on a separate holdout. OOF predictions
used to fit weights or a stacker are training evidence for that combination,
not an independent evaluation of the fitted ensemble.

## Comparing candidates before opening the final holdout

Use training-data outer CV to select between single models and ensembles. The
runner accepts ordinary `ModelRecipe` objects and `EnsemblePipeline`
configurations; no ensemble `.run()` call is needed before CV.

```python
from dataclasses import replace
from ins_gbm import (
    CVConfig, EnsemblePipeline, cross_validate_candidates,
    compare_reports, compare_cv_double_lift, load_cv_result,
)

# first and second were fitted/tuned on train. Reuse selected parameters.
single_recipe = replace(
    first.recipe, tuning=None, params=first.fitted_model.params,
)
inner = CVConfig(n_splits=3, seed=42, folds="random")
candidates = {
    "single": single_recipe,
    "blend": EnsemblePipeline(
        [first, second], blend_mode="oof", metric="auto", cv=inner,
    ),
    "stack": EnsemblePipeline([first, second], method="stacking", cv=inner),
}
results = cross_validate_candidates(
    candidates, train,
    cv=CVConfig(n_splits=5, seed=42),  # shared outer evaluation folds
    base_refit="fixed",
)
print(compare_reports(results))
print(compare_cv_double_lift(results["single"], results["stack"]))
print(results["stack"].cv_config["execution"])

results["stack"].save("output/stack_cv")
restored_cv = load_cv_result("output/stack_cv")
# Supply train when both compared CV results were loaded from disk.
# compare_cv_double_lift(restored_cv, another_loaded_cv, data=train)

# After choosing an ensemble from the CV evidence:
selected = candidates["stack"].run()  # replace "stack" with your chosen candidate
final_report = selected.evaluate(holdout)  # first use of the reserved holdout
```

Regular recipes keep their own configured tuning and selection behavior. The
`single_recipe` example reuses selected parameters and its configured feature
selection. If the original fit instead used a **manually fixed raw, encoded, or
model feature subset**, use
`EnsemblePipeline([first], blend_weights=[1.0])` as the single-model candidate.
That rebuilds exactly the template's feature specification and shares matching
fits with other ensembles. This also avoids introducing extra raw features
from a broader common candidate pool. Global `feature_names` / `feature_stage`
overrides are unsupported for ensemble reports; each template defines its own
inputs and feature stages.

Individual ensemble CV and the existing report interface are also supported:

```python
from ins_gbm import CrossValidationReport

stack_cv = candidates["stack"].cross_validate(train, cv=outer_cv)
stack_cv = CrossValidationReport(
    recipe=candidates["stack"], data=train, cv=outer_cv,
    base_refit="fixed", show_progress_bar=False,
).run()
```

Here `outer_cv` is a `CVConfig` chosen for your evaluation. The returned
`CVResult` uses the existing `"gbm"` prediction column for compatibility with
comparisons, double lift, plots, and persistence. The column name also represents
ensemble predictions. Poisson OOF predictions are **rates** by default, with
exposure omitted; metrics compare rates directly. Explicit count targets with
exposure produce count OOF predictions, exposure-weighted rate deviance, and count MSE.
Gamma OOF predictions are severity. Exposure and offsets follow the same
normalization/restoration formulas documented above.

### What happens inside an outer fold

Every model, encoder, configured selector, and preprocessor is rebuilt using
outer training rows only. Outer held-out targets are used only for scoring.
Source fitted models/transforms never predict outer held-out rows. Explicit
`train` data supplies the rows, so compact loaded templates work even though
their original training data is not attached.

Fixed blends apply their fixed weights. OOF blends and stackers generate
**inner** OOF predictions within the outer training rows to learn the
combination. Bagging/seed averaging train members directly on those rows,
avoiding an unused preliminary model when reusing parameters. Validation blends
reserve an internal blend subset (`validation_fraction=0.2`) and train bases
only on the remaining subset; bases are not refitted after learning weights.
Ensemble CV rejects external `validation_data`, so the final holdout cannot
accidentally become the blend set.

Outer `cv` and inner folds are separate. Inner defaults come from each ensemble
configuration; passing `inner_cv=CVConfig(...)` to the runner or individual CV
call overrides them. Predefined inner folds reuse only the groups remaining
in an outer training fold. Validation blending holds out complete predefined
groups when the inner policy is `"auto"` with fold IDs or `"predefined"`.
Its split seed is the inner seed plus the outer fold's ordinal position.
Random inner policies split rows. If too few groups/rows remain, CV fails before
training; it never silently substitutes random folds. For example, two
predefined outer groups leave only one group for inner OOF learning. Choose
appropriate nested groups or explicitly select a random inner policy if rows
within groups may be split. The small-budget example above intentionally uses
random inner folds; retain grouped inner folds for genuinely grouped data.

`base_refit="fixed"` reuses ensemble templates' selected model parameters.
`base_refit="retune"` runs their configured tuning inside outer training data;
templates with no tuner simply refit. The ensemble's own `refit` setting
independently controls inner fits. Retuning bagging/seed-average templates
selects parameters once per outer fold, then trains all members without tuning.
The standard tuning pipeline also performs its final template fit in this
optional retuning path. No hyperparameter search is shared between candidates.

### Runtime and fit sharing

The multi-candidate runner shares matching fixed-parameter base fits and their
response predictions within each outer fold. For example, a single model,
a blend, and a stacker can reuse matching outer fits; blends and stackers with
the same bases and inner folds reuse inner fits/predictions. Their weights and
meta-models are always learned separately. Fits differ when model parameters,
seeds, transforms, selection, feature stages, or ordered training rows differ.
Bootstrap duplicates and row ordering are part of the identity. Custom
configurations that cannot be reliably fingerprinted bypass sharing.

The runner processes regular recipes first, allowing their resolved backend
parameters to match fitted templates. Exact matching is conservative: equivalent
parameter aliases are not universally normalized, so some equivalent
configurations can still train separately. No sharing occurs across invocations
or outer folds. Models and caches are released after each fold; raw training
rows and unused diagnostic state are removed from cached fitted pipelines.
Samples are generated one at a time to limit index-array memory.

`cv_config["execution"]` contains totals for the **whole runner invocation**,
repeated on each candidate's result:

- `pipeline_fits`: successful final pipeline fits, including cache misses.
- `tuned_pipeline_fits`: those pipeline fits that ran a configured tuner.
- `fit_cache_hits`: fitting requests served by an existing matching pipeline.
- `prediction_cache_hits`: prediction requests served from the response cache.

These are not counts of individual fits inside tuning or feature selection.
Those additional fits can dominate runtime. Keeping parameters fixed, using a
small inner-fold budget, and comparing related candidates together reduces
cost. As a regression-test example, one single model plus a two-base OOF blend
and stacker requires 18 base fits together versus 39 in separate calls with
three predefined outer groups and two remaining inner groups. This is a fit
count, not a wall-time guarantee.

### Interpreting the comparison

CV evaluates the specified candidates before final holdout evaluation. Reused
parameters and manually fixed feature subsets define a **conditional**
evaluation: when selected using all training data, that earlier search is not
independently evaluated by these folds. Configured selectors are refitted per
fold; optional retuning evaluates configured tuning within the outer folds.
Selecting among many candidates can still make the best CV score optimistic.
Keep the reserved holdout for the final selected model or ensemble.

CV artifacts retain metrics, outer OOF predictions, fold provenance, fit counts,
and serializable per-fold member/combination parameters. They do not retain
fold models, raw training features, or the fitting caches. Comparing different
feature sets is supported when evaluation rows, objective, weights, exposure,
and outer folds align.

## Saving and loading

```python
from ins_gbm import load_ensemble, save_ensemble

bagged.save("output/frequency_bagged")
# Equivalent: save_ensemble(bagged, "output/frequency_bagged")
loaded = load_ensemble("output/frequency_bagged")
predicted_rates = loaded.predict(scoring_data)
report = loaded.evaluate(holdout)
```

All ensemble methods save `ensemble.pkl` and `metadata.json`, including the
objective, configuration, member metadata, model seeds, and resolved stacking
learner. Artifacts omit members' raw training rows and stacking OOF arrays;
a loaded stacker's `oof_predictions` is `None`. The original in-memory result
keeps its training rows and OOF predictions. Loaded results support prediction
and holdout evaluation, but training-dependent workflows require the original
training data. Use trusted pickle artifacts in compatible package environments.
