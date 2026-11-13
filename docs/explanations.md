# Post-fit explanations

## Start here

Run a complete example from the repository root; no input files are needed:

```bash
uv sync --all-extras --dev
.venv/bin/python examples/explain_models.py --shap --native
```

This writes single-model, final-ensemble, PCA, and reloaded-model examples under
`output/explanation_example/`. Open the PNG/SVG figures and inspect the adjacent
CSV tables and `settings.json`. Omit `--native` to use only Random Forest members;
omit `--shap` to run without SHAP. For a smaller install, use
`pip install -e '.[explain]'` from the repository root and omit `--native`.
The [notebook](../examples/example_usage.ipynb) integrates these methods into the
full modeling workflow.

| Question | Method | What to show |
| --- | --- | --- |
| Which features typically move predictions most? | SHAP `summary()` | Importance bars or beeswarm |
| Why did this prediction differ from its reference? | SHAP waterfall | One row's reference, contributions, and prediction |
| What average response shape did the model learn? | One-variable PDP | Curve or categorical points, plus sample support |
| How do two variables jointly affect predictions? | Two-variable PDP | Response heatmap and joint support |
| Does the response shape vary across profiles? | ICE / centered ICE | Individual profiles, optionally aligned at a reference value |
| What local effects appear in observed numeric regions? | ALE | Centered effect curve or second-order interaction heatmap |

Use training or development populations for exploratory model revisions. Open
the final holdout after choices are locked; changing the model after inspecting
holdout explanations uses that holdout for selection. SHAP explanations depend
on the chosen reference population. PDP/ICE can extrapolate when features are
dependent, and ALE still needs enough observations in its bins.

## First calls on your own model

`FittedPipeline.explain()` and `EnsembleResult.explain()` return the same lazy
`ExplanationReport`. They use the fitted model without refitting, selecting,
tuning, or scoring outcomes. Supply original `ModelData`, including scoring
exposure and offset where applicable. Targets are retained for the existing
data container contract but do not enter explanation calculations.

```python
report = fitted.explain(
    scoring_data,
    background=training_reference,  # required only for SHAP
    prediction_type="response",
    seed=42,
    batch_size=1024,
)
# The same calls apply when fitted is an EnsembleResult.
print(report.feature_names)       # valid names for every method below
print(report.feature_metadata)    # raw/category/component descriptions
# Replace example names below with names from report.feature_names.
importance = report.shap(max_samples=20, max_background=20, n_permutations=2)
print(importance.summary())
importance.plot_waterfall(row=0)
importance.plot_beeswarm()
importance.plot_dependence("driver_age", color_by="vehicle_age")
importance.export("output/explanations/shap")

age = report.pdp("driver_age")
joint = report.pdp(["driver_age", "territory"],
                   grids={"territory": ["north", "south"]})
individual = report.ice("driver_age", centered=True)
local = report.ale("driver_age", bins=20)
interaction = report.ale(["driver_age", "vehicle_age"], bins=5)
age.export("output/explanations/age")
```

`scoring_data`, `training_reference`, and `fitted` above are your original
`ModelData` population, reference `ModelData`, and fitted pipeline/ensemble.
Create the pipeline with `recipe.fit(training)` or restore it with `load_model`.
Choose a representative training/reference population with the same feature
definitions as scoring data. Do not pass `fitted.train_data`, one-hot matrices,
or manually computed PCA scores to `.explain()`.

The deliberately small SHAP budget is a first-run check. Increase explanation
rows, reference rows, and permutations after checking runtime and repeat stability.
All four methods return result objects; by default calculations happen when the
method is called. To calculate SHAP during `.explain()` and reuse it afterward:

```python
report = fitted.explain(
    scoring_data,
    background=training_reference,
    shap_kwargs={"max_samples": 100, "max_background": 20, "n_permutations": 2},
)
importance = report.shap()  # Already calculated, for all report features.
importance.plot_dependence("driver_age")
importance.plot_dependence("vehicle_age")
```

`shap_kwargs=None` preserves lazy construction; an empty mapping eagerly computes
SHAP with its normal defaults, including all explanation rows. Supplied options
become defaults for later `report.shap(...)` calls. Explicit arguments override
them. The report caches only the most recent SHAP result and reuses it when the
effective options match, including averaging weights. Changed options recalculate
SHAP. Treat report data, background, configuration, and cached results as immutable;
create a new report when they change. Results are retained in memory only.
PDP, ICE, and ALE do not use SHAP and are still calculated separately on every call.

Print `.table` for effects,
`.contributions`/`.predictions` for SHAP, and `.settings` to audit the calculation.

## Coordinates and encodings

Raw categorical variables are indivisible explanation features. Every member
receives the appropriate native category or fitted one-hot indicators, including
when only some indicator levels survived selection. Changes and SHAP masking
never create independent, invalid one-hot combinations. Previously unseen values
follow the fitted encoder/backend's existing behavior.

`feature_space="auto"` (the default) explains original raw features for an
ensemble and processed components for a single fitted pipeline. Set
`feature_space="raw"` or `feature_space="processed"` explicitly to use either
interpretation. The choice applies to SHAP, PDP, ICE, and ALE and is recorded
as `settings["feature_space"]` on results.

In raw space, every perturbation passes through each member's existing fitted
encoding, selection, scaling, and reducers before the ensemble combines its
predictions. Nothing is refitted. A feature used directly in one member and
through PCA in another has one attribution covering both paths. Members may
have different fitted PCA bases, including identically named components.
Only raw sources of final model features are attributed. Categorical variables
remain whole, including when only selected one-hot levels enter a model.

Raw SHAP uses the general permutation explainer; `engine="tree"` is ineligible.
It can cost more because each evaluation replays fitted preprocessing. Raw
attributions are calculated from the complete prediction function, not allocated
from component SHAP values using loadings. Predictions and output scales stay
the same; baselines and attributions can differ from processed-space explanations.
Background masking can still produce unusual combinations of correlated raw
features. Contributions describe model behavior, not causal effects, and do not
separate direct effects from effects mediated by PCA.

In processed space, fitted reducer outputs are separate continuous features.
For a targeted PCA step called `ages`, explain `ages__pca_1`, `ages__pca_2`, etc.
Importance belongs to these scores and is never spread over their original
inputs. `report.feature_metadata` and result settings include component lineage
and, for PCA, loadings, scaler state, and explained variance. Interventions vary
component scores without implying a unique change in the source variables.

Processed-space ensembles still require compatible fitted component coordinates:
input ordering, upstream transforms, scaler, loadings, and outputs must agree.
Members can select different components of one shared basis. Raw/component
source overlap and independently fitted incompatible bases raise an error.
PLS/UMAP follow the same rule. Custom reducers need lineage metadata via
`component_mapping()`; custom encoders currently raise an explicit error.

```python
# Handles different PCA bases and raw/PCA overlap across ensemble members.
report = ensemble.explain(scoring_data, background=training_reference)
print(report.feature_names)  # original variable names
values = report.shap(max_samples=20, max_background=20, n_permutations=2)

# Optional: explain a single pipeline in original variables too.
raw_report = pca_fit.explain(scoring_data, background=training_reference,
                             feature_space="raw")
```

### Example: explain PCA scores

```python
from ins_gbm import ModelRecipe, PCAReducer, PreprocessingStep, RandomForestModel, OneHotEncoder

pca_fit = ModelRecipe(
    model=RandomForestModel(),
    encoder=OneHotEncoder(),
    preprocessing=[PreprocessingStep(
        name="ages", preprocessor=PCAReducer(n_components=2),
        feature_names=["driver_age", "vehicle_age"],
    )],
).fit(training_reference)
pca_report = pca_fit.explain(scoring_data, background=training_reference)
print(pca_report.feature_names)  # includes ages__pca_1 and ages__pca_2
print(pca_report.feature_metadata["ages__pca_1"])
pca_report.pdp("ages__pca_1").plot()
pca_report.ale("ages__pca_1", bins=5).plot()
pca_values = pca_report.shap(max_samples=20, max_background=20, n_permutations=2)
print(pca_values.summary())
```

The two original age columns have been replaced by component coordinates in
this example. A request for their raw names raises an unknown-feature error.
For explicit `feature_space="processed"` ensemble explanations, matching PCA
step configurations is insufficient: the fitted states must agree. See [ensemble explanations](ensembles.md#explaining-final-ensemble-predictions)
for compatibility and scale rules.

## Output scales

`prediction_type="response"` explains the public prediction: Poisson rates when
exposure is omitted (the default), expected counts with explicit exposure, or
Gamma severity. `"rate"` divides Poisson response by scoring exposure and is
invalid for Gamma. `"link"` explains log final response, including exposure and
offset; it is not log rate. Log explanations require positive predictions.

Ensemble diagnostics evaluate the completed combination before converting
scales. This preserves arithmetic blends, distribution-aware log-link stacking,
custom stackers, clipping, and exposure/offset normalization/restoration. There
is no averaging of member log-SHAP values to explain response blends. PDP on the
link scale averages log predictions; it does not log the average response.

Exposure and offsets stay fixed to each explained row during perturbations and
masking. They are context, not attributed features. SHAP reference values can
therefore vary between rows. Reconstruction checks verify, on the selected scale:

```text
base_value + sum(feature contributions) = prediction
```

On the response scale, a contribution of `+0.02` means an additive increase of
0.02 in the displayed response units relative to the reference. On the link
scale it is a log-response difference. Its exponential is a multiplicative
factor, not an additive rate or currency amount. Centered ICE shows differences
from the chosen grid reference; ALE is centered to zero rather than showing
absolute predicted frequency or severity.

## SHAP configuration

Install the optional dependency with `pip install 'ins_gbm[explain]'`. Other
diagnostics do not import SHAP. Defaults are:

- Up to 100 uniformly sampled reference rows; the background must be explicit.
- All supplied explanation rows, or a seeded `max_samples` cap.
- Twenty complete forward/reverse permutation cycles per row; use
  `n_permutations` to set the budget and `n_repeats` for independent seeded runs.
- `engine="auto"`: use native interventional Tree SHAP only for eligible
  individual-model coordinates and output scales, otherwise permutation SHAP.

Native acceleration supports numeric model columns with the same players and
background semantics: supported GBMs on the log-response scale, and Random
Forest on response/rate scales when clipping cannot intervene. Grouped one-hot
or native categorical inputs use the general engine. Final ensembles always use
the general engine. Explicit `engine="tree"` fails for unsupported cases;
`"permutation"` always requests the general engine. The selected engine and
fallback reason are saved. Tree-path-dependent explanations are never silently
substituted for the explicit empirical reference distribution.

`ShapResult` retains contributions and per-row prediction/reference/residual
tables. `summary()` returns mean absolute contributions, signed means, and
between-repeat variability. `summary(groups={"vehicle": ["make", "vehicle_age"]})`
sums signed contributions within each row before taking absolute values; this
is a display grouping, not a recomputed group Shapley game.
`segment_summary(labels)` accepts labels aligned to `result.samples` (after
any sampling). Variability across runs is an empirical diagnostic, not a
confidence interval. Dependence coloring suggests patterns; it is not an exact
SHAP interaction decomposition.

For repeat checks and segment summaries:

```python
values = report.shap(max_samples=20, max_background=30, n_permutations=5, n_repeats=3)
print(values.summary())  # repeat_std measures variation in mean absolute importance
source_rows = values.settings["row_indices"]
# Labels must match values.samples, not the unsampled full population.
labels = scoring_data.features["territory"][source_rows]
print(values.segment_summary(labels))
print(values.settings["engine"], values.settings["fallback_reason"])
print(values.predictions.select("row", "source_row", "base_value", "prediction", "residual"))
```

`plot_waterfall(row=0)` uses the local row number in the result, while
`predictions["source_row"]` and `settings["row_indices"]` identify original
population positions. Waterfalls default to the 20 largest contributions and
combine the remaining signed contributions into one row; `top_n=None` shows all.

The general engine costs roughly explanation rows × permutation cycles ×
`(2 * number_of_features + 1)` × reference rows in underlying prediction work,
before masking/caching shortcuts. Ensembles also evaluate their members. Start
small on wide datasets. `batch_size` limits prediction-call size; it does not
cap the number of rows calculated or retained in result tables.

## PDP, ICE, and ALE settings

PDP accepts one or two variables, including categories and processed components.
ICE accepts one variable, with optional `centered=True` and `reference=...`.
Numeric grids default to 20 distinct quantiles between the 5th and 95th
percentiles. Customize `grid_size`, `percentiles`, or `grids={name: values}`
(`grid=values` for ICE). Categorical grids use observed levels including missing;
above 30 levels an explicit grid is required. ICE displays at most 100 seeded
profiles by default, controlled by `display_rows`, while keeping all calculated
profiles in its table. `max_samples` explicitly limits the calculation population.

PDP/ICE support tables report observations assigned to the nearest numeric grid
point or exact category, including joint support for two-variable PDP. Unsupported
cells are flagged; this empirical support does not certify every perturbed
combination is realistic. Correlated variables can make PDP/ICE extrapolate.

ALE supports numeric features: 20 quantile bins for one-variable effects or ten
per axis for two-variable interaction effects. Use `bins` or `edges={name: edges}`.
Second-order ALE removes lower-order effects. Effects are centered under the
weighted bin-center empirical distribution. Empty or zero-weight cells raise
with guidance to reduce bins or supply appropriate edges. Missing values and
numeric sentinels are excluded from ALE and reported; constant variables have
zero effects. Nominal categorical ALE is unsupported; use categorical PDP/ICE.

Population averaging uses `ModelData.weight` by default, otherwise equal weights.
Override with an aligned `weights` array or `weights="uniform"`. Exposure is
never implicitly multiplied into averaging weights. SHAP background probabilities
remain uniform and are separate from summary weights. No capped sampling is
applied to PDP/ALE unless explicitly requested.

Use the same grid and population to compare PDP with ICE:

```python
grid = [25, 35, 45, 55, 65]
curve = report.pdp("driver_age", grids={"driver_age": grid}, weights="uniform")
profiles = report.ice("driver_age", grid=grid, weights="uniform")
centered = report.ice("driver_age", grid=grid, centered=True, reference=45)
print(curve.table)
print(curve.support)
```

At each grid value, the PDP equals the weighted mean of the uncentered ICE
predictions for the same rows. `display_rows` changes only ICE plotting;
`max_samples` changes which rows are calculated. Numeric missing values and the
library sentinel are omitted from automatically generated grids; ordinary
PDP/ICE predictions still use all selected rows as the reference population.

## Parameter reference

| Call | Main arguments and defaults |
| --- | --- |
| `fitted.explain(data, ...)` | `background=None`, `prediction_type="response"`, `seed=42`, `batch_size=1024`, `progress_callback=None`, `should_stop=None`, `shap_kwargs=None` |
| `report.shap(...)` | `engine="auto"`, `max_background=100`, `max_samples=None`, `n_permutations=20`, `n_repeats=1`, `weights=None` |
| `report.pdp(features, ...)` | One/two feature names; `grids=None`, `grid_size=20`, `percentiles=(.05, .95)`, `weights=None`, `max_samples=None` |
| `report.ice(feature, ...)` | One name; `grid=None`, `grid_size=20`, `percentiles=(.05, .95)`, `centered=False`, `reference=None`, `display_rows=100`, `weights=None`, `max_samples=None` |
| `report.ale(features, ...)` | One/two numeric names; `bins=None` (20 or 10 per axis), `edges=None`, `weights=None`, `max_samples=None`, `xlim=None` |

Use mappings for `grids` and `edges`, such as `{"driver_age": [18, 30, 50, 80]}`.
Explicit ALE edges must cover all selected nonmissing values. Without an ICE
`reference`, centering uses the middle numeric grid point or first category.
All supplied weight arrays align to the original explanation population, before
`max_samples` sampling. A zero-weight SHAP segment has undefined summary values
(nulls). A sampled population needs positive total averaging weight.

For a one-variable ALE with a long right tail, zoom the variable axis:

```python
loss_ratio = report.ale("prior_loss_ratio", xlim=(0, 2))
loss_ratio.plot()
loss_ratio.export("output/loss_ratio_ale")  # Figures retain these limits.
loss_ratio.plot(xlim=(None, 1))  # Override this figure only; no recalculation.
loss_ratio.plot(xlim=(None, None))  # Show the full axis again.
```

Either bound can be `None` to leave that end automatic. Limits must increase
and be finite when specified. They affect only the main effect plot; the
support panel and exported tables retain all calculated bins and rows. This
option does not change ALE averaging or centering, and is not supported for
two-feature heatmaps or categorical plots. Zooming cannot recover detail lost
to coarse bins: for a variable concentrated at zero, supply explicit `edges`
with finer boundaries near zero and a final boundary covering the maximum.
Each bin must still contain observations with positive total weight.

## Outputs and progress

All results contain reusable Polars tables, settings, scale/units, sampling
indices, feature lineage, and `.export(directory)`. Exports include CSV tables,
JSON metadata, and PNG/SVG figures. SHAP exports importance/beeswarm figures;
individual waterfalls and dependence figures can be saved with `output_path`.
Effects export their plot and support tables. Figure methods return Matplotlib
figures; callers should close figures they no longer need.

| Result | Table contents | Export files |
| --- | --- | --- |
| `EffectResult.table` | `value`, optional `value_2`, and `prediction` for PDP/ICE or `effect` for ALE; ICE also has local `row`, `source_row`, and `weight` | `effects.csv` |
| `EffectResult.support` | Grid/bin coordinates, `count`, `weight`, `unsupported` | `support.csv` |
| `ShapResult.contributions` | `repeat`, local `row`, `feature`, signed `contribution` | `contributions.csv` |
| `ShapResult.predictions` | `repeat`, `row`, `source_row`, `base_value`, `prediction`, `residual` | `predictions.csv` |
| `ShapResult.samples` | Diagnostic feature values aligned to local row numbers | `samples.csv` |
| `ShapResult.summary()` | `feature`, `mean_abs_contribution`, `mean_contribution`, `repeat_std` | `importance.csv` |

Each export also writes `settings.json`. Effects write `effect.png` and
`effect.svg`; SHAP writes `importance.png/.svg` and `beeswarm.png/.svg`. SHAP
waterfall/dependence plots are saved separately with `output_path=...`.
Settings retain units, scale, input row indices, averaging weights, feature
metadata, sampling and engine configuration; SHAP also records background row
indices and version. Exports contain feature values and row positions but do
not bundle a reusable model or the full reference dataset. Save model artifacts
and retain your reference-data definition separately to reproduce results.

Pass `progress_callback` and `should_stop` to `.explain()` to use the existing
`ProgressEvent` and `PipelineCancelled` conventions. Predictions are batched.

```python
report = fitted.explain(
    scoring_data, background=training_reference,
    progress_callback=lambda event: print(event.stage, event.message, event.current, event.total),
    should_stop=lambda: False,  # replace with your cancellation flag
)
```

## Reloading a saved model or ensemble

```python
from ins_gbm import load_model, load_ensemble

restored = load_model("output/frequency_model")
restored_report = restored.explain(scoring_data, background=training_reference)
restored_report.pdp("driver_age").export("output/restored/age")

# For an ensemble saved with ensemble.save(...):
restored_ensemble = load_ensemble("output/frequency_ensemble")
ensemble_report = restored_ensemble.explain(scoring_data, background=training_reference)
```

No `training_data=` attachment is needed for explanations; supply original
scoring/reference data explicitly. The explanation results are separate exports,
not automatically part of `.save()` or `EvaluationReport.export()`.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| SHAP import error | Install the `explain` extra in the same environment/kernel used to run the model. |
| SHAP asks for a background | Pass representative original `ModelData` as `background=` to `.explain()`. |
| Unknown feature | Inspect `report.feature_names`; reducers replace raw names with processed outputs. |
| Incompatible PCA basis | Explain members individually, or use genuinely compatible fitted transformations; renaming components is insufficient. |
| Explicit `engine="tree"` fails | Use `engine="auto"` or `"permutation"`; read `settings["fallback_reason"]` on auto results. |
| SHAP takes too long | Reduce `max_samples`, `max_background`, or `n_permutations`; increase budgets after a first-run check. |
| ALE has an empty/zero-weight cell | Reduce `bins` or use suitable `edges`; two-dimensional joint support is more demanding. |
| ALE requested for a category | Use PDP or ICE for nominal categorical features. |
| More than 30 categorical levels | Provide an explicit categorical grid in `grids` (PDP) or `grid` (ICE). |
| Gamma rejects `rate` | Use `response` for severity or `link` for log severity. |
| Contributions seem to use the wrong units | Inspect `settings["prediction_type"]`/`["units"]`; link contributions are log differences. |
| Centered profiles differ from raw predictions | ICE subtracts its chosen grid reference; ALE removes its weighted mean. |

These diagnostics explain fitted model behavior. They do not establish causal
effects or predictive accuracy; use the existing evaluation tools for validation.
