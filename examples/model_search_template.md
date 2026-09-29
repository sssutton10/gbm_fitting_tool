# Model and blend search template

Edit [model_search_config.json](model_search_config.json), then run from the
repository root with the package and all model extras installed:

```bash
python examples/model_search_template.py search --config examples/model_search_config.json
python examples/model_search_template.py evaluate --config examples/model_search_config.json
```

Use `search` for development. Run `evaluate` only after accepting the locked
development choice. It loads the saved configuration and evaluates the baseline,
locked winner, and locked best blend without changing the selection.
No procedure guarantees beating the refit on new data.

## Data and configuration

- Supply separate training, development-validation, and final-test Parquet files.
  For 150k independent rows, an illustrative allocation is 105k / 22.5k / 22.5k.
  Choose splits based on event counts, customer/policy grouping, and deployment
  dates, not just percentages. The script does not create or verify row-disjoint
  splits for you. Never include a previously inspected holdout as a fresh test.
- Replace the example `candidate_features` with your explicit 677-column raw
  candidate pool, including every `existing_features` column. Exclude IDs,
  benchmark predictions, target-derived fields, and information unavailable at
  prediction time. Input lists contain raw names; selection caps count encoded
  model columns, including individual one-hot indicators.
- String/Boolean categories are inferred. Put integer-coded categorical columns
  in `categorical_features`; those columns are converted to strings. Encoding is
  learned independently in every fitting fold. An unseen category scores as all
  zeros for that raw variable, following the current library encoder.
- For severity, set `objective` to `gamma`, use a positive severity target, and
  normally set `exposure` to null. Use the same target/weight definition for every
  candidate. This template does not supply explicit offsets because Random Forest
  cannot support them. Random Forest is an approximation benchmark; CatBoost's
  Gamma objective is a Tweedie approximation in this library. All are judged on
  the same deviance.
- Set `baseline_family` to the existing model's family. Optionally put its known
  hyperparameters in `known_baseline_params`. They receive a separate untuned
  candidate, ensuring the search does not need to rediscover them. Refit from
  scratch using today's code; do not substitute a pre-fix serialized model.
- Set `selection_encoded_caps` to `[200, 100]` to start with ALL encoded columns,
  prune to 200, refit the selection model, and prune to 100. Use `[200, 50]` for
  a final 50, or `[50]` for a single pruning stage. These are TOTAL column caps,
  not counts of new features. If fewer columns are available, all available
  columns survive that stage. Counts must be positive and non-increasing.
  Replace the old `selection_new_feature_caps` setting and remove
  `additional_feature_groups`; legacy settings now raise a migration error.
- `fold_column` optionally names predefined training fold IDs (at least three
  IDs for nested CV). Keep related observations in the same fold. The built-in
  splitter trains on all other folds: it is NOT forward-only temporal CV.
  For time-sensitive work, provide chronological validation/test files and adapt
  the inner/outer split implementation before interpreting CV as temporal evidence.

## Search process

1. Fit each of LightGBM, XGBoost, CatBoost, and Random Forest starting with the
   full raw pool, encoding it, and pruning to the configured encoded count.
   Separately fit the existing-feature benchmark in its original model family.
   Screening uses the same 30k training-row
   sample, six tuning trials and three folds per candidate by default. Validation
   is a separate dataset. Sampling reduces cost, but can miss rare signals.
2. Selection ranks each encoded column independently using a LightGBM screening
   model's gain. It can retain `territory__A` while dropping `territory__B`, and
   it can drop existing-model features. No columns or categorical groups are
   protected. Each stage refits on the preceding stage's survivors. Selection is
   relearned inside tuning and outer CV folds, so selected names may differ by
   fold. All four families use this same selection method. Top-N is a ranking
   heuristic, not a significance test; ties can retain zero-gain columns to reach
   the requested count. Edit caps and screening settings on development data.
   The separate baseline is exempt from the cap. A blend's union of features can
   exceed the per-model cap when members select different columns.
3. Promote at least one candidate PER FAMILY to full training-data tuning with
   25 trials. Always include the baseline-family existing-feature candidate and
   any known-parameter baseline. There is one pruned strategy per family, so all
   four advance; `finalists_per_family` currently has no effect above one.
   Increase tuning budgets after a small initial run.
4. Generate nested out-of-fold predictions for each finalist. Each outer fold
   repeats encoding, selection, and full hyperparameter tuning on its training
   rows. Final full-data models and OOF predictions are saved. Optional additional
   `model_seeds` repeat this process and enter the blend pool independently.
5. Learn nonnegative response-scale blend weights that sum to one, minimizing
   Poisson/Gamma deviance on aligned nested OOF predictions. Try every pair, all
   finalists together, equal-weight variants, and all single-model endpoints.
   Multiple optimization starts are used. This is a bounded search for a strong
   blend, not a globally best ensemble guarantee. Unlike the library's current
   convenience blend helper, this template optimizes deviance rather than MSE.
6. Choose the single model or blend on development validation; separately retain
   the best genuine multi-model blend. Save weights and member paths before any
   test data is read. Test evaluation never changes the selected weights.

Screening, finalist selection, and blend selection all use development evidence.
The best development score is optimistic after this search. Nested OOF predictions
are used to TRAIN blend weights; their optimized score is not an independent
evaluation of the blend or the whole search. Only the untouched test evaluates
the locked result independently. Inspect calibration, lift, and uncertainty as
well as the primary deviance before deployment. For uncertainty, use a paired
bootstrap at the independent policy/customer level, or appropriate time blocks;
do not treat correlated rows or CV folds as independent observations.

## Runtime and memory for 150k rows / 677 raw columns

The wrappers build dense matrices, including CatBoost in this library: this
template does not use CatBoost's native categorical input. A 150k-by-5,000
float32 matrix alone is about 3 GB; fitting, encoding, library buffers and copies
raise peak memory beyond that. The script reports encoded width and a single
matrix estimate before fitting. `max_dense_matrix_gb` is a configurable guard,
not an estimate of total RAM needed. Audit high-cardinality variables; any learned
rare-level grouping must also be fitted inside folds.

Trials and models run sequentially, with bounded per-model threads. Full models
are saved and released before the next fit; only small prediction matrices stay
in memory for blending. Selection inside every trial can be expensive in the
current implementation. The default five screening candidates need about 90
tuning-fold fits, plus full fits and selector fits. With five finalists, the
full-data + nested-CV stage needs roughly 1,500 additional tuning-fold fits
at 25 trials, three inner folds, and three outer folds. Extra seeds
multiply the work. These are substantial searches, not quick scripts.

For a plumbing check, set trials to 1, tree ranges to [5, 5], selection_trees to 5,
and use a tiny dataset with sufficient events. Then restore realistic budgets.
The wrapper API does not currently expose a general early-stopping workflow;
the template tunes tree count. No GPU configuration is assumed.

## Outputs and scoring

`screening.csv` and `finalists.csv` summarize development performance.
`models/` holds fitted pipelines, selected encoded features, per-stage rankings,
and tuning histories.
Each finalist also has a `cv_report/` directory containing fold metrics, summary
metrics, aligned OOF predictions, fold assignments, and CV provenance. It does
not contain targets, exposure, weights, or raw features. Load two finalist CV
reports with `load_cv_result`, use
`compare_reports({"reference": reference, "candidate": candidate}, data=training_data)`
for standard metrics and pooled double lift, and call
`compare_cv_double_lift(reference, candidate, data=training_data)` for per-fold
detail. The same ordered training rows and fold
assignments are required; positive double-lift scores favor the candidate.
The existing CSV metric exports and `oof.parquet` remain available for simple
inspection and blend workflows.
`blend_search.csv` and `selection.json`
record blend comparisons, the locked choice, and weights in member order.
`test_metrics.csv` and `test_predictions.parquet` are written by `evaluate`.
Models are trained on the training partition, not refitted on validation/test;
this keeps final scoring consistent with the models used to select blend weights.
An existing output directory is rejected to avoid mixing experiments.

For new raw scoring data, load each member listed in `selection.json` with
`load_model`, obtain its response predictions, and sum them with the locked
weights. Poisson responses are expected counts; blend counts for the same rows
and exposures, then divide by exposure if rates are needed. Do not blend link
predictions. Model paths in the manifest are absolute; update them if moving
artifacts. Load only trusted pickle files.
