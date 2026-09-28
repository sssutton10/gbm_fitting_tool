# GBM correctness and public API simplification plan

Date: 2026-09-28

Status: Implemented on 2026-09-28. Local verification covered LightGBM and the
core library; CI now includes an all-extras job for XGBoost, CatBoost, and UMAP.

## Objective

Correct the prediction, validation, cross-validation, and ensemble defects found
in the review; provide a simpler public Python API; and update the architecture
documentation and executable examples to describe the resulting behavior.

Keep the package name `ins_gbm`, Polars public data structures, reusable recipes,
explicit final holdouts, and compact persistence. This remains a Python library;
a new modeling CLI, GUI, or complete scikit-learn compatibility layer is outside
this change.

## Review baseline

- Existing suite: 228 passed, 7 skipped. XGBoost, CatBoost, and UMAP were absent.
- The local `.venv` has Python 3.12 packages but its interpreter symlink resolves
  to Python 3.14. Tests ran with a separately installed Python 3.12 interpreter.
- Reproduced LightGBM Poisson double exponentiation: native mean prediction
  0.250006 became 1.284178 through the wrapper.
- Reproduced stacking fold fits receiving `params=None` despite configured params.
- Reproduced CV reporting ignoring two predefined `ModelData.cv_fold` groups.
- Reproduced validation accepting null targets, negative weights, and NaN offsets.
- Reproduced Gamma Random Forest link predictions equaling response predictions.
- Static inspection additionally found explicit offsets ignored by XGBoost and
  CatBoost, Gamma link errors in those wrappers, tuning ignored by CV reporting,
  and the same parameter-loss issue in OOF blending.

Passing the current suite is a baseline, not the acceptance criterion. Add tests
that reproduce these failures and independently check the corrected behavior.

## Proposed public workflow

The following is the target API, not currently executable code:

```python
from ins_gbm import (
    CVConfig,
    HyperparameterTuner,
    LightGBMModel,
    ModelRecipe,
    OneHotEncoder,
    load_model,
    load_model_data,
)

training = load_model_data(
    "training.parquet",
    target="claim_count",
    exposure="exposure",
    objective="poisson",
    cv_fold="fold_id",
)
holdout = load_model_data(
    "holdout.parquet",
    target="claim_count",
    exposure="exposure",
    objective="poisson",
)

recipe = ModelRecipe(
    model=LightGBMModel(),
    encoder=OneHotEncoder(),
    tuning=HyperparameterTuner(n_trials=20),
)

# Optional, more expensive: outer CV evaluates the complete tuning procedure.
cv = recipe.cross_validate(training, cv=CVConfig(n_splits=5, seed=42))
fitted = recipe.fit(training, feature_names=["age", "territory"])
report = fitted.evaluate(holdout)
predictions = fitted.predict(holdout.features, exposure=holdout.exposure)
fitted.save("output/frequency_model")
restored = load_model("output/frequency_model")
```

The example above illustrates the API. The runnable example must use the same
feature subset in CV and final fitting so its results describe the same recipe.
Do not introduce a second training implementation behind the convenience methods.

## Decisions to implement consistently

### Prediction and objective contract

- Resolve the objective once: explicit model objective, otherwise data objective,
  otherwise the existing Poisson default. Reject conflicting explicit objectives
  rather than fitting one objective and evaluating another.
- For Poisson, `response` is expected count when exposure is supplied; `rate` is
  expected count per unit exposure. Without exposure these are equal.
- Define `link = log(response)` for both objectives. With Poisson exposure this
  includes `log(exposure)`. Users needing log rate can take the log of `rate`.
  This deliberately changes LightGBM/Random Forest's previous Poisson link
  convention and must be documented in migration notes.
- Explicit offsets are additive on the link scale. For Poisson, combine offset
  with log exposure; for Gamma, use the offset alone. Apply each contribution
  exactly once in training and prediction.
- Gamma rejects `rate`; all wrappers reject unknown prediction types.
- Random Forest remains an approximation benchmark. Reject explicit offsets it
  cannot honor. Document CatBoost's Gamma approximation using Tweedie power 1.99.

### Shared cross-validation contract

- Add `CVConfig(n_splits=5, seed=42, folds="auto")` and a single internal split
  resolver. Supported fold policies: `auto`, `random`, and `predefined`.
- `auto` uses `ModelData.cv_fold` when present, otherwise shuffled K-fold splits.
  `random` explicitly ignores supplied fold IDs; `predefined` requires them.
- Fold IDs and comparison predictions remain row-level metadata, never predictors.
- Validate split feasibility and nonempty train/validation partitions. Null fold
  IDs cannot leave rows silently unscored. Every row must have exactly one outer
  OOF prediction.
- Nested tuning uses only the outer training rows. Reuse their predefined fold
  groups when at least two remain; otherwise raise an actionable error requiring
  an explicit random inner-CV configuration or more predefined groups.
- Keep explicit final holdout ownership with the caller.

### Selection and tuning contract

- Default recipe fitting with tuning refits encoder, supervised selector, and
  preprocessors inside each tuning fold. Fit the final transforms and model on
  all training rows after choosing hyperparameters.
- Retain the existing select-once workflow as an explicit
  `selection_scope="fixed"` option. Record that its tuning scores are conditional
  on selection performed using all supplied training rows.
- `retune()` intentionally retains its frozen encoder/selection behavior; make
  that limitation visible in its documentation and metadata. It must not claim
  to produce an unbiased estimate of the complete selection procedure.
- Manual encoded feature subsets remain fixed user input. Replay the same subset
  in fold fits; distinguish absent fitted one-hot levels from invalid feature
  names so folds cannot silently change the feature contract.
- `HyperparameterTuner(metric=None)` infers Poisson or Gamma deviance from the
  resolved objective. Explicit incompatible deviance choices raise an error.
- Treat `recipe.params` as fixed/base parameters. Trial suggestions override
  overlapping keys, and the effective merged parameters are used consistently
  in trials, the final fit, metadata, and later refits. This changes the previous
  behavior of discarding manual params whenever tuning was enabled.

## Implementation sequence

### 1. Establish a reproducible test environment

Files: `pyproject.toml`, `uv.lock`, development instructions; optional CI workflow.

- [ ] Create an isolated review environment with a compatible Python interpreter;
  do not destroy the existing `.venv` as part of setup.
- [ ] Install the project and all declared modeling/test extras from the lockfile
  where possible. Reconcile the lockfile only if dependency changes are required.
- [ ] Record Python and backend versions and run the existing suite as a baseline.
- [ ] Provide a minimal-dependency test job and an all-extras job. The latter must
  execute XGBoost, CatBoost, and UMAP tests rather than silently skipping them.
- [ ] Document environment recreation after a system Python upgrade and use
  interpreter-qualified test commands.

Acceptance: a clean environment runs the suite without borrowing packages through
`PYTHONPATH`; optional integrations are exercised in the all-extras run.

### 2. Correct native model predictions and offsets

Files: `models/base.py`, `models/lightgbm.py`, `models/xgboost.py`,
`models/catboost.py`, `models/random_forest.py`, `tests/models/`.

- [ ] Centralize objective resolution, prediction-type checks, and shared scale
  conversion rules without obscuring backend-specific offset handling.
- [ ] Correct LightGBM's raw-score/response distinction and double exponentiation.
- [ ] Implement explicit offsets in XGBoost base margins and CatBoost baselines
  for both objectives, including exposure plus offset for Poisson.
- [ ] Ensure prediction baselines are handled correctly when training used
  exposure/offsets but scoring requests unit exposure or a different offset.
- [ ] Correct Gamma link predictions in all wrappers and align Poisson link
  predictions with the documented contract.
- [ ] Make offset capability flags truthful and fail clearly for unsupported use.
- [ ] Compare wrapper predictions to independently obtained native backend scores,
  including heterogeneous exposures and nonzero offsets at fit and score time.
- [ ] Test `response = rate * exposure`, `exp(link) = response`, unit exposure,
  absent exposure, and valid low Poisson rates below one.

Acceptance: the reproduced LightGBM failure is fixed; scale and offset tests pass
for every installed backend. Existing positivity-only tests are insufficient.

### 3. Strengthen input validation at public boundaries

Files: `data/model_data.py`, `data/loader.py`, `models/base.py`, public fit,
prediction, tuning, and evaluation entry points; corresponding tests.

- [ ] Require targets to be numeric, non-null, finite, and objective-valid.
- [ ] Require exposure to be numeric, finite, non-null, and strictly positive.
- [ ] Require weights to be numeric, finite, non-null, nonnegative, and have
  positive total weight for the rows being fitted or evaluated.
- [ ] Require offsets to be numeric, finite, non-null, and row-aligned.
- [ ] Validate comparison prediction values and lengths before metric computation.
- [ ] Reject empty training data, empty feature selections, duplicate feature
  names, and inconsistent explicit objectives with useful error messages.
- [ ] Separate training/evaluation checks from scoring checks so target-free
  prediction does not depend on fabricated valid targets.
- [ ] Separate structural fold-field validation from split feasibility checks:
  a held-out slice may legitimately contain only one fold ID.
- [ ] Preserve existing feature missing-value handling; do not impose target
  finiteness rules on predictor columns supported by the backend.
- [ ] Validate at public boundaries while avoiding repeated scans in every
  transform and low-level numerical helper.

Acceptance: the invalid inputs reproduced during review fail before backend
training, and valid scoring frames, zero observation weights, and fold slices work.

### 4. Unify folds, parameter resolution, and recipe fitting

Files: new `data/folds.py` (or equivalently scoped internal module), `pipeline.py`,
`preprocessing/chain.py`, `tuning/tuner.py`, `tuning/_process_worker.py`,
`evaluation/cv_report.py`, relevant tests.

- [ ] Implement the shared CV configuration and resolver described above.
- [ ] Adapt old `cv_folds`, `n_folds`, `use_data_folds`, and `fold_col` arguments
  through one compatibility layer. Reject conflicting old/new configurations.
  Preserve explicitly requested old policies; warn on changes to implicit defaults.
- [ ] Adapt legacy feature-based fold/benchmark columns into row-level metadata;
  fail when conflicting metadata sources are supplied.
- [ ] Ensure CV honors the existing ordered `feature_names` subset rather than
  promoting every column in the feature frame to a predictor.
- [ ] Refactor a shared recipe fit operation for pipeline, CV, and ensemble use.
  Keep progress callbacks, cancellation, process tuning, and compact transforms.
- [ ] Implement fold-local selection as the default and explicit fixed-selection
  tuning as the alternative, recording selection scope in result metadata.
- [ ] Infer default tuning metrics and preserve merged fixed/trial parameters.
- [ ] Make `CrossValidationReport` evaluate `recipe.tuning` through inner CV when
  supplied. Store outer-fold parameters and fold provenance for inspection.
- [ ] Use the resolved model objective for all metrics and reports.
- [ ] Test predefined fold membership, training-only fitting of supervised
  transforms, nesting, feature subset preservation, and complete OOF coverage.
- [ ] Exercise identical semantics through thread and subprocess tuning backends.

Acceptance: fitting, tuning, and reporting share split semantics; no workflow
silently drops configured tuning, features, or fixed parameters.

### 5. Repair ensemble OOF refits

Files: `ensemble/_utils.py`, `ensemble/blending.py`, `ensemble/stacking.py`,
`ensemble/pipeline.py`, `tests/ensemble/`.

- [ ] Use the shared fold resolver and recipe fit operation in both ensemble types.
- [ ] Add an explicit refit policy: default `refit="fixed"` reuses the fitted
  pipeline's effective model parameters; optional `refit="retune"` runs tuning
  inside each outer training fold when the recipe has a tuner.
- [ ] Document that fixed refits of globally tuned parameters yield OOF
  predictions conditional on those parameters, not a fully nested estimate.
- [ ] Refit learned selection and preprocessing on each outer training partition,
  and preserve manual raw/encoded subsets. Do not inadvertently refit all encoded
  columns when the supplied pipeline uses a fixed encoded subset.
- [ ] Validate nonempty base models, compatible objectives, row counts, and
  training row alignment. Reject detectable target/exposure/weight/fold ordering
  mismatches; document that identical row-level values cannot prove identity.
- [ ] Verify fold fits receive effective parameters and prediction columns match
  the base models used at inference. Cover both stacking and OOF blending.

Acceptance: configured hyperparameters and feature choices survive fold refits;
the review reproduction no longer records default-only fits.

### 6. Add the simpler public API

Files: `src/ins_gbm/__init__.py`, selected subpackage `__init__.py` files,
`pipeline.py`, `persistence/io.py`, `tests/test_pipeline.py`, persistence tests.

- [ ] Export common data, model, recipe, preprocessing, selection, tuning, CV,
  reporting, ensemble, and loading objects through documented public imports.
- [ ] Keep optional backend imports lazy so `import ins_gbm` works without all
  modeling extras installed.
- [ ] Add `ModelRecipe.fit(data, ...)` delegating to the shared pipeline engine.
- [ ] Add `ModelRecipe.cross_validate(data, cv=..., feature_names=...)` delegating
  to the corrected CV report workflow.
- [ ] Extend `FittedPipeline.predict()` to accept either `ModelData` or a Polars
  feature frame with keyword-only exposure/offset/weight metadata. Reject
  ambiguous attempts to override fields of a supplied `ModelData` object.
- [ ] Retain `predict_raw()` as a forwarding compatibility method, adding offset
  support and preserving its existing positional arguments.
- [ ] Add `fitted.save(path)` and public `load_model(path, training_data=None)`
  wrappers around existing persistence functions.
- [ ] Preserve deep imports, `ModelPipeline(...).run()`, `save_pipeline()`, and
  `load_pipeline()`. Keep them supported; do not force an immediate migration.
- [ ] Retain `feature_names` as the canonical predictor argument. Document the
  legacy loader spelling `feature_cols` rather than adding more synonymous names.

Acceptance: the common workflow needs no deep imports or pipeline orchestrator
construction; old documented workflows still run except intentional correctness
and statistical-default changes, which have clear migration notes.

### 7. Update persistence metadata and compatibility notes

Files: `persistence/metadata.py`, `persistence/io.py`, persistence tests, docs.

- [ ] Record effective parameters, resolved objective/metric, selection scope,
  fold/refit policy, and relevant package versions using compact serializable data.
- [ ] Introduce an artifact/behavior version and handle absent new metadata fields
  for older artifacts without claiming corrected prediction behavior.
- [ ] Check how old cloudpickled prediction closures behave under new code.
  Warn for affected legacy artifacts and recommend refitting; do not silently
  relabel old models as corrected or promise automatic mathematical repair.
- [ ] Round-trip predictions and evaluation for new artifacts, with offsets and
  target-free scoring. Preserve omission of raw training rows and optional
  reattachment for retuning/ensembles.

Acceptance: new artifacts preserve corrected semantics, and old artifact behavior
is explicit rather than silently assumed to match the new implementation.

### 8. Rewrite documentation and usage examples

Files: `README.md`, `PROJECT_STRUCTURE.md`, `CLAUDE.md`,
`examples/example_usage.ipynb`, new `examples/example_usage.py`.

- [ ] Make README start with a complete minimal load/fit/evaluate/predict/save
  example using public imports, followed by tuning and advanced options.
- [ ] Update `PROJECT_STRUCTURE.md`'s package map, main flow, public API examples,
  data validation rules, prediction-scale definitions, offsets, selection order,
  shared CV semantics, nested tuning, ensemble parameter policy, and persistence.
- [ ] Replace outdated statements that recommend deep imports or describe known
  defects as ongoing behavior. Keep genuine limitations such as RF approximation,
  CatBoost Gamma approximation, UMAP retained state, and pickle compatibility.
- [ ] Include an old-to-new API migration table and distinguish numerical bug fixes
  from changed defaults. Explicitly call out the Poisson link convention.
- [ ] Align `CLAUDE.md` with the new contracts so it no longer instructs future
  changes to perform global selection before all tuning by default.
- [ ] Update the existing notebook to public imports and the simpler workflow.
  Remove the workaround that copies fold IDs and benchmark predictions into
  feature columns. Explain fixed versus nested tuning and selection explicitly.
- [ ] Add `examples/example_usage.py` with deterministic synthetic data so it runs
  without private files. Use a main guard, explicit training/holdout separation,
  a short default run, and an explicit output directory argument.
- [ ] Show frequency and severity, benchmark evaluation, saving/loading, offsets,
  and target-free predictions. Put expensive CV/ensemble/optional-backend examples
  behind explicit demo options or clearly marked notebook sections.
- [ ] Use small trial/tree counts for demonstration; label them as demonstration
  settings rather than recommended production hyperparameters.
- [ ] Execute the script and notebook from a fresh environment. Keep notebook
  outputs consistent with the new implementation and free of machine-specific paths.

Acceptance: both examples run from start to finish, the structure document matches
the actual code, and copied README examples contain all necessary setup/imports.

### 9. Final verification and handoff

- [ ] Run focused regression tests as each implementation stage lands, then the
  complete suite once all changes are integrated.
- [ ] Run the all-extras backend contract checks; explicitly report any remaining
  skipped or unavailable integration instead of claiming full verification.
- [ ] Run end-to-end frequency and severity workflows covering tuning, selection,
  holdout metrics, ensembles, persistence, and target-free scoring.
- [ ] Run legacy API compatibility checks and new public API import checks.
- [ ] Confirm correct cancellation/progress reporting and process backend behavior
  after sharing fit/split logic.
- [ ] Check for unexpected runtime growth on the same small workload. Explain the
  expected extra cost of fold-local selection and nested CV without weakening
  their isolation guarantees.
- [ ] Deliver a concise change summary, test results, migration notes, and any
  remaining limitations. Do not claim old saved model outputs have been repaired.

## Suggested implementation boundaries

Use reviewable changes in this order: environment and backend prediction fixes;
validation; shared CV and fitting semantics; ensembles; public convenience API
and persistence metadata; documentation/examples and integrated verification.

Documentation and regression tests should accompany the behavior they describe.
The final docs/example pass checks consistency across the completed work rather
than postponing all documentation until the end.
