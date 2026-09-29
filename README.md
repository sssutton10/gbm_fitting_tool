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
double_lift = compare_cv_double_lift(saved_cv, candidate_cv)
# double_lift has an overall score and one score per fold;
# positive scores favor the candidate.
```

`auto` uses `ModelData.cv_fold` when present and otherwise uses shuffled K-fold
splits. Encoding, supervised selection, and preprocessing are refit within tuning
folds by default. Set `selection_scope="fixed"` only when tuning scores should be
conditional on selection learned from all supplied training rows. Outer CV of a
tuned recipe is nested CV.

The CV artifact stores fold metrics, out-of-fold predictions, fold assignments,
and provenance. It does not store targets, exposure, weights, or features. If both
CV results are loaded from disk, pass the original training data to
`compare_cv_double_lift(saved_cv, candidate_cv, data=training)`. Results must use
the same ordered rows and fold assignments. `compare_reports()` compares standard
metrics; double lift is calculated separately because it depends on the model pair.
The overall double-lift score is calculated from all out-of-fold rows, rather
than averaging the fold scores.

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
one loaded candidate pool across fits. Fixed post-encoding names remain available
through `feature_stage="encoded"`.

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for architecture and migration
details. The runnable workflow is `examples/example_usage.py`, with a companion
notebook at `examples/example_usage.ipynb`.
