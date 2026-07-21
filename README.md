# GBM Fitting

Polars-native Python library for insurance frequency and severity GBM modeling.

## Install

```bash
pip install -e ".[all,dev]"
```

## Run tests

```bash
pytest
```

## Fit and evaluate

`ModelPipeline` fits on every supplied training row. Keep a final holdout
separate and evaluate it explicitly after fitting:

```python
fitted = ModelPipeline(data=training_data, recipe=recipe).run()
report = fitted.evaluate(holdout_data)
metrics = report.metrics()
```

## Reuse data across model fits

Load the full candidate feature pool once, then choose the predictor subset for
each fit. The selected columns are used consistently for tuning, transforms,
fitting, and later scoring.

```python
fit = ModelPipeline(data=data, recipe=recipe).run(
    feature_names=["age", "territory", "vehicle_age"],
)
```

To reduce only part of the feature frame while retaining other columns, wrap a
reducer in `PreprocessingStep`:

```python
from ins_gbm.preprocessing.pca import PCAReducer
from ins_gbm.preprocessing.steps import PreprocessingStep

recipe.preprocessing = [
    PreprocessingStep(
        name="usage_pca",
        preprocessor=PCAReducer(n_components=2),
        feature_names=["annual_miles", "commute_miles", "vehicle_age"],
    ),
]
```
