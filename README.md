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

Model wrappers also build their design matrix at fit time. This is useful for
lightweight iterations that do not need tuning or learned feature selection:

```python
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.preprocessing.encoder import OneHotEncoder
from ins_gbm.preprocessing.pca import PCAReducer
from ins_gbm.preprocessing.steps import PreprocessingStep

model = LightGBMModel(objective="poisson").fit(
    data,
    params={"n_estimators": 200},
    feature_names=["age", "territory", "annual_miles", "vehicle_age"],
    encoder=OneHotEncoder(),
    preprocessing=[
        PreprocessingStep(
            name="usage_pca",
            preprocessor=PCAReducer(n_components=1),
            feature_names=["annual_miles", "vehicle_age"],
        ),
    ],
)
predictions = model.predict(data)  # raw ModelData; fitted transforms replayed
```

The expanded matrix is temporary. A fitted pipeline retains the reusable raw
training-data reference for OOF ensembles; `fitted.train_data` reconstructs the
transformed matrix on demand without caching it. Persistence omits that raw
dataset:

```python
from ins_gbm.persistence.io import load_pipeline, save_pipeline

save_pipeline(fitted, "output/my_model")

loaded = load_pipeline("output/my_model")  # scoring and evaluation
loaded_for_oof = load_pipeline(
    "output/my_model",
    training_data=training_data,
)  # also enables train_data and later OOF ensemble fitting
```

When reattaching data, supply the original training rows in their original
order. Fitted transforms and the input schema remain in the compact artifact,
so prediction does not require training data. UMAP is an exception to the
general compact-state rule: a fitted UMAP reducer may retain training-derived
matrices required by its transform implementation.

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

## Staged importance selection

Use `StagedImportanceSelector` to prune encoded model columns in one or more
fits before the final model is trained. Each stage owns its learner and fixed
parameters, so a shallow screen can be followed by a more realistic pruning
fit. The pipeline completes every selection stage first, then tunes the final
recipe model on that fixed selected feature set. Importance names are native to
the selected framework.

For a pool of roughly 500 eligible candidates, prefer one cheap global screen
followed by one or more narrower refinement fits. Do not divide columns into
arbitrary batches: importance scores from different batches are not directly
comparable, and removing competing or interacting variables changes the
rankings. Separate the final holdout before selection, and remember that stage
limits count encoded model columns rather than raw source variables. The
complete runnable workflow is in `examples/example_usage.ipynb`.

```python
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import ModelPipeline, ModelRecipe
from ins_gbm.selection import ImportanceSelectionStage, StagedImportanceSelector

recipe = ModelRecipe(
    model=LightGBMModel(objective="poisson"),
    selection=StagedImportanceSelector(stages=[
        ImportanceSelectionStage(
            name="screen",
            model=LightGBMModel(objective="poisson"),
            params={"n_estimators": 40, "num_leaves": 8},
            max_features=100,
            importance_type="gain",
        ),
        ImportanceSelectionStage(
            name="final_prune",
            model=LightGBMModel(objective="poisson"),
            params={"n_estimators": 250, "num_leaves": 32},
            max_features=30,
            importance_type="split",
        ),
    ]),
)

fitted = ModelPipeline(data=training_data, recipe=recipe).run()
fitted.selection_results[0].ranking  # feature, importance, rank, selected
```
