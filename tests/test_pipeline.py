import polars as pl
import pytest

from ins_gbm.data.loader import load_model_data
from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.data.schema import infer_schema
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import FittedPipeline, ModelPipeline, ModelRecipe
from ins_gbm.preprocessing.encoder import OneHotEncoder
from ins_gbm.preprocessing.pca import PCAReducer
from ins_gbm.preprocessing.steps import PreprocessingStep
from ins_gbm.tuning.tuner import HyperparameterTuner


def _data(path):
    return load_model_data(
        path=str(path), target="claim_count", exposure="exposure",
        feature_cols=["x1", "x3"], objective="poisson",
    )


def test_pipeline_fits_all_supplied_rows(poisson_parquet):
    data = _data(poisson_parquet)
    result = ModelPipeline(data=data, recipe=ModelRecipe(model=LightGBMModel(objective="poisson"))).run()
    assert isinstance(result, FittedPipeline)
    assert result.train_data.n_rows == data.n_rows
    assert not hasattr(result, "test_data")
    assert not hasattr(result, "report")


def test_pipeline_manual_params_reach_model(poisson_parquet):
    result = ModelPipeline(
        data=_data(poisson_parquet),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson"), params={"n_estimators": 7, "learning_rate": 0.123}),
    ).run()
    assert result.fitted_model.params.get("learning_rate") == 0.123


def test_explicit_evaluation_transforms_holdout_without_retaining_it(poisson_raw):
    schema = infer_schema(poisson_raw, ["x1", "x2", "x3"])
    data = ModelData(
        features=poisson_raw.select(["x1", "x2", "x3"]), target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"], weight=None, feature_names=["x1", "x2", "x3"],
        schema=schema, objective="poisson",
    ).validate()
    result = ModelPipeline(
        data=data, recipe=ModelRecipe(model=LightGBMModel(objective="poisson"), encoder=OneHotEncoder()),
    ).run()
    holdout = slice_model_data(data, range(100))
    report = result.evaluate(holdout)
    assert report.evaluation_data.n_rows == holdout.n_rows
    assert report.evaluation_data.features.columns != holdout.features.columns
    assert not hasattr(result, "evaluation_data")
    assert "metric" in report.metrics().columns


def test_predict_raw_matches_explicit_evaluation(poisson_raw):
    schema = infer_schema(poisson_raw, ["x1", "x2", "x3"])
    data = ModelData(
        features=poisson_raw.select(["x1", "x2", "x3"]), target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"], weight=None, feature_names=["x1", "x2", "x3"],
        schema=schema, objective="poisson",
    ).validate()
    result = ModelPipeline(data=data, recipe=ModelRecipe(model=LightGBMModel(objective="poisson"), encoder=OneHotEncoder())).run()
    holdout = slice_model_data(data, range(100))
    report = result.evaluate(holdout)
    raw_predictions = result.predict_raw(holdout.features, exposure=holdout.exposure)
    direct_predictions = result.fitted_model.predict(report.evaluation_data, "response")
    assert raw_predictions.to_list() == pytest.approx(direct_predictions.to_list(), rel=1e-6)


def test_tuning_still_uses_cv_and_stores_history(poisson_parquet):
    result = ModelPipeline(
        data=_data(poisson_parquet),
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            tuning=HyperparameterTuner(n_trials=2, cv_folds=2, seed=42),
        ),
    ).run()
    assert result.train_data.n_rows == _data(poisson_parquet).n_rows
    assert len(result.tuning_history) == 2


def test_comparison_predictions_are_taken_from_holdout(poisson_raw):
    data = _data_from_raw = ModelData(
        features=poisson_raw.select(["x1", "x3"]), target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"], weight=None, feature_names=["x1", "x3"],
        objective="poisson", comparisons=pl.DataFrame({"legacy": [1.0] * poisson_raw.height}),
    ).validate()
    result = ModelPipeline(data=data, recipe=ModelRecipe(model=LightGBMModel(objective="poisson"))).run()
    report = result.evaluate(slice_model_data(data, range(100)))
    assert set(report.metrics()["model"].unique()) == {"GBM", "legacy"}


def test_pipeline_run_can_select_reusable_feature_subset(poisson_raw):
    schema = infer_schema(poisson_raw, ["x1", "x2", "x3"])
    data = ModelData(
        features=poisson_raw.select(["x1", "x2", "x3"]),
        target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"],
        weight=None,
        feature_names=["x1", "x2", "x3"],
        schema=schema,
        objective="poisson",
    ).validate()

    result = ModelPipeline(
        data=data,
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            encoder=OneHotEncoder(),
            params={"n_estimators": 5},
        ),
    ).run(feature_names=["x1", "x2"])

    assert result.input_feature_names == ["x1", "x2"]
    assert result.raw_train_data.features.columns == ["x1", "x2"]
    assert result.metadata.input_feature_names == ["x1", "x2"]
    assert result.predict(data).len() == data.n_rows


def test_pipeline_targeted_preprocessor_retains_other_features(poisson_raw):
    data = ModelData(
        features=poisson_raw.select(["x1", "x3"]),
        target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"],
        weight=None,
        feature_names=["x1", "x3"],
        objective="poisson",
    ).validate()

    result = ModelPipeline(
        data=data,
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            preprocessing=[
                PreprocessingStep(
                    name="x1_pca",
                    preprocessor=PCAReducer(n_components=1),
                    feature_names=["x1"],
                ),
            ],
            params={"n_estimators": 5},
        ),
    ).run()

    assert result.train_data.features.columns == ["x1_pca__pca_1", "x3"]
    assert result.predict(data).len() == data.n_rows
