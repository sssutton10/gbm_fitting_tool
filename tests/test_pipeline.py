import polars as pl
import pytest
from ins_gbm.data.loader import load_model_data
from ins_gbm.data.splitter import TrainTestSplit
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import FittedPipeline, ModelPipeline, ModelRecipe
from ins_gbm.tuning.tuner import HyperparameterTuner


# ── Basic pipeline ──────────────────────────────────────────────────────────────

def test_pipeline_run_returns_fitted_pipeline(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    pipeline = ModelPipeline(
        data=data,
        split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    )
    result = pipeline.run()
    assert isinstance(result, FittedPipeline)


def test_pipeline_result_has_train_and_test(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    result = ModelPipeline(
        data=data,
        split=TrainTestSplit(train_ratio=0.7, seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()
    assert result.train_data.n_rows > 0
    assert result.test_data.n_rows > 0


def test_pipeline_split_proportions(poisson_parquet):
    """70/30 split on 400 rows → 280 train, 120 test."""
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    result = ModelPipeline(
        data=data,
        split=TrainTestSplit(train_ratio=0.7, seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()
    assert result.train_data.n_rows == 280
    assert result.test_data.n_rows == 120


# ── Report and metrics ──────────────────────────────────────────────────────────

def test_pipeline_report_has_metrics(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    result = ModelPipeline(
        data=data,
        split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()
    metrics = result.report.metrics()
    assert isinstance(metrics, pl.DataFrame)
    assert "metric" in metrics.columns
    assert len(metrics) > 0


def test_pipeline_fitted_model_predicts_on_test(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    result = ModelPipeline(
        data=data,
        split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()
    preds = result.fitted_model.predict(result.test_data, "response")
    assert len(preds) == result.test_data.n_rows


# ── Tuning ──────────────────────────────────────────────────────────────────────

def test_pipeline_with_tuning_stores_history(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    result = ModelPipeline(
        data=data,
        split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            tuning=HyperparameterTuner(n_trials=2, cv_folds=2, seed=42),
        ),
    ).run()
    assert result.tuning_history is not None
    assert len(result.tuning_history) == 2


def test_pipeline_without_tuning_has_no_history(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    result = ModelPipeline(
        data=data,
        split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()
    assert result.tuning_history is None


# ── Recipe stored in result ──────────────────────────────────────────────────────

def test_pipeline_recipe_stored(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    recipe = ModelRecipe(model=LightGBMModel(objective="poisson"))
    result = ModelPipeline(data=data, split=TrainTestSplit(seed=42), recipe=recipe).run()
    assert result.recipe is recipe


# ── Metadata ────────────────────────────────────────────────────────────────────

def test_pipeline_metadata_is_populated(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    result = ModelPipeline(
        data=data,
        split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()
    assert result.metadata is not None
    assert result.metadata.objective == "poisson"
    assert len(result.metadata.feature_names) > 0
