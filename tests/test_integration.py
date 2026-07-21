from ins_gbm.data.loader import load_model_data
from ins_gbm.data.model_data import slice_model_data
from ins_gbm.ensemble.pipeline import EnsemblePipeline
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import ModelPipeline, ModelRecipe


def test_full_data_fit_followed_by_explicit_holdout_evaluation(poisson_parquet):
    data = load_model_data(path=str(poisson_parquet), target="claim_count", exposure="exposure", feature_cols=["x1", "x3"], objective="poisson")
    pipeline = ModelPipeline(data=data, recipe=ModelRecipe(model=LightGBMModel(objective="poisson"))).run()
    report = pipeline.evaluate(slice_model_data(data, range(100)))
    assert pipeline.train_data.n_rows == data.n_rows
    assert report.evaluation_data.n_rows == 100


def test_ensemble_explicit_holdout_evaluation(poisson_parquet):
    data = load_model_data(path=str(poisson_parquet), target="claim_count", exposure="exposure", feature_cols=["x1", "x3"], objective="poisson")
    pipelines = [ModelPipeline(data=data, recipe=ModelRecipe(model=LightGBMModel(objective="poisson"))).run() for _ in range(2)]
    report = EnsemblePipeline(pipelines, blend_weights=[0.5, 0.5]).run().evaluate(slice_model_data(data, range(100)))
    assert "metric" in report.metrics().columns
