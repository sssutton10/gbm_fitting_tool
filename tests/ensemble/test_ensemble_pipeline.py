from ins_gbm.data.loader import load_model_data
from ins_gbm.data.model_data import slice_model_data
from ins_gbm.ensemble.pipeline import EnsemblePipeline
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import ModelPipeline, ModelRecipe


def _pipelines(path):
    data = load_model_data(path=str(path), target="claim_count", exposure="exposure", feature_cols=["x1", "x3"], objective="poisson")
    first = ModelPipeline(data=data, recipe=ModelRecipe(model=LightGBMModel(objective="poisson"), params={"n_estimators": 10})).run()
    second = ModelPipeline(data=data, recipe=ModelRecipe(model=LightGBMModel(objective="poisson"), params={"n_estimators": 15})).run()
    return data, first, second


def test_ensemble_run_does_not_evaluate(poisson_parquet):
    _, first, second = _pipelines(poisson_parquet)
    result = EnsemblePipeline([first, second], blend_weights=[0.5, 0.5]).run()
    assert not hasattr(result, "report")


def test_ensemble_evaluate_uses_explicit_holdout(poisson_parquet):
    data, first, second = _pipelines(poisson_parquet)
    result = EnsemblePipeline([first, second], blend_weights=[0.5, 0.5]).run()
    holdout = slice_model_data(data, range(100))
    report = result.evaluate(holdout)
    assert report.evaluation_data.n_rows == 100
    assert "metric" in report.metrics().columns
