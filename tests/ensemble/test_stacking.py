from ins_gbm.data.loader import load_model_data
from ins_gbm.ensemble.stacking import StackingEnsemble
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import ModelPipeline, ModelRecipe


def test_stacking_uses_full_training_data(poisson_parquet):
    data = load_model_data(path=str(poisson_parquet), target="claim_count", exposure="exposure", feature_cols=["x1", "x3"], objective="poisson")
    pipelines = [ModelPipeline(data=data, recipe=ModelRecipe(model=LightGBMModel(objective="poisson"))).run() for _ in range(2)]
    ensemble = StackingEnsemble(cv_folds=2).fit(pipelines)
    assert len(ensemble.predict(data)) == data.n_rows
