import os

import pytest

from ins_gbm.data.loader import load_model_data
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.persistence.io import load_pipeline, save_pipeline
from ins_gbm.pipeline import ModelPipeline, ModelRecipe


def test_save_load_preserves_predictions_without_metrics_artifact(poisson_parquet, tmp_path):
    data = load_model_data(path=str(poisson_parquet), target="claim_count", exposure="exposure", feature_cols=["x1", "x3"], objective="poisson")
    fitted = ModelPipeline(data=data, recipe=ModelRecipe(model=LightGBMModel(objective="poisson"))).run()
    save_pipeline(fitted, str(tmp_path))
    loaded = load_pipeline(str(tmp_path))
    assert fitted.predict(data).to_list() == pytest.approx(loaded.predict(data).to_list(), rel=1e-6)
    assert os.path.exists(tmp_path / "pipeline.pkl")
    assert os.path.exists(tmp_path / "metadata.json")
    assert not os.path.exists(tmp_path / "metrics.csv")
