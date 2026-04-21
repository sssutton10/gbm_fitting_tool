import json
import os

import polars as pl
import pytest
from ins_gbm.data.loader import load_model_data
from ins_gbm.data.splitter import TrainTestSplit
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.persistence.io import load_pipeline, save_pipeline
from ins_gbm.pipeline import FittedPipeline, ModelPipeline, ModelRecipe


def _build_pipeline(poisson_parquet) -> FittedPipeline:
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    return ModelPipeline(
        data=data,
        split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()


# ── Save ───────────────────────────────────────────────────────────────────────

def test_save_creates_output_dir(poisson_parquet, tmp_path):
    result = _build_pipeline(poisson_parquet)
    out = str(tmp_path / "pipeline_out")
    save_pipeline(result, out)
    assert os.path.isdir(out)


def test_save_creates_metadata_json(poisson_parquet, tmp_path):
    result = _build_pipeline(poisson_parquet)
    out = str(tmp_path / "pipeline_out")
    save_pipeline(result, out)
    assert os.path.exists(os.path.join(out, "metadata.json"))


def test_save_metadata_json_is_valid(poisson_parquet, tmp_path):
    result = _build_pipeline(poisson_parquet)
    out = str(tmp_path / "pipeline_out")
    save_pipeline(result, out)
    with open(os.path.join(out, "metadata.json")) as f:
        meta = json.load(f)
    assert "objective" in meta
    assert "feature_names" in meta
    assert meta["objective"] == "poisson"


def test_save_creates_metrics_csv(poisson_parquet, tmp_path):
    result = _build_pipeline(poisson_parquet)
    out = str(tmp_path / "pipeline_out")
    save_pipeline(result, out)
    assert os.path.exists(os.path.join(out, "metrics.csv"))


def test_save_creates_pipeline_artifact(poisson_parquet, tmp_path):
    result = _build_pipeline(poisson_parquet)
    out = str(tmp_path / "pipeline_out")
    save_pipeline(result, out)
    assert os.path.exists(os.path.join(out, "pipeline.pkl"))


# ── Load ───────────────────────────────────────────────────────────────────────

def test_load_returns_fitted_pipeline(poisson_parquet, tmp_path):
    result = _build_pipeline(poisson_parquet)
    out = str(tmp_path / "pipeline_out")
    save_pipeline(result, out)
    loaded = load_pipeline(out)
    assert isinstance(loaded, FittedPipeline)


def test_load_predictions_identical(poisson_parquet, tmp_path):
    """Predictions from a loaded pipeline must be bit-for-bit identical."""
    result = _build_pipeline(poisson_parquet)
    out = str(tmp_path / "pipeline_out")
    save_pipeline(result, out)
    loaded = load_pipeline(out)

    orig_preds = result.fitted_model.predict(result.test_data, "response")
    loaded_preds = loaded.fitted_model.predict(loaded.test_data, "response")

    assert orig_preds.to_list() == loaded_preds.to_list()


def test_load_metadata_preserved(poisson_parquet, tmp_path):
    result = _build_pipeline(poisson_parquet)
    out = str(tmp_path / "pipeline_out")
    save_pipeline(result, out)
    loaded = load_pipeline(out)
    assert loaded.metadata.objective == "poisson"
    assert loaded.metadata.feature_names == result.metadata.feature_names
