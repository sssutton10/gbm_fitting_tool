import polars as pl
import pytest
from ins_gbm.data.loader import load_model_data
from ins_gbm.data.splitter import TrainTestSplit
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import FittedPipeline, ModelPipeline, ModelRecipe
from ins_gbm.ensemble.pipeline import EnsemblePipeline


def _make_pipeline(poisson_parquet) -> FittedPipeline:
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    return ModelPipeline(
        data=data,
        split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()


# ── Blending via EnsemblePipeline ──────────────────────────────────────────────

def test_ensemble_pipeline_blending_run(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    ensemble = EnsemblePipeline(
        fitted_pipelines=[p1, p2],
        method="blending",
        blend_mode="fixed",
        blend_weights=[0.5, 0.5],
    )
    result = ensemble.run()
    assert result is not None


def test_ensemble_pipeline_blending_has_report(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    ensemble = EnsemblePipeline(
        fitted_pipelines=[p1, p2],
        method="blending",
        blend_mode="fixed",
        blend_weights=[0.5, 0.5],
    )
    result = ensemble.run()
    metrics = result.report.metrics()
    assert "metric" in metrics.columns
    assert len(metrics) > 0


def test_ensemble_pipeline_blending_predictions_length(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    result = EnsemblePipeline(
        fitted_pipelines=[p1, p2],
        method="blending",
        blend_mode="fixed",
        blend_weights=[0.5, 0.5],
    ).run()
    preds = result.predict(p1.test_data)
    assert len(preds) == p1.test_data.n_rows


# ── Stacking via EnsemblePipeline ──────────────────────────────────────────────

def test_ensemble_pipeline_stacking_run(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    ensemble = EnsemblePipeline(
        fitted_pipelines=[p1, p2],
        method="stacking",
        cv_folds=2,
        seed=42,
    )
    result = ensemble.run()
    assert result is not None


def test_ensemble_pipeline_stacking_has_report(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    result = EnsemblePipeline(
        fitted_pipelines=[p1, p2],
        method="stacking",
        cv_folds=2,
        seed=42,
    ).run()
    metrics = result.report.metrics()
    assert len(metrics) > 0


# ── Invalid method ─────────────────────────────────────────────────────────────

def test_ensemble_pipeline_invalid_method(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    with pytest.raises(ValueError, match="method"):
        EnsemblePipeline(fitted_pipelines=[p1], method="unknown").run()
