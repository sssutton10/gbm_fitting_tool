import polars as pl
import pytest
import numpy as np
from ins_gbm.data.loader import load_model_data
from ins_gbm.data.splitter import TrainTestSplit
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import FittedPipeline, ModelPipeline, ModelRecipe
from ins_gbm.ensemble.blending import BlendingEnsemble, FittedBlendingEnsemble


def _make_pipeline(poisson_parquet, params=None) -> FittedPipeline:
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    result = ModelPipeline(
        data=data,
        split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()
    return result


# ── Fixed mode ─────────────────────────────────────────────────────────────────

def test_fixed_blend_returns_fitted(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    ensemble = BlendingEnsemble(mode="fixed", weights=[0.5, 0.5])
    fitted = ensemble.fit([p1, p2])
    assert isinstance(fitted, FittedBlendingEnsemble)


def test_fixed_blend_weights_stored(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    fitted = BlendingEnsemble(mode="fixed", weights=[0.6, 0.4]).fit([p1, p2])
    assert abs(fitted.weights[0] - 0.6) < 1e-9
    assert abs(fitted.weights[1] - 0.4) < 1e-9


def test_fixed_blend_requires_weights():
    with pytest.raises(ValueError, match="weights"):
        BlendingEnsemble(mode="fixed", weights=None).fit([None, None])


def test_fixed_blend_weights_must_sum_to_one(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    with pytest.raises(ValueError, match="sum"):
        BlendingEnsemble(mode="fixed", weights=[0.3, 0.3]).fit([p1, p2])


def test_fixed_blend_equal_weights_gives_average(poisson_parquet):
    """Blending identical pipelines with equal weights should give same preds."""
    p1 = _make_pipeline(poisson_parquet)
    fitted = BlendingEnsemble(mode="fixed", weights=[1.0]).fit([p1])
    preds = fitted.predict(p1.test_data)
    orig = p1.fitted_model.predict(p1.test_data, "response")
    np.testing.assert_allclose(preds.to_numpy(), orig.to_numpy(), rtol=1e-6)


# ── Validation mode ────────────────────────────────────────────────────────────

def test_validation_blend_optimizes_weights(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    # Use training data as stand-in validation set
    fitted = BlendingEnsemble(mode="validation").fit([p1, p2], validation_data=p1.train_data)
    assert len(fitted.weights) == 2
    assert abs(sum(fitted.weights) - 1.0) < 1e-6


def test_validation_blend_weights_nonnegative(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    fitted = BlendingEnsemble(mode="validation").fit([p1, p2], validation_data=p1.train_data)
    assert all(w >= 0 for w in fitted.weights)


def test_validation_blend_requires_validation_data(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    with pytest.raises(ValueError, match="validation_data"):
        BlendingEnsemble(mode="validation").fit([p1, p1])


# ── Predict ────────────────────────────────────────────────────────────────────

def test_blend_predict_length_matches_data(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    fitted = BlendingEnsemble(mode="fixed", weights=[0.5, 0.5]).fit([p1, p2])
    preds = fitted.predict(p1.test_data)
    assert len(preds) == p1.test_data.n_rows


def test_blend_predictions_are_positive(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    fitted = BlendingEnsemble(mode="fixed", weights=[0.5, 0.5]).fit([p1, p2])
    preds = fitted.predict(p1.test_data)
    assert all(v > 0 for v in preds.to_list())
