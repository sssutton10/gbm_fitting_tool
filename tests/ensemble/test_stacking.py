import polars as pl
import pytest
import numpy as np
from ins_gbm.data.loader import load_model_data
from ins_gbm.data.splitter import TrainTestSplit
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import FittedPipeline, ModelPipeline, ModelRecipe
from ins_gbm.ensemble.stacking import StackingEnsemble, FittedStackingEnsemble


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


# ── Basic structure ─────────────────────────────────────────────────────────────

def test_stacking_run_returns_fitted(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    stacker = StackingEnsemble(cv_folds=2, seed=42)
    fitted = stacker.fit([p1, p2])
    assert isinstance(fitted, FittedStackingEnsemble)


def test_stacking_has_meta_learner(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    fitted = StackingEnsemble(cv_folds=2, seed=42).fit([p1, p2])
    assert fitted.meta_learner is not None


# ── Predict ─────────────────────────────────────────────────────────────────────

def test_stacking_predict_length(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    fitted = StackingEnsemble(cv_folds=2, seed=42).fit([p1, p2])
    preds = fitted.predict(p1.test_data)
    assert len(preds) == p1.test_data.n_rows


def test_stacking_predictions_are_positive(poisson_parquet):
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    fitted = StackingEnsemble(cv_folds=2, seed=42).fit([p1, p2])
    preds = fitted.predict(p1.test_data)
    assert all(v > 0 for v in preds.to_list())


# ── OOF predictions ─────────────────────────────────────────────────────────────

def test_stacking_oof_shape(poisson_parquet):
    """OOF matrix should have n_train rows and n_base_models columns."""
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    fitted = StackingEnsemble(cv_folds=2, seed=42).fit([p1, p2])
    assert fitted.oof_predictions.shape == (p1.train_data.n_rows, 2)


# ── Custom meta-learner ─────────────────────────────────────────────────────────

def test_stacking_custom_meta_learner(poisson_parquet):
    from sklearn.linear_model import Ridge
    p1 = _make_pipeline(poisson_parquet)
    p2 = _make_pipeline(poisson_parquet)
    fitted = StackingEnsemble(cv_folds=2, seed=42, meta_learner=Ridge()).fit([p1, p2])
    preds = fitted.predict(p1.test_data)
    assert len(preds) == p1.test_data.n_rows
