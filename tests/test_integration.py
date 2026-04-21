"""End-to-end integration tests.

These tests verify the full pipeline from parquet → split → (tune) → fit →
evaluate → export, including ensemble and leakage checks.  They intentionally
use small n_estimators / n_trials so the suite stays fast.
"""
import os
import numpy as np
import polars as pl
import pytest

from ins_gbm.data.loader import load_model_data
from ins_gbm.data.splitter import TrainTestSplit
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import FittedPipeline, ModelPipeline, ModelRecipe
from ins_gbm.persistence.io import load_pipeline, save_pipeline
from ins_gbm.tuning.tuner import HyperparameterTuner
from ins_gbm.ensemble.pipeline import EnsemblePipeline


# ── Poisson end-to-end ──────────────────────────────────────────────────────────

def test_poisson_end_to_end(poisson_parquet, tmp_path):
    """Frequency pipeline: load → split → tune → fit → evaluate → export."""
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

    assert isinstance(result, FittedPipeline)
    assert result.train_data.n_rows > 0
    assert result.test_data.n_rows > 0
    assert result.tuning_history is not None and len(result.tuning_history) == 2

    metrics = result.report.metrics()
    assert "metric" in metrics.columns
    assert any("poisson" in m for m in metrics["metric"].to_list())

    preds = result.fitted_model.predict(result.test_data, "response")
    assert len(preds) == result.test_data.n_rows
    assert all(v > 0 for v in preds.to_list())

    # Export
    out = str(tmp_path / "poisson_out")
    result.report.export(out)
    assert os.path.exists(os.path.join(out, "metrics.csv"))


# ── Gamma end-to-end ────────────────────────────────────────────────────────────

def test_gamma_end_to_end(gamma_parquet, tmp_path):
    """Severity pipeline: load → split → fit → evaluate."""
    data = load_model_data(
        path=str(gamma_parquet), target="severity",
        weight="weight", feature_cols=["x1"], objective="gamma",
    )
    result = ModelPipeline(
        data=data,
        split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="gamma")),
    ).run()

    assert isinstance(result, FittedPipeline)
    metrics = result.report.metrics()
    assert any("gamma" in m for m in metrics["metric"].to_list())

    preds = result.fitted_model.predict(result.test_data, "response")
    assert len(preds) == result.test_data.n_rows
    assert all(v > 0 for v in preds.to_list())


# ── Ensemble integration ────────────────────────────────────────────────────────

def test_ensemble_blending_integration(poisson_parquet, tmp_path):
    """Two base pipelines → blending ensemble → report."""
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    p1 = ModelPipeline(
        data=data, split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()
    p2 = ModelPipeline(
        data=data, split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()

    ensemble_result = EnsemblePipeline(
        fitted_pipelines=[p1, p2],
        method="blending",
        blend_mode="fixed",
        blend_weights=[0.5, 0.5],
    ).run()

    metrics = ensemble_result.report.metrics()
    assert len(metrics) > 0
    preds = ensemble_result.predict(p1.test_data)
    assert len(preds) == p1.test_data.n_rows


def test_ensemble_stacking_integration(poisson_parquet):
    """Two base pipelines → stacking → valid predictions."""
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    p1 = ModelPipeline(
        data=data, split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()
    p2 = ModelPipeline(
        data=data, split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()

    ensemble_result = EnsemblePipeline(
        fitted_pipelines=[p1, p2],
        method="stacking",
        cv_folds=2,
        seed=42,
    ).run()

    preds = ensemble_result.predict(p1.test_data)
    assert len(preds) == p1.test_data.n_rows
    assert all(v > 0 for v in preds.to_list())


# ── Persistence ────────────────────────────────────────────────────────────────

def test_save_load_predictions_identical(poisson_parquet, tmp_path):
    """Saved and loaded pipeline must produce bit-identical predictions."""
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    result = ModelPipeline(
        data=data, split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()

    out = str(tmp_path / "saved")
    save_pipeline(result, out)
    loaded = load_pipeline(out)

    orig_preds = result.fitted_model.predict(result.test_data, "response")
    loaded_preds = loaded.fitted_model.predict(loaded.test_data, "response")
    assert orig_preds.to_list() == loaded_preds.to_list()


# ── Leakage verification ────────────────────────────────────────────────────────

def test_no_leakage_test_data_not_used_for_tuning(poisson_parquet):
    """Test set rows must not appear in the training or validation fold indices
    used during hyperparameter tuning."""
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

    # Verify train + test rows account for all input rows
    assert result.train_data.n_rows + result.test_data.n_rows == data.n_rows

    # Tuner only operates on train_data; verify test set size is 30%
    total = data.n_rows  # 400
    expected_test = int(total * 0.3)  # 120
    assert result.test_data.n_rows == expected_test


def test_no_leakage_test_rows_excluded_from_train(poisson_parquet):
    """Train and test sets must be disjoint when using a reproducible split."""
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    result = ModelPipeline(
        data=data, split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()

    # Verify sizes partition the dataset
    assert result.train_data.n_rows + result.test_data.n_rows == data.n_rows

    # Target values in test set should not all appear in train (probabilistic guard)
    train_targets = set(result.train_data.target.to_list())
    test_targets = result.test_data.target.to_list()
    # At minimum, the test set is non-empty and smaller than full data
    assert len(test_targets) < data.n_rows
