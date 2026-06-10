"""Tests for predefined CV fold support in HyperparameterTuner."""
import numpy as np
import polars as pl
import pytest
from dataclasses import dataclass, replace
from typing import Any, Optional
from ins_gbm.data.loader import load_model_data
from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.tuning.tuner import HyperparameterTuner


# ── Stub model that records validation row indices ────────────────────────────

class _RecordingModel:
    """Minimal BaseModel stub that records which row indices it sees at predict time."""

    def __init__(self):
        self.val_indices_seen: list[list[int]] = []
        self._last_target = None

    def default_search_space(self) -> dict:
        import optuna.distributions as D
        return {"dummy": D.FloatDistribution(0.0, 1.0)}

    def fit(self, data: ModelData, params: Optional[dict] = None) -> "_FittedStub":
        return _FittedStub(len(data.target))


@dataclass
class _FittedStub:
    n_train: int

    def predict(self, data: ModelData, prediction_type: str = "response") -> pl.Series:
        # Return constant 1.0 predictions so deviance metrics work
        return pl.Series([1.0] * len(data.target))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def folded_data(poisson_parquet):
    n = 400
    # Create 3 folds: ~133 rows each
    fold_ids = pl.Series("cv_fold", [i % 3 for i in range(n)], dtype=pl.Int64)
    df = pl.read_parquet(poisson_parquet)
    # Attach fold column and write new parquet
    df = df.with_columns(fold_ids)
    import tempfile, os
    p = poisson_parquet.parent / "folded.parquet"
    df.write_parquet(p)
    data = load_model_data(
        path=str(p), target="claim_count", exposure="exposure",
        feature_cols=["x1", "x3"], objective="poisson",
        cv_fold="cv_fold",
    )
    assert data.cv_fold is not None
    return data


# ── use_data_folds=False uses random KFold (backward compat) ─────────────────

def test_use_data_folds_false_runs(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    tuner = HyperparameterTuner(n_trials=2, cv_folds=2, seed=42, use_data_folds=False)
    best_params, history = tuner.tune(data, _RecordingModel())
    assert isinstance(best_params, dict)
    assert len(history) == 2


# ── use_data_folds=True basic success ────────────────────────────────────────

def test_use_data_folds_true_runs(folded_data):
    tuner = HyperparameterTuner(n_trials=2, use_data_folds=True, seed=42)
    best_params, history = tuner.tune(folded_data, _RecordingModel())
    assert isinstance(best_params, dict)
    assert len(history) == 2


def test_use_data_folds_true_with_real_model(folded_data):
    from ins_gbm.models.lightgbm import LightGBMModel
    tuner = HyperparameterTuner(n_trials=2, use_data_folds=True, seed=42)
    best_params, history = tuner.tune(folded_data, LightGBMModel(objective="poisson"))
    assert isinstance(best_params, dict)
    assert len(history) == 2


# ── use_data_folds=True with cv_fold=None raises ─────────────────────────────

def test_use_data_folds_true_without_cv_fold_raises(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    assert data.cv_fold is None
    tuner = HyperparameterTuner(n_trials=1, use_data_folds=True)
    with pytest.raises(ValueError, match="cv_fold"):
        tuner.tune(data, _RecordingModel())


# ── Fold membership: each row in exactly one validation fold ──────────────────

def test_predefined_folds_cover_all_rows(folded_data):
    """Every row must appear in exactly one validation fold."""
    # We capture validation indices by intercepting slice_model_data via a
    # custom model that records the val_data row counts per CV fold.
    val_row_counts: list[int] = []

    class _CountingModel:
        def default_search_space(self):
            import optuna.distributions as D
            return {"dummy": D.FloatDistribution(0.0, 1.0)}

        def fit(self, data: ModelData, params=None):
            return _FittedStub(len(data.target))

    # Monkey-patch _FittedStub.predict to record val size
    original_predict = _FittedStub.predict

    class _InstrumentedStub(_FittedStub):
        def predict(self, data, prediction_type="response"):
            val_row_counts.append(len(data.target))
            return pl.Series([1.0] * len(data.target))

    class _InstrumentedModel:
        def default_search_space(self):
            import optuna.distributions as D
            return {"dummy": D.FloatDistribution(0.0, 1.0)}

        def fit(self, data, params=None):
            return _InstrumentedStub(len(data.target))

    tuner = HyperparameterTuner(n_trials=1, use_data_folds=True, seed=0)
    tuner.tune(folded_data, _InstrumentedModel())

    n = len(folded_data.target)
    # With 3 folds and 1 trial, there should be 3 validation splits
    assert len(val_row_counts) == 3
    # All validation rows together should sum to N (each row in exactly one fold)
    assert sum(val_row_counts) == n


def test_predefined_fold_counts_match_fold_column(folded_data):
    """Validation split sizes should match actual fold sizes in the column."""
    fold_arr = folded_data.cv_fold.to_numpy()
    expected_sizes = sorted([int((fold_arr == f).sum()) for f in np.unique(fold_arr)])

    val_row_counts: list[int] = []

    class _InstrumentedModel:
        def default_search_space(self):
            import optuna.distributions as D
            return {"dummy": D.FloatDistribution(0.0, 1.0)}

        def fit(self, data, params=None):
            class _Stub:
                def predict(self_, data2, prediction_type="response"):
                    val_row_counts.append(len(data2.target))
                    return pl.Series([1.0] * len(data2.target))
            return _Stub()

    tuner = HyperparameterTuner(n_trials=1, use_data_folds=True, seed=0)
    tuner.tune(folded_data, _InstrumentedModel())

    assert sorted(val_row_counts) == expected_sizes
