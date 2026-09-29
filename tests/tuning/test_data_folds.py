"""Tests for predefined CV fold support in HyperparameterTuner."""

from dataclasses import dataclass

import numpy as np
import polars as pl
import pytest

from ins_gbm.data.loader import load_model_data
from ins_gbm.data.model_data import ModelData
from ins_gbm.tuning.tuner import HyperparameterTuner

# ── Stub model that records validation row indices ────────────────────────────


class _RecordingModel:
    """Minimal BaseModel stub that records which row indices it sees at predict time."""

    def __init__(self):
        """Init."""
        self.val_indices_seen: list[list[int]] = []
        self._last_target = None

    def default_search_space(self) -> dict:
        """Default search space."""
        import optuna.distributions as D

        return {"dummy": D.FloatDistribution(0.0, 1.0)}

    def fit(self, data: ModelData, params: dict | None = None) -> "_FittedStub":
        """Fit.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
            params (Optional[dict]): Optional model or estimator parameter mapping.
        """
        return _FittedStub(len(data.target))


@dataclass
class _FittedStub:
    """Fittedstub.

    Args:
        n_train (int): The n train.
    """

    n_train: int

    def predict(self, data: ModelData, prediction_type: str = "response") -> pl.Series:
        # Return constant 1.0 predictions so deviance metrics work
        """Predict.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
            prediction_type (str): Prediction scale: "response", "rate", or "link"; "rate" is
                unavailable for Gamma. Defaults to 'response'.
        """
        return pl.Series([1.0] * len(data.target))


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def folded_data(poisson_parquet):
    """Folded data.

    Args:
        poisson_parquet (object): The poisson parquet.
    """
    n = 400
    # Create 3 folds: ~133 rows each
    fold_ids = pl.Series("cv_fold", [i % 3 for i in range(n)], dtype=pl.Int64)
    df = pl.read_parquet(poisson_parquet)
    # Attach fold column and write new parquet
    df = df.with_columns(fold_ids)
    p = poisson_parquet.parent / "folded.parquet"
    df.write_parquet(p)
    data = load_model_data(
        path=str(p),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
        cv_fold="cv_fold",
    )
    assert data.cv_fold is not None
    return data


# ── use_data_folds=False uses random KFold (backward compat) ─────────────────


def test_use_data_folds_false_runs(poisson_parquet):
    """Verify use data folds false runs.

    Args:
        poisson_parquet (object): The poisson parquet.
    """
    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
    )
    tuner = HyperparameterTuner(n_trials=2, cv_folds=2, seed=42, use_data_folds=False)
    best_params, history = tuner.tune(data, _RecordingModel())
    assert isinstance(best_params, dict)
    assert len(history) == 2


# ── use_data_folds=True basic success ────────────────────────────────────────


def test_use_data_folds_true_runs(folded_data):
    """Verify use data folds true runs.

    Args:
        folded_data (object): The folded data.
    """
    tuner = HyperparameterTuner(n_trials=2, use_data_folds=True, seed=42)
    best_params, history = tuner.tune(folded_data, _RecordingModel())
    assert isinstance(best_params, dict)
    assert len(history) == 2


def test_use_data_folds_true_with_real_model(folded_data):
    """Verify use data folds true with real model.

    Args:
        folded_data (object): The folded data.
    """
    from ins_gbm.models.lightgbm import LightGBMModel

    tuner = HyperparameterTuner(n_trials=2, use_data_folds=True, seed=42)
    best_params, history = tuner.tune(folded_data, LightGBMModel(objective="poisson"))
    assert isinstance(best_params, dict)
    assert len(history) == 2


# ── use_data_folds=True with cv_fold=None raises ─────────────────────────────


def test_use_data_folds_true_without_cv_fold_raises(poisson_parquet):
    """Verify use data folds true without cv fold raises.

    Args:
        poisson_parquet (object): The poisson parquet.
    """
    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
    )
    assert data.cv_fold is None
    tuner = HyperparameterTuner(n_trials=1, use_data_folds=True)
    with pytest.raises(ValueError, match="cv_fold"):
        tuner.tune(data, _RecordingModel())


# ── Fold membership: each row in exactly one validation fold ──────────────────


def test_predefined_folds_cover_all_rows(folded_data):
    """Every row must appear in exactly one validation fold.

    Args:
        folded_data (object): The folded data.
    """
    # We capture validation indices by intercepting slice_model_data via a
    # custom model that records the val_data row counts per CV fold.
    val_row_counts: list[int] = []

    class _CountingModel:
        """Countingmodel."""

        def default_search_space(self):
            """Default search space."""
            import optuna.distributions as D

            return {"dummy": D.FloatDistribution(0.0, 1.0)}

        def fit(self, data: ModelData, params=None):
            """Fit.

            Args:
                data (ModelData): Model data to fit, transform, predict, or evaluate.
                params (object): Optional model or estimator parameter mapping.
            """
            return _FittedStub(len(data.target))

    class _InstrumentedStub(_FittedStub):
        """Instrumentedstub."""

        def predict(self, data, prediction_type="response"):
            """Predict.

            Args:
                data (object): Model data to fit, transform, predict, or evaluate.
                prediction_type (object): Prediction scale: "response", "rate", or "link"; "rate" is
                    unavailable for Gamma. Defaults to 'response'.
            """
            val_row_counts.append(len(data.target))
            return pl.Series([1.0] * len(data.target))

    class _InstrumentedModel:
        """Instrumentedmodel."""

        def default_search_space(self):
            """Default search space."""
            import optuna.distributions as D

            return {"dummy": D.FloatDistribution(0.0, 1.0)}

        def fit(self, data, params=None):
            """Fit.

            Args:
                data (object): Model data to fit, transform, predict, or evaluate.
                params (object): Optional model or estimator parameter mapping.
            """
            return _InstrumentedStub(len(data.target))

    tuner = HyperparameterTuner(n_trials=1, use_data_folds=True, seed=0)
    tuner.tune(folded_data, _InstrumentedModel())

    n = len(folded_data.target)
    # With 3 folds and 1 trial, there should be 3 validation splits
    assert len(val_row_counts) == 3
    # All validation rows together should sum to N (each row in exactly one fold)
    assert sum(val_row_counts) == n


def test_predefined_fold_counts_match_fold_column(folded_data):
    """Validation split sizes should match actual fold sizes in the column.

    Args:
        folded_data (object): The folded data.
    """
    fold_arr = folded_data.cv_fold.to_numpy()
    expected_sizes = sorted([int((fold_arr == f).sum()) for f in np.unique(fold_arr)])

    val_row_counts: list[int] = []

    class _InstrumentedModel:
        """Instrumentedmodel."""

        def default_search_space(self):
            """Default search space."""
            import optuna.distributions as D

            return {"dummy": D.FloatDistribution(0.0, 1.0)}

        def fit(self, data, params=None):
            """Fit.

            Args:
                data (object): Model data to fit, transform, predict, or evaluate.
                params (object): Optional model or estimator parameter mapping.
            """

            class _Stub:
                """Stub."""

                def predict(self_, data2, prediction_type="response"):
                    """Predict.

                    Args:
                        self_ (object): The self.
                        data2 (object): The data2.
                        prediction_type (object): Prediction scale: "response", "rate", or "link"; "rate" is
                            unavailable for Gamma. Defaults to 'response'.
                    """
                    val_row_counts.append(len(data2.target))
                    return pl.Series([1.0] * len(data2.target))

            return _Stub()

    tuner = HyperparameterTuner(n_trials=1, use_data_folds=True, seed=0)
    tuner.tune(folded_data, _InstrumentedModel())

    assert sorted(val_row_counts) == expected_sizes
