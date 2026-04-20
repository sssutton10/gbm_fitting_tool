import polars as pl
import pytest
from ins_gbm.data.loader import load_model_data
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.tuning.tuner import HyperparameterTuner


# ── Basic return types ──────────────────────────────────────────────────────────

def test_tuner_returns_dict_and_dataframe(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    tuner = HyperparameterTuner(n_trials=3, cv_folds=2, seed=42)
    best_params, history = tuner.tune(data, LightGBMModel(objective="poisson"))
    assert isinstance(best_params, dict)
    assert isinstance(history, pl.DataFrame)


def test_tuner_history_has_required_columns(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    tuner = HyperparameterTuner(n_trials=3, cv_folds=2, seed=42)
    _, history = tuner.tune(data, LightGBMModel(objective="poisson"))
    assert "trial" in history.columns
    assert "value" in history.columns


def test_tuner_history_row_count_equals_n_trials(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    tuner = HyperparameterTuner(n_trials=4, cv_folds=2, seed=42)
    _, history = tuner.tune(data, LightGBMModel(objective="poisson"))
    assert len(history) == 4


# ── Best params ─────────────────────────────────────────────────────────────────

def test_tuner_best_params_keys_are_subset_of_search_space(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    model = LightGBMModel(objective="poisson")
    tuner = HyperparameterTuner(n_trials=2, cv_folds=2, seed=42)
    best_params, _ = tuner.tune(data, model)
    search_space_keys = set(model.default_search_space().keys())
    assert set(best_params.keys()).issubset(search_space_keys)


def test_tuner_best_params_nonempty(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    tuner = HyperparameterTuner(n_trials=2, cv_folds=2, seed=42)
    best_params, _ = tuner.tune(data, LightGBMModel(objective="poisson"))
    assert len(best_params) > 0


# ── Metric values ───────────────────────────────────────────────────────────────

def test_tuner_values_are_nonnegative(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    tuner = HyperparameterTuner(n_trials=3, cv_folds=2, metric="poisson_deviance", seed=42)
    _, history = tuner.tune(data, LightGBMModel(objective="poisson"))
    assert all(v >= 0 for v in history["value"].to_list())


def test_tuner_invalid_metric_raises(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    tuner = HyperparameterTuner(n_trials=2, cv_folds=2, metric="bad_metric", seed=42)
    with pytest.raises(ValueError, match="Unknown metric"):
        tuner.tune(data, LightGBMModel(objective="poisson"))


# ── With encoder ────────────────────────────────────────────────────────────────

def test_tuner_runs_with_encoder(poisson_parquet):
    """Encoder should be fit per fold (not on full data)."""
    from ins_gbm.preprocessing.encoder import OneHotEncoder
    from ins_gbm.data.schema import FeatureSchema

    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    # Purely numeric data — encoder should be a no-op but must not error
    schema = FeatureSchema(numeric=["x1", "x3"], categorical=[], ordinal=[], passthrough=[])
    encoder = OneHotEncoder()
    tuner = HyperparameterTuner(n_trials=2, cv_folds=2, seed=42)
    best_params, history = tuner.tune(data, LightGBMModel(objective="poisson"),
                                      encoder=encoder, schema=schema)
    assert best_params is not None
    assert len(history) == 2
