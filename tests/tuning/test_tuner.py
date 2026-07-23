from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from ins_gbm.data.loader import load_model_data
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.preprocessing.pca import PCAReducer
from ins_gbm.preprocessing.steps import PreprocessingStep
from ins_gbm.tuning.tuner import HyperparameterTuner


class _RecordingModel:
    objective = "poisson"

    def __init__(self):
        self.fit_feature_names = []

    def default_search_space(self):
        return {}

    def fit(self, data, params=None):
        self.fit_feature_names.append(list(data.feature_names))

        class Fitted:
            def predict(self, validation_data, prediction_type="response"):
                return pl.Series([1.0] * validation_data.n_rows)

        return Fitted()


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


def test_tuner_selects_runtime_feature_subset(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    model = _RecordingModel()

    HyperparameterTuner(n_trials=1, cv_folds=2).tune(
        data,
        model,
        feature_names=["x3"],
    )

    assert model.fit_feature_names == [["x3"], ["x3"]]


def test_tuner_feature_subset_filters_explicit_encoder_schema(poisson_parquet):
    from ins_gbm.data.schema import FeatureSchema
    from ins_gbm.preprocessing.encoder import OneHotEncoder

    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    model = _RecordingModel()
    schema = FeatureSchema(
        numeric=["x1", "x3"],
        categorical=[],
        ordinal=[],
        passthrough=[],
    )

    HyperparameterTuner(n_trials=1, cv_folds=2).tune(
        data,
        model,
        encoder=OneHotEncoder(),
        schema=schema,
        feature_names=["x3"],
    )

    assert model.fit_feature_names == [["x3"], ["x3"]]


def test_tuner_passes_parallel_job_count_to_optuna(
    poisson_parquet,
    monkeypatch,
):
    import optuna

    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    observed_n_jobs = []
    original_optimize = optuna.study.Study.optimize

    def recording_optimize(study, objective, *args, **kwargs):
        observed_n_jobs.append(kwargs["n_jobs"])
        return original_optimize(study, objective, *args, **kwargs)

    monkeypatch.setattr(optuna.study.Study, "optimize", recording_optimize)
    _, history = HyperparameterTuner(
        n_trials=4,
        cv_folds=2,
        n_jobs=2,
    ).tune(data, _RecordingModel())

    assert observed_n_jobs == [2]
    assert len(history) == 4


@pytest.mark.parametrize("n_jobs", [0, -2, True, 1.5])
def test_tuner_rejects_invalid_n_jobs(poisson_parquet, n_jobs):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    with pytest.raises(ValueError, match="n_jobs"):
        HyperparameterTuner(
            n_trials=1,
            cv_folds=2,
            n_jobs=n_jobs,
        ).tune(data, _RecordingModel())


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


def test_tuner_poisson_deviance_uses_rate_and_combined_weight(
    poisson_parquet, monkeypatch
):
    import ins_gbm.tuning.tuner as tuner_module

    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    model_weight = pl.Series("model_weight", [2.0] * data.n_rows)
    data = replace(data, weight=model_weight).validate()
    expected_calls = []
    observed_calls = []

    class RecordingModel:
        objective = "poisson"

        def default_search_space(self):
            return {}

        def fit(self, train_data, params=None):
            class Fitted:
                def predict(self, validation_data, prediction_type="response"):
                    expected_calls.append((
                        validation_data.target.to_numpy()
                        / validation_data.exposure.to_numpy(),
                        np.ones(validation_data.n_rows),
                        validation_data.exposure.to_numpy()
                        * validation_data.weight.to_numpy(),
                    ))
                    return validation_data.exposure

            return Fitted()

    def recording_deviance(actual, predicted, weights=None):
        observed_calls.append((
            actual.to_numpy(),
            predicted.to_numpy(),
            weights.to_numpy(),
        ))
        return 0.0

    monkeypatch.setitem(
        tuner_module._METRIC_FN, "poisson_deviance", recording_deviance
    )
    HyperparameterTuner(
        n_trials=1, cv_folds=2, metric="poisson_deviance", seed=42
    ).tune(data, RecordingModel())

    assert len(observed_calls) == len(expected_calls) == 2
    for observed, expected in zip(observed_calls, expected_calls):
        for observed_values, expected_values in zip(observed, expected):
            np.testing.assert_allclose(observed_values, expected_values)


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


def test_tuner_applies_full_targeted_preprocessing_chain(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    tuner = HyperparameterTuner(n_trials=1, cv_folds=2, seed=42)

    _, history = tuner.tune(
        data,
        LightGBMModel(objective="poisson"),
        preprocessors=[
            PreprocessingStep(
                name="x1_pca",
                preprocessor=PCAReducer(n_components=1),
                feature_names=["x1"],
            ),
        ],
    )

    assert len(history) == 1
