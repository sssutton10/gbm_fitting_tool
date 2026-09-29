import os
import subprocess
import sys
import threading
import time
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from ins_gbm.data.loader import load_model_data
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.preprocessing.pca import PCAReducer
from ins_gbm.preprocessing.steps import PreprocessingStep
from ins_gbm.tuning.tuner import HyperparameterTuner, _create_journal_storage


class _RecordingModel:
    """Recordingmodel."""

    objective = "poisson"

    def __init__(self):
        """Init."""
        self.fit_feature_names = []

    def default_search_space(self):
        """Default search space."""
        return {}

    def fit(self, data, params=None):
        """Fit.

        Args:
            data (object): Model data to fit, transform, predict, or evaluate.
            params (object): Optional model or estimator parameter mapping.
        """
        self.fit_feature_names.append(list(data.feature_names))

        class Fitted:
            """Configure Fitted."""

            def predict(self, validation_data, prediction_type="response"):
                """Predict.

                Args:
                    validation_data (object): Holdout data used to learn blend weights.
                    prediction_type (object): Prediction scale: "response", "rate", or "link"; "rate" is
                        unavailable for Gamma. Defaults to 'response'.
                """
                return pl.Series([1.0] * validation_data.n_rows)

        return Fitted()


class _PidModel:
    """Pidmodel."""

    objective = "poisson"

    def default_search_space(self):
        """Default search space."""
        return {}

    def fit(self, data, params=None):
        """Fit.

        Args:
            data (object): Model data to fit, transform, predict, or evaluate.
            params (object): Optional model or estimator parameter mapping.
        """
        worker_pid = float(os.getpid())

        class Fitted:
            """Configure Fitted."""

            def predict(self, validation_data, prediction_type="response"):
                """Predict.

                Args:
                    validation_data (object): Holdout data used to learn blend weights.
                    prediction_type (object): Prediction scale: "response", "rate", or "link"; "rate" is
                        unavailable for Gamma. Defaults to 'response'.
                """
                return pl.Series([worker_pid] * validation_data.n_rows)

        return Fitted()


class _FailingModel:
    """Failingmodel."""

    objective = "poisson"

    def default_search_space(self):
        """Default search space."""
        return {}

    def fit(self, data, params=None):
        """Fit.

        Args:
            data (object): Model data to fit, transform, predict, or evaluate.
            params (object): Optional model or estimator parameter mapping.
        """
        raise RuntimeError("worker boom")


class _SlowPidModel(_PidModel):
    """Slowpidmodel."""

    def fit(self, data, params=None):
        """Fit.

        Args:
            data (object): Model data to fit, transform, predict, or evaluate.
            params (object): Optional model or estimator parameter mapping.
        """
        time.sleep(0.05)
        return super().fit(data, params=params)


# ── Basic return types ──────────────────────────────────────────────────────────


def test_tuner_returns_dict_and_dataframe(poisson_parquet):
    """Verify tuner returns dict and dataframe.

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
    tuner = HyperparameterTuner(n_trials=3, cv_folds=2, seed=42)
    best_params, history = tuner.tune(data, LightGBMModel(objective="poisson"))
    assert isinstance(best_params, dict)
    assert isinstance(history, pl.DataFrame)


def test_tuner_history_has_required_columns(poisson_parquet):
    """Verify tuner history has required columns.

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
    tuner = HyperparameterTuner(n_trials=3, cv_folds=2, seed=42)
    _, history = tuner.tune(data, LightGBMModel(objective="poisson"))
    assert "trial" in history.columns
    assert "value" in history.columns


def test_tuner_history_row_count_equals_n_trials(poisson_parquet):
    """Verify tuner history row count equals n trials.

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
    tuner = HyperparameterTuner(n_trials=4, cv_folds=2, seed=42)
    _, history = tuner.tune(data, LightGBMModel(objective="poisson"))
    assert len(history) == 4


def test_tuner_selects_runtime_feature_subset(poisson_parquet):
    """Verify tuner selects runtime feature subset.

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
    model = _RecordingModel()

    HyperparameterTuner(n_trials=1, cv_folds=2).tune(
        data,
        model,
        feature_names=["x3"],
    )

    assert model.fit_feature_names == [["x3"], ["x3"]]


def test_tuner_feature_subset_filters_explicit_encoder_schema(poisson_parquet):
    """Verify tuner feature subset filters explicit encoder schema.

    Args:
        poisson_parquet (object): The poisson parquet.
    """
    from ins_gbm.data.schema import FeatureSchema
    from ins_gbm.preprocessing.encoder import OneHotEncoder

    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
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
    """Verify tuner passes parallel job count to optuna.

    Args:
        poisson_parquet (object): The poisson parquet.
        monkeypatch (object): The monkeypatch.
    """
    import optuna

    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
    )
    observed_n_jobs = []
    original_optimize = optuna.study.Study.optimize

    def recording_optimize(study, objective, *args, **kwargs):
        """Recording optimize.

        Args:
            study (object): Optuna study containing the trial results.
            objective (object): Model objective: "poisson" or "gamma".
            args (object): The args.
            kwargs (object): Additional keyword arguments forwarded to the pipeline run.
        """
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


def test_tuner_process_backend_uses_distinct_worker_processes(poisson_parquet):
    """Verify tuner process backend uses distinct worker processes.

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
    _, history = HyperparameterTuner(
        n_trials=4,
        cv_folds=2,
        n_jobs=2,
        backend="process",
        show_progress_bar=False,
    ).tune(data, _PidModel())

    assert len(history) == 4
    # The objective value encodes the fitting PID. Each worker has a non-zero
    # quota, so two distinct values demonstrate that both subprocesses ran.
    assert history["value"].n_unique() == 2


def test_tuner_process_backend_resolves_all_available_cpus(
    poisson_parquet,
    monkeypatch,
):
    """Verify tuner process backend resolves all available cpus.

    Args:
        poisson_parquet (object): The poisson parquet.
        monkeypatch (object): The monkeypatch.
    """
    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
    )
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    _, history = HyperparameterTuner(
        n_trials=2,
        cv_folds=2,
        n_jobs=-1,
        backend="process",
        show_progress_bar=False,
    ).tune(data, _PidModel())

    assert len(history) == 2
    assert history["value"].n_unique() == 2


def test_journal_storage_uses_open_lock_on_windows(tmp_path, monkeypatch):
    """Verify journal storage uses open lock on windows.

    Args:
        tmp_path (object): The tmp path.
        monkeypatch (object): The monkeypatch.
    """
    from optuna.storages.journal import JournalFileOpenLock

    import ins_gbm.tuning.tuner as tuner_module

    monkeypatch.setattr(tuner_module.platform, "system", lambda: "Windows")
    storage = _create_journal_storage(str(tmp_path / "windows.journal"))

    assert isinstance(storage._backend._lock, JournalFileOpenLock)


def test_journal_storage_retains_default_lock_off_windows(
    tmp_path,
    monkeypatch,
):
    """Verify journal storage retains default lock off windows.

    Args:
        tmp_path (object): The tmp path.
        monkeypatch (object): The monkeypatch.
    """
    from optuna.storages.journal import JournalFileSymlinkLock

    import ins_gbm.tuning.tuner as tuner_module

    monkeypatch.setattr(tuner_module.platform, "system", lambda: "Linux")
    storage = _create_journal_storage(str(tmp_path / "linux.journal"))

    assert isinstance(storage._backend._lock, JournalFileSymlinkLock)


def test_tuner_process_backend_retains_explicit_journal(
    poisson_parquet,
    tmp_path,
):
    """Verify tuner process backend retains explicit journal.

    Args:
        poisson_parquet (object): The poisson parquet.
        tmp_path (object): The tmp path.
    """
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
    )
    journal_path = tmp_path / "tuning.journal"
    HyperparameterTuner(
        n_trials=2,
        cv_folds=2,
        n_jobs=2,
        backend="process",
        journal_path=journal_path,
        show_progress_bar=False,
    ).tune(data, _PidModel())

    storage = JournalStorage(JournalFileBackend(file_path=str(journal_path)))
    summaries = optuna.get_all_study_summaries(storage=storage)
    study = optuna.load_study(
        study_name=summaries[0].study_name,
        storage=storage,
    )
    assert journal_path.exists()
    assert len(summaries) == 1
    assert len(study.trials) == 2


def test_tuner_process_progress_runs_in_parent(poisson_parquet):
    """Verify tuner process progress runs in parent.

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
    parent_pid = os.getpid()
    callback_pids = []
    HyperparameterTuner(
        n_trials=2,
        cv_folds=2,
        n_jobs=2,
        backend="process",
        show_progress_bar=False,
    ).tune(
        data,
        _PidModel(),
        progress=lambda event: callback_pids.append(os.getpid()),
    )

    assert callback_pids == [parent_pid, parent_pid]


def test_tuner_process_worker_error_is_reported(poisson_parquet):
    """Verify tuner process worker error is reported.

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
    with pytest.raises(RuntimeError, match="worker boom"):
        HyperparameterTuner(
            n_trials=2,
            cv_folds=2,
            n_jobs=2,
            backend="process",
            show_progress_bar=False,
        ).tune(data, _FailingModel())


def test_tuner_process_cancellation_stops_workers(poisson_parquet):
    """Verify tuner process cancellation stops workers.

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
    stop_event = threading.Event()

    def stop_after_first_trial(event):
        """Stop after first trial.

        Args:
            event (object): The event.
        """
        stop_event.set()

    from ins_gbm.progress import PipelineCancelled

    with pytest.raises(PipelineCancelled):
        HyperparameterTuner(
            n_trials=6,
            cv_folds=2,
            n_jobs=2,
            backend="process",
            show_progress_bar=False,
        ).tune(
            data,
            _SlowPidModel(),
            progress=stop_after_first_trial,
            should_stop=stop_event.is_set,
        )


def test_tuner_process_backend_runs_from_python_c(poisson_parquet):
    # `python -c` has no importable user __main__, matching the multiprocessing
    # constraint that normally makes notebook-defined worker functions fail.
    """Verify tuner process backend runs from python c.

    Args:
        poisson_parquet (object): The poisson parquet.
    """
    code = """
import polars as pl
import sys
from ins_gbm.data.loader import load_model_data
from ins_gbm.tuning.tuner import HyperparameterTuner

class NotebookModel:
    objective = "poisson"
    def default_search_space(self):
        return {}
    def fit(self, data, params=None):
        class Fitted:
            def predict(self, validation_data, prediction_type="response"):
                return pl.Series([1.0] * validation_data.n_rows)
        return Fitted()

data = load_model_data(
    path=sys.argv[1],
    target="claim_count",
    exposure="exposure",
    feature_cols=["x1", "x3"],
    objective="poisson",
)
_, history = HyperparameterTuner(
    n_trials=1,
    cv_folds=2,
    n_jobs=1,
    backend="process",
    show_progress_bar=False,
).tune(data, NotebookModel())
assert len(history) == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(poisson_parquet)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("n_jobs", [0, -2, True, 1.5])
def test_tuner_rejects_invalid_n_jobs(poisson_parquet, n_jobs):
    """Verify tuner rejects invalid n jobs.

    Args:
        poisson_parquet (object): The poisson parquet.
        n_jobs (object): Number of concurrent tuning workers.
    """
    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
    )
    with pytest.raises(ValueError, match="n_jobs"):
        HyperparameterTuner(
            n_trials=1,
            cv_folds=2,
            n_jobs=n_jobs,
        ).tune(data, _RecordingModel())


def test_tuner_rejects_invalid_backend(poisson_parquet):
    """Verify tuner rejects invalid backend.

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
    with pytest.raises(ValueError, match="backend"):
        HyperparameterTuner(
            n_trials=1,
            cv_folds=2,
            backend="invalid",
        ).tune(data, _RecordingModel())


def test_tuner_rejects_journal_path_for_thread_backend(
    poisson_parquet,
    tmp_path,
):
    """Verify tuner rejects journal path for thread backend.

    Args:
        poisson_parquet (object): The poisson parquet.
        tmp_path (object): The tmp path.
    """
    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
    )
    with pytest.raises(ValueError, match="journal_path"):
        HyperparameterTuner(
            n_trials=1,
            cv_folds=2,
            journal_path=tmp_path / "journal.log",
        ).tune(data, _RecordingModel())


# ── Best params ─────────────────────────────────────────────────────────────────


def test_tuner_best_params_keys_are_subset_of_search_space(poisson_parquet):
    """Verify tuner best params keys are subset of search space.

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
    model = LightGBMModel(objective="poisson")
    tuner = HyperparameterTuner(n_trials=2, cv_folds=2, seed=42)
    best_params, _ = tuner.tune(data, model)
    search_space_keys = set(model.default_search_space().keys())
    assert set(best_params.keys()).issubset(search_space_keys)


def test_tuner_best_params_nonempty(poisson_parquet):
    """Verify tuner best params nonempty.

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
    tuner = HyperparameterTuner(n_trials=2, cv_folds=2, seed=42)
    best_params, _ = tuner.tune(data, LightGBMModel(objective="poisson"))
    assert len(best_params) > 0


# ── Metric values ───────────────────────────────────────────────────────────────


def test_tuner_values_are_nonnegative(poisson_parquet):
    """Verify tuner values are nonnegative.

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
    tuner = HyperparameterTuner(
        n_trials=3, cv_folds=2, metric="poisson_deviance", seed=42
    )
    _, history = tuner.tune(data, LightGBMModel(objective="poisson"))
    assert all(v >= 0 for v in history["value"].to_list())


def test_tuner_poisson_deviance_uses_rate_and_combined_weight(
    poisson_parquet, monkeypatch
):
    """Verify tuner poisson deviance uses rate and combined weight.

    Args:
        poisson_parquet (object): The poisson parquet.
        monkeypatch (object): The monkeypatch.
    """
    import ins_gbm.tuning.tuner as tuner_module

    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
    )
    model_weight = pl.Series("model_weight", [2.0] * data.n_rows)
    data = replace(data, weight=model_weight).validate()
    expected_calls = []
    observed_calls = []

    class RecordingModel:
        """Configure RecordingModel."""

        objective = "poisson"

        def default_search_space(self):
            """Default search space."""
            return {}

        def fit(self, train_data, params=None):
            """Fit.

            Args:
                train_data (object): The train data.
                params (object): Optional model or estimator parameter mapping.
            """

            class Fitted:
                """Configure Fitted."""

                def predict(self, validation_data, prediction_type="response"):
                    """Predict.

                    Args:
                        validation_data (object): Holdout data used to learn blend weights.
                        prediction_type (object): Prediction scale: "response", "rate", or "link"; "rate" is
                            unavailable for Gamma. Defaults to 'response'.
                    """
                    expected_calls.append(
                        (
                            validation_data.target.to_numpy()
                            / validation_data.exposure.to_numpy(),
                            np.ones(validation_data.n_rows),
                            validation_data.exposure.to_numpy()
                            * validation_data.weight.to_numpy(),
                        )
                    )
                    return validation_data.exposure

            return Fitted()

    def recording_deviance(actual, predicted, weights=None):
        """Recording deviance.

        Args:
            actual (object): Observed outcomes aligned with predictions.
            predicted (object): Predicted outcomes aligned with actual outcomes.
            weights (object): Nonnegative blending or observation weights. Optional.
        """
        observed_calls.append(
            (
                actual.to_numpy(),
                predicted.to_numpy(),
                weights.to_numpy(),
            )
        )
        return 0.0

    monkeypatch.setitem(tuner_module._METRIC_FN, "poisson_deviance", recording_deviance)
    HyperparameterTuner(
        n_trials=1, cv_folds=2, metric="poisson_deviance", seed=42
    ).tune(data, RecordingModel())

    assert len(observed_calls) == len(expected_calls) == 2
    for observed, expected in zip(observed_calls, expected_calls):
        for observed_values, expected_values in zip(observed, expected):
            np.testing.assert_allclose(observed_values, expected_values)


def test_tuner_invalid_metric_raises(poisson_parquet):
    """Verify tuner invalid metric raises.

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
    tuner = HyperparameterTuner(n_trials=2, cv_folds=2, metric="bad_metric", seed=42)
    with pytest.raises(ValueError, match="Unknown metric"):
        tuner.tune(data, LightGBMModel(objective="poisson"))


# ── With encoder ────────────────────────────────────────────────────────────────


def test_tuner_runs_with_encoder(poisson_parquet):
    """Encoder should be fit per fold (not on full data).

    Args:
        poisson_parquet (object): The poisson parquet.
    """
    from ins_gbm.data.schema import FeatureSchema
    from ins_gbm.preprocessing.encoder import OneHotEncoder

    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
    )
    # Purely numeric data — encoder should be a no-op but must not error
    schema = FeatureSchema(
        numeric=["x1", "x3"], categorical=[], ordinal=[], passthrough=[]
    )
    encoder = OneHotEncoder()
    tuner = HyperparameterTuner(n_trials=2, cv_folds=2, seed=42)
    best_params, history = tuner.tune(
        data, LightGBMModel(objective="poisson"), encoder=encoder, schema=schema
    )
    assert best_params is not None
    assert len(history) == 2


def test_tuner_applies_full_targeted_preprocessing_chain(poisson_parquet):
    """Verify tuner applies full targeted preprocessing chain.

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
