"""Real-process OOF coverage, alongside deterministic scheduling checks."""

import json
import logging
import os
import time
from dataclasses import dataclass, replace

import numpy as np
import polars as pl
import pytest

from ins_gbm.data.folds import CVConfig
from ins_gbm.data.model_data import ModelData
from ins_gbm.ensemble._oof import (
    _FoldRecipe, _limited_params, _parallel_folds, _prepare_recipe,
    _worker_settings,
)
from ins_gbm.ensemble.blending import BlendingEnsemble
from ins_gbm.ensemble.pipeline import EnsemblePipeline
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import ModelRecipe
from ins_gbm.preprocessing.encoder import OneHotEncoder
from ins_gbm.preprocessing.pls import PLSReducer
from ins_gbm.preprocessing.steps import PreprocessingStep
from ins_gbm.selection import ImportanceSelectionStage, StagedImportanceSelector
from ins_gbm.tuning.tuner import HyperparameterTuner


class CapturingBlender(BlendingEnsemble):
    def _learn_weights(self, preds, data, pipelines):
        self.oof = preds.copy()
        return super()._learn_weights(preds, data, pipelines)


@pytest.fixture
def blend_data():
    rng = np.random.default_rng(11)
    n = 120
    x = rng.normal(size=n)
    return ModelData(
        features=pl.DataFrame({"x": x, "z": rng.normal(size=n), "cat": ["a", "b", "c"] * 40}),
        target=pl.Series(rng.poisson(np.exp(x / 3)).astype(float)),
        weight=pl.Series(rng.uniform(.5, 2, n)),
        feature_names=["x", "z", "cat"], objective="poisson",
        cv_fold=pl.Series([10, 30, 70] * 40),
    )


def make_pipelines(data, **kwargs):
    return [
        ModelRecipe(
            model=LightGBMModel(), encoder=OneHotEncoder(),
            params={"n_estimators": 8, "num_leaves": leaves, "num_threads": 2, "seed": 7},
            **kwargs,
        ).fit(data)
        for leaves in (3, 5)
    ]


@pytest.mark.parametrize("metric,objective,exposure,folds", [
    ("mse", "poisson", False, "random"),
    ("poisson_deviance", "poisson", False, "predefined"),
    ("poisson_deviance", "poisson", True, "random"),
    ("gamma_deviance", "gamma", False, "predefined"),
])
def test_parallel_matches_serial(blend_data, metric, objective, exposure, folds, caplog):
    data = replace(
        blend_data, objective=objective,
        target=blend_data.target + .1 if objective == "gamma" else blend_data.target,
        exposure=pl.Series(np.linspace(.5, 2, 120)) if exposure else None,
        offset=pl.Series(np.linspace(-.1, .1, 120)),
    )
    pipelines = make_pipelines(data)
    kwargs = dict(mode="oof", cv=CVConfig(n_splits=3, folds=folds), metric=metric, show_progress_bar=False)
    serial = CapturingBlender(n_jobs=1, **kwargs)
    parallel = CapturingBlender(n_jobs=2, **kwargs)
    reference = serial.fit(pipelines)
    with caplog.at_level(logging.INFO, logger="ins_gbm.ensemble.blending"):
        result = parallel.fit(pipelines)
    np.testing.assert_allclose(parallel.oof, serial.oof, rtol=1e-6)
    np.testing.assert_allclose(result.weights, reference.weights, atol=1e-5)
    np.testing.assert_allclose(result.predict(data), reference.predict(data), rtol=1e-5)
    assert "OOF prediction generation completed" in caplog.text
    assert "Blend weight optimization completed" in caplog.text
    assert pipelines[0].recipe.params["num_threads"] == 2


@pytest.mark.parametrize("feature_stage,features", [
    ("encoded", ["x", "z"]), ("model", ["latent__pls_1"]),
])
def test_manual_features_and_supervised_preprocessing(blend_data, feature_stage, features):
    steps = [] if feature_stage == "encoded" else [
        PreprocessingStep("latent", PLSReducer(n_components=1), ["x", "z"])
    ]
    recipe = ModelRecipe(
        LightGBMModel(), encoder=OneHotEncoder(), preprocessing=steps,
        params={"n_estimators": 5, "num_threads": 2},
    )
    pipeline = recipe.fit(blend_data, feature_names=features, feature_stage=feature_stage)
    serial = CapturingBlender(mode="oof", n_jobs=1, show_progress_bar=False)
    parallel = CapturingBlender(mode="oof", n_jobs=2, show_progress_bar=False)
    serial.fit([pipeline])
    parallel.fit([pipeline])
    np.testing.assert_allclose(parallel.oof, serial.oof, rtol=1e-6)


def test_selection_and_retuning(blend_data):
    from optuna.distributions import IntDistribution

    tuner = HyperparameterTuner(
        n_trials=2, cv_folds=2, use_data_folds=False, n_jobs=3, backend="process",
        show_progress_bar=False,
        search_space={"num_leaves": IntDistribution(3, 4), "n_jobs": IntDistribution(20, 30)},
    )
    selector = StagedImportanceSelector([
        ImportanceSelectionStage(LightGBMModel(), max_features=2,
                                 params={"n_estimators": 3, "n_jobs": 32})
    ])
    # Attach the tuning recipe after the base fit; the OOF folds do the tuning.
    pipelines = make_pipelines(blend_data, selection=selector)
    pipelines = [replace(p, recipe=replace(p.recipe, tuning=tuner)) for p in pipelines]
    fixed = _prepare_recipe(_FoldRecipe.from_pipeline(pipelines[0]), 2)
    assert fixed.recipe.tuning.n_jobs == 1
    assert fixed.recipe.tuning.backend == "thread"
    assert fixed.recipe.tuning.search_space.keys() == {"num_leaves"}
    assert fixed.recipe.selection.stages[0].params["num_threads"] == 2
    parallel = CapturingBlender(mode="oof", refit="retune", n_jobs=2, show_progress_bar=False)
    result = parallel.fit(pipelines)
    serial_pipelines = [replace(p, recipe=_prepare_recipe(_FoldRecipe.from_pipeline(p), 2).recipe)
                        for p in pipelines]
    serial = CapturingBlender(mode="oof", refit="retune", n_jobs=1, show_progress_bar=False)
    reference = serial.fit(serial_pipelines)
    np.testing.assert_allclose(parallel.oof, serial.oof, rtol=1e-6)
    np.testing.assert_allclose(result.weights, reference.weights, atol=1e-5)
    assert np.isfinite(parallel.oof).all()
    assert sum(result.weights) == pytest.approx(1)
    assert tuner.n_jobs == 3
    assert "n_jobs" in tuner.search_space
    assert selector.stages[0].params["n_jobs"] == 32


@dataclass
class RecordingModel(LightGBMModel):
    directory: str = ""
    fail: bool = False

    def fit(self, data, params=None, **kwargs):
        from pathlib import Path

        started = time.monotonic()
        time.sleep(.25)  # Ensure task intervals overlap without speed assertions.
        if self.fail:
            Path(self.directory, f"{os.getpid()}.pid").write_text(str(os.getpid()))
            raise ValueError("intentional fold failure")
        result = super().fit(data, params, **kwargs)
        record = {
            "pid": os.getpid(), "start": started, "end": time.monotonic(),
            "rows": data.features["x"].to_list(), "params": params,
            "env": os.environ["POLARS_MAX_THREADS"], "polars_threads": pl.thread_pool_size(),
        }
        Path(self.directory, f"{os.getpid()}-{time.monotonic_ns()}.json").write_text(json.dumps(record))
        return result


def test_real_workers_overlap_and_preserve_fold_isolation(blend_data, tmp_path, monkeypatch):
    monkeypatch.setenv("POLARS_MAX_THREADS", "17")
    base = make_pipelines(blend_data)[0]
    recipe = replace(base.recipe, model=RecordingModel(directory=str(tmp_path)))
    pipeline = replace(base, recipe=recipe)
    result = EnsemblePipeline(
        [pipeline], blend_mode="oof", n_jobs=2, threads_per_worker=2, show_progress_bar=False,
    ).run()
    records = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert len(records) == 3
    if result.configuration["effective_n_jobs"] > 1:
        assert len({record["pid"] for record in records}) == 2
        assert any(a["start"] < b["end"] and b["start"] < a["end"]
                   for a in records for b in records if a["pid"] != b["pid"])
    expected = [set(blend_data.features.filter(blend_data.cv_fold != fold)["x"].to_list())
                for fold in (10, 30, 70)]
    assert all(set(record["rows"]) in expected for record in records)
    for record in records:
        assert record["pid"] != os.getpid()
        assert record["polars_threads"] == int(record["env"]) <= 2
        assert record["params"]["num_threads"] <= 2
    assert os.environ["POLARS_MAX_THREADS"] == "17"
    assert result.configuration["n_jobs"] == 2


@pytest.mark.parametrize("name", ["n_jobs", "threads_per_worker"])
@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "4"])
def test_invalid_settings(name, bad):
    with pytest.raises(ValueError, match=name):
        _worker_settings(**{**dict(n_jobs=4, threads_per_worker=2, tasks=10), name: bad})


def test_cpu_and_task_caps(monkeypatch):
    monkeypatch.setattr("ins_gbm.ensemble._oof.cpu_count", lambda: 8)
    assert _worker_settings(4, 2, 20)["effective_n_jobs"] == 4
    assert _worker_settings(20, 20, 3) == dict(
        n_jobs=20, threads_per_worker=20, effective_n_jobs=3, effective_threads_per_worker=2,
    )
    assert _worker_settings(1, 2, 10)["effective_threads_per_worker"] is None


@pytest.mark.parametrize("framework,aliases,canonical", [
    ("lightgbm", ["n_jobs", "nthread", "num_thread", "nthreads", "num_threads"], "num_threads"),
    ("xgboost", ["n_jobs", "nthread"], "nthread"),
    ("catboost", ["thread_count"], "thread_count"),
    ("random_forest", ["n_jobs"], "n_jobs"),
])
def test_thread_aliases(framework, aliases, canonical):
    original = {**dict.fromkeys(aliases, 100), "seed": 7}
    assert _limited_params(original, framework, 2) == {canonical: 2, "seed": 7}
    assert original[aliases[0]] == 100


class UnserializableModel(LightGBMModel):
    def __getstate__(self):
        raise TypeError("cannot serialize this component")

    def __deepcopy__(self, memo):
        return UnserializableModel()


def test_serialization_error_and_serial_escape_hatch(blend_data):
    base = make_pipelines(blend_data)[0]
    pipeline = replace(base, recipe=replace(base.recipe, model=UnserializableModel()))
    with pytest.raises(TypeError, match="model 1, fold 1.*n_jobs=1"):
        BlendingEnsemble(mode="oof", n_jobs=2, show_progress_bar=False).fit([pipeline])
    assert BlendingEnsemble(mode="oof", n_jobs=1, show_progress_bar=False).fit([pipeline]).weights == [1.]


def test_worker_error_has_fold_context(blend_data, tmp_path):
    base = make_pipelines(blend_data)[0]
    pipeline = replace(base, recipe=replace(base.recipe, model=RecordingModel(directory=str(tmp_path), fail=True)))
    with pytest.raises(RuntimeError, match="OOF model 1, fold .*intentional fold failure") as error:
        BlendingEnsemble(mode="oof", n_jobs=2, show_progress_bar=False).fit([pipeline])
    assert error.value.__cause__ is not None
    for path in tmp_path.glob("*.pid"):
        with pytest.raises(ProcessLookupError):
            os.kill(int(path.read_text()), 0)


def test_serial_model_thread_settings_are_unchanged(blend_data, monkeypatch):
    base = make_pipelines(blend_data)[0]
    original_fit = LightGBMModel.fit
    observed = []

    def record(self, data, params=None, **kwargs):
        observed.append(params.copy())
        return original_fit(self, data, params, **kwargs)

    monkeypatch.setattr(LightGBMModel, "fit", record)
    BlendingEnsemble(mode="oof", n_jobs=1, show_progress_bar=False).fit([base])
    assert len(observed) == 3
    assert all(params == base.fitted_model.params for params in observed)


def test_parallel_configuration_survives_persistence(blend_data, tmp_path):
    from ins_gbm.persistence.ensemble_io import load_ensemble

    pipelines = make_pipelines(blend_data)
    result = EnsemblePipeline(pipelines, blend_mode="oof", show_progress_bar=False).run()
    assert result.configuration["n_jobs"] == 4
    assert result.configuration["threads_per_worker"] == 2
    result.save(str(tmp_path))
    loaded = load_ensemble(str(tmp_path))
    assert loaded.configuration == result.configuration
    np.testing.assert_allclose(loaded.predict(blend_data), result.predict(blend_data))


def test_loading_blend_state_without_parallel_settings(blend_data, tmp_path):
    import cloudpickle

    from ins_gbm.persistence.ensemble_io import load_ensemble

    pipelines = make_pipelines(blend_data)
    result = EnsemblePipeline(pipelines, blend_weights=[.5, .5]).run()
    # Emulate state written before the new optional fitted-blend field existed.
    del result.ensemble.oof_configuration
    restored = cloudpickle.loads(cloudpickle.dumps(result))
    assert restored.ensemble.oof_configuration == {}
    restored.save(str(tmp_path))
    np.testing.assert_allclose(load_ensemble(str(tmp_path)).predict(blend_data), result.predict(blend_data))


@pytest.mark.parametrize("framework", ["xgboost", "catboost", "random_forest"])
def test_other_model_backends(blend_data, framework):
    from ins_gbm.models.catboost import CatBoostModel
    from ins_gbm.models.random_forest import RandomForestModel
    from ins_gbm.models.xgboost import XGBoostModel

    if framework == "xgboost":
        pytest.importorskip("xgboost")
        model, params = XGBoostModel(), {"n_estimators": 5, "max_depth": 2, "nthread": 2}
    elif framework == "catboost":
        pytest.importorskip("catboost")
        model, params = CatBoostModel(), {"iterations": 5, "depth": 2, "thread_count": 2}
    else:
        model, params = RandomForestModel(), {"n_estimators": 5, "max_depth": 2, "n_jobs": 2}
    pipeline = ModelRecipe(model, encoder=OneHotEncoder(), params=params).fit(blend_data)
    serial = CapturingBlender(mode="oof", n_jobs=1, show_progress_bar=False)
    parallel = CapturingBlender(mode="oof", n_jobs=2, show_progress_bar=False)
    serial.fit([pipeline])
    parallel.fit([pipeline])
    np.testing.assert_allclose(parallel.oof, serial.oof, rtol=1e-6)


@pytest.mark.parametrize("exit_mode", ["success", "failure", "interrupt"])
def test_bounded_submission_out_of_order_and_cleanup(monkeypatch, exit_mode):
    """Exercise scheduling deterministically without worker timing assumptions."""
    from concurrent.futures import Future
    import cloudpickle

    executors = []

    class Executor:
        def __init__(self, **kwargs):
            self.submitted = []
            self.shutdown_calls = []
            executors.append(self)

        def submit(self, fn, payload):
            future = Future()
            task = cloudpickle.loads(payload)
            if exit_mode == "failure":
                future.set_exception(ValueError("broken task"))
            else:
                future.set_result(task)
            self.submitted.append(future)
            return future

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

    def complete_last(pending, **kwargs):
        return {list(pending)[-1]}, set(list(pending)[:-1])

    monkeypatch.setattr("ins_gbm.ensemble._oof.ProcessPoolExecutor", Executor)
    monkeypatch.setattr("ins_gbm.ensemble._oof.wait", complete_last)
    tasks = ((None, None, 0, index) for index in range(5))
    results = _parallel_folds(tasks, 2, 2)
    if exit_mode == "failure":
        with pytest.raises(RuntimeError, match="model 1, fold 2.*broken task"):
            next(results)
    else:
        assert next(results)[3] == 1  # Second task completes before the first.
        assert len(executors[0].submitted) == 2
        if exit_mode == "interrupt":
            with pytest.raises(KeyboardInterrupt):
                results.throw(KeyboardInterrupt)
        else:
            assert [result[3] for result in results] == [2, 3, 4, 0]
    assert executors[0].shutdown_calls == [
        {"wait": True, **({"kill_workers": True} if exit_mode != "success" else {})}
    ]
