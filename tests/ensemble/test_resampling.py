from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from ins_gbm import EnsemblePipeline, ModelData, ModelRecipe
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.ensemble.resampling import _member_params
from ins_gbm.preprocessing.encoder import OneHotEncoder
from ins_gbm.preprocessing.pca import PCAReducer


@pytest.fixture
def template():
    n = 48
    data = ModelData(
        features=pl.DataFrame({"row": range(n), "x": np.sin(np.arange(n)), "cat": ["a", "b"] * (n // 2)}),
        target=pl.Series(np.array([0., 1., 2., 3.] * (n // 4))),
        feature_names=["row", "x", "cat"],
        exposure=pl.Series(np.linspace(.5, 2., n)),
        weight=pl.Series(np.linspace(1., 3., n)),
        offset=pl.Series(np.linspace(-.2, .2, n)),
        cv_fold=pl.Series(np.arange(n) % 3),
        objective="poisson",
    )
    fitted = ModelRecipe(
        model=LightGBMModel(), encoder=OneHotEncoder(),
        params={"n_estimators": 5, "min_data_in_leaf": 2, "bagging_fraction": .8, "num_threads": 1},
    ).fit(data)
    return data, fitted


@pytest.mark.parametrize("method", ["seed_average", "bagging"])
def test_members_reproducible_independent_and_average(template, method):
    data, fitted = template
    original_params = dict(fitted.fitted_model.params)
    first = EnsemblePipeline([fitted], method=method, n_members=3).run()
    second = EnsemblePipeline([fitted], method=method, n_members=3).run()
    assert len(set(first.configuration["member_seeds"])) == 3
    assert first.configuration == second.configuration
    expected = np.mean([member.predict(data).to_numpy() for member in first.base_pipelines], axis=0)
    np.testing.assert_allclose(first.predict(data), expected, rtol=1e-6)
    np.testing.assert_allclose(first.predict(data), second.predict(data), rtol=1e-6)
    for member, other in zip(first.base_pipelines, second.base_pipelines):
        assert member is not fitted
        assert member.recipe.tuning is None
        assert member.encoder is not fitted.encoder
        assert member.fitted_model.params["n_estimators"] == 5
        assert member.fitted_model.params["seed"] == member.metadata.random_seeds["member"]
        assert member.raw_train_data.features.equals(other.raw_train_data.features)
    assert fitted.fitted_model.params == original_params
    assert fitted.raw_train_data is data


def test_bagging_samples_keep_row_fields_aligned(template):
    data, fitted = template
    result = EnsemblePipeline([fitted], method="bagging", n_members=2, sample_fraction=.5).run()
    for member in result.base_pipelines:
        sample = member.raw_train_data
        assert sample.n_rows == 24
        indices = sample.features["row"].to_numpy()
        for field in ("target", "exposure", "weight", "offset", "cv_fold"):
            assert getattr(sample, field).to_list() == getattr(data, field)[indices].to_list()
        assert len(set(indices)) < len(indices)  # Bootstrap duplicates retained.
    subset = EnsemblePipeline(
        [fitted], method="bagging", n_members=1, sample_fraction=.5, bootstrap=False,
    ).run().base_pipelines[0].raw_train_data
    assert subset.features["row"].n_unique() == 24


@pytest.mark.parametrize("objective", ["poisson", "gamma"])
def test_resampling_response_scale(template, objective):
    data, fitted = template
    data = replace(data, objective=objective, target=data.target + 1)
    fitted = replace(fitted, raw_train_data=data)
    fitted = ModelRecipe(
        model=LightGBMModel(objective=objective), encoder=OneHotEncoder(),
        params={"n_estimators": 4, "min_data_in_leaf": 2, "num_threads": 1},
    ).fit(data)
    result = EnsemblePipeline([fitted], method="bagging", n_members=2).run()
    changed = replace(data, exposure=data.exposure * 3)
    expected = result.predict(data).to_numpy() * (3 if objective == "poisson" else 1)
    np.testing.assert_allclose(result.predict(changed), expected, rtol=2e-6)


def test_reuses_params_without_tuning_and_preserves_feature_stages(template, monkeypatch):
    data, _ = template
    fitted = ModelRecipe(
        model=LightGBMModel(), encoder=OneHotEncoder(), preprocessing=[PCAReducer(2)],
        params={"n_estimators": 3, "num_threads": 1},
    ).fit(data, feature_names=["row", "x"], feature_stage="encoded")
    # Exercise simultaneous fixed encoded and post-preprocessing subsets.
    fitted = replace(fitted, model_selected_features=["pca_1"])
    from ins_gbm.tuning.tuner import HyperparameterTuner
    fitted.recipe.tuning = HyperparameterTuner()
    monkeypatch.setattr(HyperparameterTuner, "tune", lambda *a, **k: pytest.fail("must not tune"))
    result = EnsemblePipeline([fitted], method="bagging", n_members=2).run()
    for member in result.base_pipelines:
        assert member.selected_features == ["row", "x"]
        assert member.model_selected_features == ["pca_1"]
        assert member.fitted_model.feature_names == ["pca_1"]
        assert member.preprocessors[0] is not fitted.preprocessors[0]
        assert len(member.predict(data)) == data.n_rows


@pytest.mark.parametrize("framework, params, expected", [
    ("lightgbm", {"random_state": 1, "bagging_seed": 2, "data_seed": 3}, "seed"),
    ("xgboost", {"random_state": 1}, "seed"),
    ("catboost", {"random_state": 1}, "random_seed"),
    ("random_forest", {"random_state": 1}, "random_state"),
])
def test_backend_seed_aliases(framework, params, expected):
    changed = _member_params(params, framework, 123)
    assert changed[expected] == 123
    assert params != changed
    if framework != "random_forest":
        assert "random_state" not in changed
    if framework == "lightgbm":
        assert "data_seed" not in changed
        assert changed["bagging_seed"] == changed["data_random_seed"] == 123


@pytest.mark.parametrize("kwargs, message", [
    ({"n_members": 0}, "n_members"),
    ({"n_members": True}, "n_members"),
    ({"n_members": 1.5}, "n_members"),
    ({"seed": -1}, "seed"),
    ({"sample_fraction": 0}, "sample_fraction"),
    ({"sample_fraction": 1.1}, "sample_fraction"),
    ({"sample_fraction": .001}, "at least one row"),
    ({"bootstrap": "yes"}, "bootstrap"),
])
def test_invalid_options(template, kwargs, message):
    with pytest.raises(ValueError, match=message):
        EnsemblePipeline([template[1]], method="bagging", **kwargs).run()


def test_requires_one_template_with_training_data(template):
    _, fitted = template
    with pytest.raises(ValueError, match="exactly one"):
        EnsemblePipeline([fitted, fitted], method="bagging").run()
    with pytest.raises(RuntimeError, match="Training data is not attached"):
        EnsemblePipeline([replace(fitted, raw_train_data=None)], method="seed_average").run()


def test_failed_member_identifies_seed(template, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("bad sample")
    monkeypatch.setattr("ins_gbm.ensemble.resampling._refit_pipeline", fail)
    with pytest.raises(RuntimeError, match=r"member 1 \(seed=\d+\) failed: bad sample"):
        EnsemblePipeline([template[1]], method="bagging").run()


@pytest.mark.parametrize("framework", ["lightgbm", "xgboost", "catboost", "random_forest"])
@pytest.mark.parametrize("objective", ["poisson", "gamma"])
@pytest.mark.parametrize("method", ["seed_average", "bagging"])
def test_supported_backends_fit_predict_and_persist(template, tmp_path, framework, objective, method):
    from ins_gbm import load_ensemble
    from ins_gbm.models.catboost import CatBoostModel
    from ins_gbm.models.random_forest import RandomForestModel
    from ins_gbm.models.xgboost import XGBoostModel

    if framework != "random_forest":
        pytest.importorskip(framework)
    model_class = {
        "lightgbm": LightGBMModel, "xgboost": XGBoostModel,
        "catboost": CatBoostModel, "random_forest": RandomForestModel,
    }[framework]
    data, _ = template
    data = replace(data, objective=objective, target=data.target + 1, offset=None)
    params = {
        "lightgbm": {"n_estimators": 3, "min_data_in_leaf": 2, "num_threads": 1},
        "xgboost": {"n_estimators": 3, "max_depth": 3, "nthread": 1},
        "catboost": {"iterations": 3, "depth": 3, "thread_count": 1},
        "random_forest": {"n_estimators": 3, "min_samples_leaf": 2, "n_jobs": 1},
    }[framework]
    fitted = ModelRecipe(model=model_class(objective=objective), encoder=OneHotEncoder(), params=params).fit(data)
    result = EnsemblePipeline([fitted], method=method, n_members=2).run()
    predictions = result.predict(data).to_numpy()
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()
    for member, seed in zip(result.base_pipelines, result.configuration["member_seeds"]):
        seed_key = "random_seed" if framework == "catboost" else "random_state" if framework == "random_forest" else "seed"
        assert member.fitted_model.params[seed_key] == seed
    if framework == "catboost":
        assert result.base_pipelines[0].fitted_model.importance_fn.pool is not None
    result.save(tmp_path)
    if framework == "catboost":
        assert result.base_pipelines[0].fitted_model.importance_fn.pool is not None
        assert result.base_pipelines[0].fitted_model.importance_fn.loss_function_importance is None
    loaded = load_ensemble(tmp_path)
    if framework == "catboost":
        assert loaded.base_pipelines[0].fitted_model.importance_fn.pool is None
    np.testing.assert_allclose(loaded.predict(data), predictions, rtol=1e-12)
    assert all(member.raw_train_data is None for member in loaded.base_pipelines)


def test_configured_selection_refits_per_member_with_model_subset(template):
    class Selector:
        calls = 0

        def fit(self, data):
            self.calls += 1
            self.names = ["row", "x"]
            return self

        def selected_features(self):
            return self.names

    data, fitted = template
    selector = Selector()
    recipe = replace(fitted.recipe, selection=selector, preprocessing=[PCAReducer(2)])
    fitted = replace(fitted, recipe=recipe, model_selected_features=["pca_1"])
    result = EnsemblePipeline([fitted], method="bagging", n_members=2).run()
    assert selector.calls == 0
    for member in result.base_pipelines:
        assert member.selection_results[0].calls == 1
        assert member.selected_features == ["row", "x"]
        assert member.fitted_model.feature_names == ["pca_1"]
        assert len(member.predict(data)) == data.n_rows
    assert result.base_pipelines[0].selection_results[0] is not result.base_pipelines[1].selection_results[0]
