"""Regression coverage for model API contracts and tuning correctness."""

from dataclasses import replace

import numpy as np
import optuna
import polars as pl
import pytest

from ins_gbm import (
    BorutaSelector, CatBoostModel, HyperparameterTuner, LightGBMModel,
    ModelData, RandomForestModel, XGBoostModel,
)
from ins_gbm.evaluation.metrics import (
    _poisson_rate_metric_inputs, gamma_deviance, mae, poisson_deviance, rmse,
)
from ins_gbm.selection import ImportancePruner
from ins_gbm.tuning.tuner import _study_history, _suggest_from_distribution


def _data(objective="poisson", n=60):
    return ModelData(
        features=pl.DataFrame({"x": np.linspace(-1, 1, n)}),
        target=pl.Series(np.resize([1., 2., 4.], n)),
        feature_names=["x"], objective=objective,
        exposure=pl.Series(np.linspace(.3, 2, n)) if objective != "gamma" else None,
        weight=pl.Series(np.linspace(.5, 3, n)),
    )


def _model(kind):
    dependency = {"rf": "sklearn"}.get(kind, kind)
    pytest.importorskip(dependency)
    return {
        "lightgbm": (LightGBMModel(), {"n_estimators": 5, "num_threads": 1}),
        "xgboost": (XGBoostModel(), {"n_estimators": 5, "nthread": 1}),
        "catboost": (CatBoostModel(), {"iterations": 5, "thread_count": 1}),
        "rf": (RandomForestModel(), {"n_estimators": 5, "n_jobs": 1}),
    }[kind]


@pytest.mark.parametrize("kind", ["lightgbm", "xgboost", "catboost", "rf"])
@pytest.mark.parametrize("objective", ["poisson", "gamma"])
def test_model_scales_and_exposure_contract(kind, objective):
    model, params = _model(kind)
    data = _data(objective)
    fitted = model.fit(data, params)
    response = fitted.predict(data).to_numpy()
    assert np.isfinite(response).all() and (response > 0).all()
    np.testing.assert_allclose(np.exp(fitted.predict(data, "link")), response, rtol=2e-6)
    assert fitted.feature_importance()["feature"].to_list() == ["x"]
    if objective == "poisson":
        np.testing.assert_allclose(
            fitted.predict(data, "rate"), response / data.exposure.to_numpy(), rtol=2e-6,
        )
        unit = replace(data, exposure=pl.Series(np.ones(data.n_rows)))
        omitted = replace(data, exposure=None)
        np.testing.assert_allclose(fitted.predict(unit), fitted.predict(omitted), rtol=2e-6)
    else:
        with pytest.raises(ValueError, match="rate.*gamma"):
            fitted.predict(data, "rate")


@pytest.mark.parametrize("kind", ["lightgbm", "xgboost", "catboost"])
@pytest.mark.parametrize("objective", ["poisson", "gamma"])
@pytest.mark.parametrize("training_offset", [False, True])
def test_offsets_add_once_on_the_link_scale(kind, objective, training_offset):
    model, params = _model(kind)
    data = replace(_data(objective), exposure=None)
    if training_offset:
        data = replace(data, offset=pl.Series(np.linspace(-.3, .5, data.n_rows)))
    fitted = model.fit(data, params)
    plain = replace(data, offset=None)
    shifted = replace(plain, offset=pl.Series(np.full(data.n_rows, np.log(2))))
    np.testing.assert_allclose(fitted.predict(shifted), 2 * fitted.predict(plain), rtol=2e-6)


@pytest.mark.parametrize("kind,param", [
    ("lightgbm", "objective"), ("lightgbm", "application"),
    ("xgboost", "objective"), ("catboost", "loss_function"),
])
def test_conflicting_native_objectives_are_rejected(kind, param):
    model, params = _model(kind)
    with pytest.raises(ValueError, match="objective"):
        model.fit(_data(), {**params, param: "RMSE"})


def test_catboost_accepts_matching_objective_alias():
    model, params = _model("catboost")
    fitted = model.fit(_data(), {**params, "objective": "Poisson"})
    assert fitted.params["loss_function"] == "Poisson"
    assert "objective" not in fitted.params


@pytest.mark.parametrize("kind", ["lightgbm", "xgboost", "catboost", "rf"])
@pytest.mark.parametrize("objective", ["poisson", "gamma"])
def test_native_search_space_parameters_fit_and_refit(kind, objective):
    model, base = _model(kind)
    space = model.default_search_space()
    rounds = "iterations" if kind == "catboost" else "n_estimators"
    space[rounds] = optuna.distributions.IntDistribution(3, 3)
    params, history = HyperparameterTuner(
        n_trials=1, cv_folds=2, search_space=space, show_progress_bar=False,
    ).tune(_data(objective), model, base_params=base)
    assert np.isfinite(history["value"][0])
    fitted = model.fit(_data(objective), params)
    assert np.isfinite(fitted.predict(_data(objective)).to_numpy()).all()


def test_lightgbm_subsample_is_effective_and_explicit_disable_is_respected():
    model, params = _model("lightgbm")
    enabled = model.fit(_data(), {**params, "subsample": .7})
    assert enabled.model.params["bagging_freq"] == 1
    disabled = model.fit(_data(), {**params, "subsample": .7, "subsample_freq": 0})
    assert "bagging_freq" not in disabled.params


@pytest.mark.parametrize("distribution,value", [
    (optuna.distributions.IntDistribution(2, 10, step=2), 6),
    (optuna.distributions.FloatDistribution(.1, .9, step=.2), .5),
])
def test_suggestions_preserve_distribution_steps(distribution, value):
    trial = optuna.trial.FixedTrial({"parameter": value})
    _suggest_from_distribution(trial, "parameter", distribution)
    assert trial.distributions["parameter"] == distribution


def test_pruned_trials_are_excluded_from_completed_history():
    study = optuna.create_study()
    study.add_trial(optuna.trial.create_trial(value=3))
    study.add_trial(optuna.trial.create_trial(
        state=optuna.trial.TrialState.PRUNED, intermediate_values={0: .01},
    ))
    history = _study_history(study)
    assert history["trial"].to_list() == [0]


class _ConstantModel:
    def default_search_space(self):
        return {"constant": optuna.distributions.FloatDistribution(.5, 3)}

    def fit(self, data, params=None):
        constant = params["constant"]

        class Fitted:
            def predict(self, data, prediction_type="response"):
                return pl.Series(np.full(data.n_rows, constant))

        return Fitted()


@pytest.mark.parametrize("metric,objective", [
    ("poisson_deviance", "poisson"), ("gamma_deviance", "gamma"),
    ("poisson_deviance", None),
    ("rmse", "poisson"), ("mae", "poisson"),
])
@pytest.mark.parametrize("backend", ["thread", "process"])
def test_unequal_folds_score_the_pooled_weighted_predictions(metric, objective, backend):
    data = replace(_data(objective, n=12), cv_fold=pl.Series([0] * 3 + [1] * 9))
    tuner = HyperparameterTuner(
        n_trials=1, metric=metric, search_space={}, backend=backend,
        show_progress_bar=False,
    )
    params, history = tuner.tune(data, _ConstantModel(), base_params={"constant": 1.5})
    prediction = pl.Series(np.full(data.n_rows, 1.5))
    actual, weights = data.target, data.weight
    if metric == "poisson_deviance":
        actual, prediction, weights = _poisson_rate_metric_inputs(
            actual, prediction, data.exposure, weights,
        )
    function = {"poisson_deviance": poisson_deviance, "gamma_deviance": gamma_deviance,
                "rmse": rmse, "mae": mae}[metric]
    assert history["value"][0] == pytest.approx(function(actual, prediction, weights))
    assert params == {"constant": 1.5}


@pytest.mark.parametrize("cache", [False, True])
def test_transform_cache_reuses_training_folds_without_validation_leakage(cache):
    seen_targets = []

    class Preprocessor:
        def fit(self, features, target):
            seen_targets.append(target.to_list())
            return self

        def transform(self, features):
            return features

    data = replace(_data(n=12), target=pl.Series(range(12)),
                   cv_fold=pl.Series([0] * 3 + [1] * 9))
    _, history = HyperparameterTuner(
        n_trials=3, cache_transforms=cache, search_space={}, show_progress_bar=False,
    ).tune(data, _ConstantModel(), preprocessors=[Preprocessor()], base_params={"constant": 1.5})
    assert len(seen_targets) == (2 if cache else 6)
    assert all(rows == list(range(3)) or rows == list(range(3, 12)) for rows in seen_targets)
    assert history["value"].n_unique() == 1


@pytest.mark.parametrize("kwargs", [{"n_trials": 0}, {"n_trials": True}, {"n_trials": 1.5}])
def test_invalid_trial_counts_fail_before_training(kwargs):
    with pytest.raises(ValueError, match="n_trials"):
        HyperparameterTuner(**kwargs).tune(_data(), _ConstantModel())


@pytest.mark.parametrize("kwargs", [{"top_n": 0}, {"top_n": -1}, {"percentile": 101},
                                   {"threshold": float("nan")}])
def test_invalid_pruning_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ImportancePruner(**kwargs)


@pytest.mark.parametrize("kwargs", [{"max_iter": 0}, {"alpha": 0}, {"base_estimator": "typo"}])
def test_invalid_boruta_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        BorutaSelector(**kwargs)


def test_boruta_adjusts_final_tests_across_features(monkeypatch):
    # Six perfect hits yield p=.03125: significant per feature, but not
    # after adjustment for the two input features.
    data = _data().with_features(pl.DataFrame({"a": range(60), "b": range(60)}))

    class Fitted:
        def feature_importance(self):
            return pl.DataFrame({"feature": ["a", "b", "shadow__a", "shadow__b"],
                                 "importance": [10., 10., 1., 1.]})

    monkeypatch.setattr(BorutaSelector, "_fit_base", lambda *args: Fitted())
    corrected = BorutaSelector(max_iter=6).fit(data)
    uncorrected = BorutaSelector(max_iter=6, multiple_testing="none").fit(data)
    assert corrected.confirmed_features() == []
    assert uncorrected.confirmed_features() == ["a", "b"]


@pytest.mark.parametrize("kind", ["lightgbm", "xgboost", "catboost", "rf"])
def test_model_objective_checks_targets_when_data_objective_is_unspecified(kind):
    model, params = _model(kind)
    model.objective = "gamma"
    data = replace(_data(), objective=None, target=pl.Series(np.zeros(60)))
    with pytest.raises(ValueError, match="strictly positive"):
        model.fit(data, params)


def test_gini_supports_the_declared_numpy_126_api(monkeypatch):
    from ins_gbm.evaluation.metrics import normalized_gini

    monkeypatch.delattr(np, "trapezoid", raising=False)
    target = pl.Series([1., 2., 3.])
    assert normalized_gini(target, target) == pytest.approx(1.)
