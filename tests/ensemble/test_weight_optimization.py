from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from scipy.optimize import minimize_scalar

from ins_gbm.data.folds import CVConfig
from ins_gbm.data.model_data import ModelData
from ins_gbm.ensemble.blending import BlendingEnsemble, _optimize_weights
from ins_gbm.ensemble.pipeline import EnsemblePipeline
from ins_gbm.evaluation.metrics import (
    _poisson_rate_metric_inputs,
    gamma_deviance,
    poisson_deviance,
)
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import ModelPipeline, ModelRecipe


@pytest.fixture
def folded_pipeline():
    data = ModelData(
        features=pl.DataFrame({"x": list(range(12))}),
        target=pl.Series([1., 2., 4., 3.] * 3),
        feature_names=["x"],
        objective="poisson",
        cv_fold=pl.Series([10, 30, 70] * 4),
    )
    pipeline = ModelPipeline(
        data, ModelRecipe(model=LightGBMModel(), params={"n_estimators": 3})
    ).run()
    return data, pipeline


@pytest.mark.parametrize("method", ["blending", "stacking"])
@pytest.mark.parametrize("config", [{}, {"use_data_folds": True}, {"cv": CVConfig(folds="predefined")}])
def test_oof_reuses_predefined_groups(folded_pipeline, monkeypatch, method, config):
    data, pipeline = folded_pipeline
    seen = []
    original_fit = LightGBMModel.fit

    def record_fit(self, train, *args, **kwargs):
        seen.append(train.features["x"].to_list())
        return original_fit(self, train, *args, **kwargs)

    monkeypatch.setattr(LightGBMModel, "fit", record_fit)
    EnsemblePipeline(
        [pipeline], method=method, blend_mode="oof", cv_folds=2,
        metric="auto", **config,
    ).run()
    assert seen == [
        data.features.filter(data.cv_fold != fold)["x"].to_list()
        for fold in [10, 30, 70]
    ]


@pytest.mark.parametrize("method", ["blending", "stacking"])
def test_random_folds_can_override_data_folds(folded_pipeline, monkeypatch, method):
    _, pipeline = folded_pipeline
    seen = []
    original_fit = LightGBMModel.fit

    def record_fit(self, train, *args, **kwargs):
        seen.append(train.n_rows)
        return original_fit(self, train, *args, **kwargs)

    monkeypatch.setattr(LightGBMModel, "fit", record_fit)
    EnsemblePipeline(
        [pipeline], method=method, blend_mode="oof", cv_folds=2,
        use_data_folds=False,
    ).run()
    assert seen == [6, 6]


@pytest.mark.parametrize("config, message", [
    ({"use_data_folds": True}, "cv_fold"),
    ({"cv": CVConfig(), "use_data_folds": False}, "either cv or use_data_folds"),
])
def test_invalid_fold_configuration(folded_pipeline, config, message):
    data, pipeline = folded_pipeline
    pipeline = replace(pipeline, raw_train_data=replace(data, cv_fold=None))
    with pytest.raises(ValueError, match=message):
        BlendingEnsemble(mode="oof", **config).fit([pipeline])


@pytest.mark.parametrize("metric", ["mse", "poisson_deviance", "gamma_deviance"])
def test_weights_minimize_selected_loss(metric):
    # Float32 inputs exercise finite-difference precision in SLSQP.
    preds = np.array([[.2, 3], [1, 5], [4, 10], [7, 2]], dtype=np.float32)
    actual = np.array([1., 2., 12., 4.])
    if metric == "poisson_deviance":
        actual[0] = 0  # Zero claim counts are valid for Poisson deviance.
    observation_weight = np.array([1., 3., 2., 5.])
    exposure = np.array([.5, 2., 4., 1.])

    def loss(w):
        blended = preds.astype(float) @ np.array([w, 1 - w])
        if metric == "mse":
            return np.average((actual - blended) ** 2, weights=observation_weight)
        y, mu, weight = pl.Series(actual), pl.Series(blended), pl.Series(observation_weight)
        if metric == "poisson_deviance":
            return poisson_deviance(*_poisson_rate_metric_inputs(
                y, mu, pl.Series(exposure), weight,
            ))
        return gamma_deviance(y, mu, weight)

    reference = minimize_scalar(loss, bounds=(0, 1), method="bounded")
    weights = _optimize_weights(
        preds, actual, metric=metric, sample_weight=observation_weight,
        exposure=exposure,
    )
    assert weights.sum() == pytest.approx(1)
    assert np.all(weights >= 0)
    assert loss(weights[0]) <= reference.fun + 1e-7
    assert weights[0] == pytest.approx(reference.x, abs=1e-4)


@pytest.mark.parametrize("objective", ["poisson", "gamma"])
def test_auto_metric_uses_fitted_distribution(folded_pipeline, monkeypatch, objective):
    data, pipeline = folded_pipeline
    data = replace(data, objective=objective)
    pipeline = ModelPipeline(
        data, ModelRecipe(model=LightGBMModel(objective=objective), params={"n_estimators": 3})
    ).run()
    metrics = []

    def record_weights(preds, actual, *, metric, **kwargs):
        metrics.append(metric)
        return np.array([1.])

    monkeypatch.setattr("ins_gbm.ensemble.blending._optimize_weights", record_weights)
    EnsemblePipeline([pipeline], blend_mode="validation", validation_data=data, metric="auto").run()
    assert metrics == [f"{objective}_deviance"]


@pytest.mark.parametrize("metric", ["unknown", "gamma_deviance"])
def test_invalid_metric_raises(folded_pipeline, metric):
    data, pipeline = folded_pipeline
    with pytest.raises(ValueError, match="metric"):
        BlendingEnsemble(mode="validation", metric=metric).fit([pipeline], data)


def test_deviance_rejects_nonpositive_predictions():
    with pytest.raises(ValueError, match="positive"):
        _optimize_weights(np.array([[0., 1.]]), np.array([1.]), metric="poisson_deviance")


def test_optimizer_failure_is_reported(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr("scipy.optimize.minimize", lambda *a, **k: SimpleNamespace(
        success=False, message="iteration limit", fun=1., x=np.array([.5, .5]),
    ))
    with pytest.raises(RuntimeError, match="iteration limit"):
        _optimize_weights(np.ones((2, 2)), np.ones(2))
