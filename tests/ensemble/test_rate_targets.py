"""Rate targets remain on their supplied scale through every ensemble workflow."""

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import PoissonRegressor

from ins_gbm import (
    CVConfig, EnsemblePipeline, LightGBMModel, ModelData, ModelRecipe,
    load_ensemble, load_model_data,
)
from ins_gbm.ensemble.stacking import StackingEnsemble
from ins_gbm.evaluation.metrics import compute_metrics


@pytest.fixture
def rate_pipeline(tmp_path):
    n = 48
    frame = pl.DataFrame({
        "x": np.linspace(-1, 1, n),
        "claim_rate": np.resize([0., .15, .4, .85], n),
        "weight": np.linspace(.5, 2, n),
        "fold": np.arange(n) % 3,
        # A column called exposure must not become a model offset automatically.
        "exposure": np.linspace(.2, 3, n),
    })
    path = tmp_path / "rates.parquet"
    frame.write_parquet(path)
    data = load_model_data(
        str(path), target="claim_rate", feature_cols=["x"],
        weight="weight", cv_fold="fold",
    )
    recipe = ModelRecipe(
        LightGBMModel(),
        params={"n_estimators": 4, "num_threads": 1, "min_data_in_leaf": 2},
    )
    return data, recipe.fit(data)


@pytest.mark.parametrize("method,mode", [
    ("blending", "fixed"), ("blending", "validation"), ("blending", "oof"),
    ("stacking", "fixed"), ("seed_average", "fixed"), ("bagging", "fixed"),
])
def test_rate_ensemble_prediction_evaluation_cv_and_roundtrip(rate_pipeline, tmp_path, method, mode):
    data, fitted = rate_pipeline
    assert data.exposure is None
    np.testing.assert_allclose(fitted.predict(data), fitted.predict(data, "rate"))
    candidate = EnsemblePipeline(
        [fitted], method=method, blend_mode=mode, blend_weights=[1.0],
        validation_data=data if mode == "validation" else None,
        n_members=2, cv_folds=3, metric="auto",
    )
    result = candidate.run()
    predicted = result.predict(data)
    assert predicted.is_finite().all() and predicted.min() > 0
    for member in result.base_pipelines:
        assert member.raw_train_data.exposure is None
    if method != "stacking":
        expected = np.mean([member.predict(data).to_numpy() for member in result.base_pipelines], axis=0)
        np.testing.assert_allclose(predicted, expected, rtol=2e-6)
    expected_metrics = compute_metrics(
        objective="poisson", actual=data.target, predicted=predicted, weight=data.weight,
    )
    report = result.evaluate(data)
    np.testing.assert_allclose(report.metrics()["value"], expected_metrics["value"])
    destination = tmp_path / "saved"
    result.save(str(destination))
    restored = load_ensemble(str(destination))
    np.testing.assert_allclose(restored.predict(data), predicted, rtol=2e-6)
    # Outer CV uses an internal validation subset rather than external data.
    candidate.validation_data = None
    cv = candidate.cross_validate(
        data, cv=CVConfig(3), inner_cv=CVConfig(2, folds="random"),
        show_progress_bar=False,
    )
    assert cv.exposure is None
    assert cv.actual.equals(data.target)
    assert cv.predictions["gbm"].is_finite().all()


@pytest.mark.parametrize("weighted", [False, True])
def test_rate_stacking_fits_supplied_targets_and_only_observation_weights(monkeypatch, weighted):
    data = ModelData(
        features=pl.DataFrame({"x": [0., 1., 2., 3.]}),
        feature_names=["x"], target=pl.Series([0., .2, .7, .4]),
        weight=pl.Series([1., 2., 3., 4.]) if weighted else None,
    )
    predictions = np.array([[.1, .3], [.2, .5], [.6, .7], [.3, .4]])
    calls = []
    original = PoissonRegressor.fit

    def record(self, x, y, **kwargs):
        calls.append((x.copy(), y.copy(), kwargs.copy()))
        return original(self, x, y, **kwargs)

    monkeypatch.setattr(PoissonRegressor, "fit", record)
    pipeline = SimpleNamespace(fitted_model=SimpleNamespace(objective="poisson"))
    StackingEnsemble().fit_from_predictions([pipeline], predictions, data)
    x, y, kwargs = calls[0]
    np.testing.assert_allclose(x, np.log(predictions))
    np.testing.assert_array_equal(y, data.target.to_numpy())
    if weighted:
        np.testing.assert_array_equal(kwargs["sample_weight"], data.weight.to_numpy())
    else:
        assert "sample_weight" not in kwargs
    assert data.exposure is None
