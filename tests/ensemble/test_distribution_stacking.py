from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import GammaRegressor, PoissonRegressor, Ridge

from ins_gbm import CVConfig, EnsemblePipeline, ModelData, ModelRecipe
from ins_gbm.ensemble.stacking import StackingEnsemble, _stacking_inputs
from ins_gbm.models.lightgbm import LightGBMModel


def make_pipeline(objective="poisson", weighted=True, offset=True):
    n = 48
    data = ModelData(
        features=pl.DataFrame({"x": np.linspace(-1, 1, n)}),
        target=pl.Series(np.array([0., 1., 2., 4.] * 12) + (1 if objective == "gamma" else 0)),
        feature_names=["x"],
        exposure=pl.Series(np.linspace(.3, 2, n)),
        weight=pl.Series(np.linspace(.5, 3, n)) if weighted else None,
        offset=pl.Series(np.linspace(-.4, .4, n)) if offset else None,
        cv_fold=pl.Series(np.arange(n) % 3),
        objective=objective,
    )
    pipeline = ModelRecipe(
        model=LightGBMModel(objective=objective),
        params={"n_estimators": 5, "min_data_in_leaf": 2, "num_threads": 1},
    ).fit(data)
    return data, pipeline


@pytest.mark.parametrize("objective", ["poisson", "gamma"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("offset", [False, True])
def test_distribution_stacking_matches_reference(objective, weighted, offset):
    data, pipeline = make_pipeline(objective, weighted, offset)
    ensemble = StackingEnsemble(cv=CVConfig(folds="predefined"), meta_alpha=.2).fit([pipeline])
    estimator = PoissonRegressor if objective == "poisson" else GammaRegressor
    assert isinstance(ensemble.meta_learner, estimator)
    baseline = np.ones(data.n_rows)
    if offset:
        baseline *= np.exp(data.offset.to_numpy().astype(float))
    if objective == "poisson":
        baseline *= data.exposure.to_numpy().astype(float)
    # Independent construction of the exposure/offset-adjusted reference fit.
    features = np.log(ensemble.oof_predictions.astype(float) / baseline[:, None])
    target = data.target.to_numpy().astype(float) / baseline
    weights = data.weight.to_numpy().astype(float) if weighted else None
    if objective == "poisson":
        weights = baseline if weights is None else baseline * weights
    reference = estimator(alpha=.2, max_iter=1000).fit(features, target, sample_weight=weights)
    np.testing.assert_allclose(ensemble.meta_learner.coef_, reference.coef_, atol=1e-9)
    np.testing.assert_allclose(ensemble.meta_learner.intercept_, reference.intercept_, atol=1e-9)
    base_response = pipeline.predict(data).to_numpy().astype(float)
    expected = baseline * reference.predict(np.log(base_response[:, None] / baseline[:, None]))
    np.testing.assert_allclose(ensemble.predict(data), expected, rtol=2e-6)
    assert np.isfinite(ensemble.predict(data).to_numpy()).all()
    assert (ensemble.predict(data) > 0).all()


@pytest.mark.parametrize("objective", ["poisson", "gamma"])
def test_scoring_exposure_and_offset_restored_exactly_once(objective):
    data, pipeline = make_pipeline(objective)
    result = EnsemblePipeline([pipeline], method="stacking").run()
    original = result.predict(data).to_numpy()
    exposure_changed = replace(data, exposure=data.exposure * 3)
    counts_or_severity = result.predict(exposure_changed).to_numpy()
    np.testing.assert_allclose(counts_or_severity, original * (3 if objective == "poisson" else 1), rtol=2e-6)
    if objective == "poisson":
        np.testing.assert_allclose(
            counts_or_severity / exposure_changed.exposure.to_numpy(),
            original / data.exposure.to_numpy(), rtol=2e-6,
        )
    both_changed = replace(exposure_changed, offset=data.offset + np.log(2))
    np.testing.assert_allclose(
        result.predict(both_changed), original * (6 if objective == "poisson" else 2), rtol=2e-6,
    )


def test_gamma_does_not_use_exposure_for_fit():
    data, pipeline = make_pipeline("gamma")
    first = StackingEnsemble().fit([pipeline])
    changed = replace(data, exposure=data.exposure * 13)
    second = StackingEnsemble().fit([replace(pipeline, raw_train_data=changed)])
    np.testing.assert_allclose(first.meta_learner.coef_, second.meta_learner.coef_, atol=1e-12)
    np.testing.assert_allclose(first.predict(data), second.predict(data), rtol=1e-6)


def test_poisson_without_exposure_or_offset():
    data, pipeline = make_pipeline()
    data = replace(data, exposure=None, offset=None)
    pipeline = ModelRecipe(model=LightGBMModel(), params={"n_estimators": 3, "num_threads": 1}).fit(data)
    result = EnsemblePipeline([pipeline], method="stacking").run()
    assert np.isfinite(result.predict(data).to_numpy()).all()


def test_ridge_preserves_response_features_and_passes_observation_weights():
    data, pipeline = make_pipeline()
    result = StackingEnsemble(meta_learner="ridge", meta_alpha=.3).fit([pipeline])
    reference = Ridge(alpha=.3).fit(
        result.oof_predictions.astype(float), data.target.to_numpy().astype(float),
        sample_weight=data.weight.to_numpy().astype(float),
    )
    assert not result.distribution_aware
    np.testing.assert_allclose(result.meta_learner.coef_, reference.coef_)
    expected = np.maximum(reference.predict(pipeline.predict(data).to_numpy()[:, None]), 1e-9)
    np.testing.assert_allclose(result.predict(data), expected)


def test_custom_learner_is_independent_and_weighted():
    data, pipeline = make_pipeline()
    supplied = Ridge(alpha=.7)
    result = StackingEnsemble(meta_learner=supplied).fit([pipeline])
    assert not hasattr(supplied, "coef_")
    assert result.meta_learner.alpha == .7


def test_custom_learner_without_weight_support_rejected():
    class Unweighted:
        def fit(self, x, y):
            return self
        def predict(self, x):
            return np.ones(len(x))
    _, pipeline = make_pipeline()
    with pytest.raises(ValueError, match="accept sample_weight"):
        StackingEnsemble(meta_learner=Unweighted()).fit([pipeline])


@pytest.mark.parametrize("prediction", [0., -1., np.nan, np.inf])
def test_bad_distribution_features_rejected(prediction):
    data, _ = make_pipeline()
    with pytest.raises(ValueError, match="finite positive"):
        _stacking_inputs(np.full((data.n_rows, 1), prediction), data, "poisson")


def test_invalid_baseline_rejected():
    data, _ = make_pipeline()
    data = replace(data, offset=pl.Series([1000.] * data.n_rows))
    with pytest.raises(ValueError, match="baseline"):
        _stacking_inputs(np.ones((data.n_rows, 1)), data, "poisson")


@pytest.mark.parametrize("alpha", [-1., np.inf, np.nan])
def test_bad_meta_alpha_rejected(alpha):
    _, pipeline = make_pipeline()
    with pytest.raises(ValueError, match="meta_alpha"):
        StackingEnsemble(meta_alpha=alpha).fit([pipeline])
