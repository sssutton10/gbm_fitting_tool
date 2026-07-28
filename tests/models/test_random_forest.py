from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from ins_gbm.data.loader import load_model_data
from ins_gbm.models.random_forest import RandomForestModel


def _poisson(poisson_parquet):
    return load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )


def _gamma(gamma_parquet):
    return load_model_data(
        path=str(gamma_parquet), target="severity",
        weight="weight", feature_cols=["x1"], objective="gamma",
    )


def test_rf_poisson_fit_predict(poisson_parquet):
    data = _poisson(poisson_parquet)
    train = test = data
    fitted = RandomForestModel(objective="poisson").fit(train, params={"n_estimators": 10})
    preds = fitted.predict(test, prediction_type="response")
    assert isinstance(preds, pl.Series)
    assert len(preds) == test.n_rows
    assert (preds > 0).all()


def test_rf_poisson_response_floors_zero_rate_predictions(poisson_parquet):
    data = _poisson(poisson_parquet)
    zero_target_data = data.__class__(
        features=data.features,
        target=pl.Series("claim_count", [0.0] * data.n_rows),
        exposure=data.exposure,
        weight=data.weight,
        feature_names=data.feature_names,
        schema=data.schema,
        objective=data.objective,
    ).validate()

    fitted = RandomForestModel(objective="poisson").fit(
        zero_target_data, params={"n_estimators": 10}
    )
    preds = fitted.predict(zero_target_data, prediction_type="response")

    assert (preds == 1e-10).all()


def test_rf_poisson_combines_exposure_and_model_weight(
    poisson_parquet, monkeypatch
):
    data = _poisson(poisson_parquet)
    model_weight = pl.Series("model_weight", [2.0] * data.n_rows)
    weighted_data = replace(data, weight=model_weight).validate()
    captured = {}

    from sklearn.ensemble import RandomForestRegressor

    original_fit = RandomForestRegressor.fit

    def recording_fit(self, X, y, sample_weight=None):
        captured["sample_weight"] = np.asarray(sample_weight)
        return original_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(RandomForestRegressor, "fit", recording_fit)
    RandomForestModel(objective="poisson").fit(
        weighted_data, params={"n_estimators": 1}
    )

    expected = data.exposure.to_numpy() * model_weight.to_numpy()
    np.testing.assert_allclose(captured["sample_weight"], expected)


def test_rf_fit_receives_float32_arrays(poisson_parquet, monkeypatch):
    data = _poisson(poisson_parquet)
    captured = {}

    from sklearn.ensemble import RandomForestRegressor

    original_fit = RandomForestRegressor.fit

    def recording_fit(self, X, y, sample_weight=None):
        captured["X"] = np.asarray(X)
        captured["y"] = np.asarray(y)
        captured["sample_weight"] = np.asarray(sample_weight)
        return original_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(RandomForestRegressor, "fit", recording_fit)
    RandomForestModel(objective="poisson").fit(
        data, params={"n_estimators": 1}
    )

    assert captured["X"].dtype == np.float32
    assert captured["y"].dtype == np.float32
    assert captured["sample_weight"].dtype == np.float32


def test_rf_gamma_fit_predict(gamma_parquet):
    data = _gamma(gamma_parquet)
    train = test = data
    fitted = RandomForestModel(objective="gamma").fit(train, params={"n_estimators": 10})
    preds = fitted.predict(test, prediction_type="response")
    assert (preds > 0).all()


def test_rf_uses_model_data_objective_when_omitted(gamma_parquet):
    data = _gamma(gamma_parquet)

    fitted = RandomForestModel().fit(data, params={"n_estimators": 5})

    assert fitted.objective == "gamma"


def test_rf_poisson_without_exposure_passes_no_exposure_weight(
    poisson_parquet, monkeypatch
):
    data = replace(_poisson(poisson_parquet), exposure=None).validate()
    captured = {}

    from sklearn.ensemble import RandomForestRegressor

    original_fit = RandomForestRegressor.fit

    def recording_fit(self, X, y, **kwargs):
        captured["kwargs"] = kwargs
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(RandomForestRegressor, "fit", recording_fit)

    fitted = RandomForestModel().fit(data, params={"n_estimators": 5})
    response = fitted.predict(data, prediction_type="response")
    rate = fitted.predict(data, prediction_type="rate")

    assert fitted.objective == "poisson"
    assert "sample_weight" not in captured["kwargs"]
    np.testing.assert_allclose(response.to_numpy(), rate.to_numpy())


def test_rf_gamma_rejects_rate(gamma_parquet):
    data = _gamma(gamma_parquet)
    train = test = data
    fitted = RandomForestModel(objective="gamma").fit(train, params={"n_estimators": 10})
    with pytest.raises(ValueError, match="(?i)rate.*gamma"):
        fitted.predict(test, prediction_type="rate")


def test_rf_capabilities_no_native_offset():
    caps = RandomForestModel(objective="poisson").capabilities()
    assert not caps.supports_offset


def test_rf_feature_importance(poisson_parquet):
    data = _poisson(poisson_parquet)
    fitted = RandomForestModel(objective="poisson").fit(data, params={"n_estimators": 10})
    imp = fitted.feature_importance()
    assert "feature" in imp.columns
    assert "importance" in imp.columns


def test_rf_search_space_keys():
    space = RandomForestModel(objective="poisson").default_search_space()
    assert "n_estimators" in space
    assert "max_depth" in space
    assert "min_samples_leaf" in space
