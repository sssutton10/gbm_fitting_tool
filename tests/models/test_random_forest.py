from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from ins_gbm.data.loader import load_model_data
from ins_gbm.models.random_forest import RandomForestModel


def _poisson(poisson_parquet):
    """Poisson.

    Args:
        poisson_parquet (object): The poisson parquet.
    """
    return load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
    )


def _gamma(gamma_parquet):
    """Gamma.

    Args:
        gamma_parquet (object): The gamma parquet.
    """
    return load_model_data(
        path=str(gamma_parquet),
        target="severity",
        weight="weight",
        feature_cols=["x1"],
        objective="gamma",
    )


def test_rf_poisson_fit_predict(poisson_parquet):
    """Verify rf poisson fit predict.

    Args:
        poisson_parquet (object): The poisson parquet.
    """
    data = _poisson(poisson_parquet)
    train = test = data
    fitted = RandomForestModel(objective="poisson").fit(
        train, params={"n_estimators": 10}
    )
    preds = fitted.predict(test, prediction_type="response")
    assert isinstance(preds, pl.Series)
    assert len(preds) == test.n_rows
    assert (preds > 0).all()


def test_rf_poisson_response_floors_zero_rate_predictions(poisson_parquet):
    """Verify rf poisson response floors zero rate predictions.

    Args:
        poisson_parquet (object): The poisson parquet.
    """
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
        zero_target_data, params={"n_estimators": 10, "criterion": "squared_error"}
    )
    preds = fitted.predict(zero_target_data, prediction_type="response")

    assert (preds == 1e-10).all()


def test_rf_poisson_criterion_rejects_all_zero_targets(poisson_parquet):
    data = _poisson(poisson_parquet)
    data = replace(data, target=pl.Series("claim_count", np.zeros(data.n_rows)))

    with pytest.raises(ValueError, match="(?i)sum of y is not strictly positive"):
        RandomForestModel().fit(data, params={"n_estimators": 1})


@pytest.mark.parametrize("objective, leaf_size", [("poisson", 10), ("gamma", 5)])
@pytest.mark.parametrize("explicit_objective", [False, True])
def test_rf_objective_defaults(
    poisson_parquet, gamma_parquet, objective, leaf_size, explicit_objective
):
    data = _poisson(poisson_parquet) if objective == "poisson" else _gamma(gamma_parquet)
    model = RandomForestModel(objective=objective if explicit_objective else None)
    if explicit_objective:
        data = replace(data, objective=None)

    fitted = model.fit(data)

    expected = {
        "criterion": "poisson" if objective == "poisson" else "squared_error",
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": leaf_size,
        "max_features": 0.8,
        "n_jobs": 1,
        "random_state": 42,
    }
    assert fitted.objective == objective
    assert fitted.params == expected
    assert all(fitted.model.get_params()[key] == value for key, value in expected.items())


def test_rf_explicit_params_override_defaults_without_mutation(poisson_parquet):
    params = {
        "criterion": "squared_error",
        "n_estimators": 3,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "n_jobs": 2,
        "random_state": 7,
        "bootstrap": False,
    }
    original = params.copy()

    fitted = RandomForestModel().fit(_poisson(poisson_parquet), params=params)

    assert params == original
    assert fitted.params == original
    assert all(fitted.model.get_params()[key] == value for key, value in original.items())


def test_rf_poisson_combines_exposure_and_model_weight(poisson_parquet, monkeypatch):
    """Verify rf poisson combines exposure and model weight.

    Args:
        poisson_parquet (object): The poisson parquet.
        monkeypatch (object): The monkeypatch.
    """
    data = _poisson(poisson_parquet)
    model_weight = pl.Series("model_weight", [2.0] * data.n_rows)
    weighted_data = replace(data, weight=model_weight).validate()
    captured = {}

    from sklearn.ensemble import RandomForestRegressor

    original_fit = RandomForestRegressor.fit

    def recording_fit(self, X, y, sample_weight=None):
        """Recording fit.

        Args:
            X (object): The X.
            y (object): The y.
            sample_weight (object): The sample weight. Optional.
        """
        captured["y"] = np.asarray(y)
        captured["criterion"] = self.criterion
        captured["sample_weight"] = np.asarray(sample_weight)
        return original_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(RandomForestRegressor, "fit", recording_fit)
    RandomForestModel(objective="poisson").fit(
        weighted_data, params={"n_estimators": 1}
    )

    expected = data.exposure.to_numpy() * model_weight.to_numpy()
    assert captured["criterion"] == "poisson"
    np.testing.assert_allclose(
        captured["y"], data.target.to_numpy() / data.exposure.to_numpy()
    )
    np.testing.assert_allclose(captured["sample_weight"], expected)


def test_rf_fit_receives_float32_arrays(poisson_parquet, monkeypatch):
    """Verify rf fit receives float32 arrays.

    Args:
        poisson_parquet (object): The poisson parquet.
        monkeypatch (object): The monkeypatch.
    """
    data = _poisson(poisson_parquet)
    captured = {}

    from sklearn.ensemble import RandomForestRegressor

    original_fit = RandomForestRegressor.fit

    def recording_fit(self, X, y, sample_weight=None):
        """Recording fit.

        Args:
            X (object): The X.
            y (object): The y.
            sample_weight (object): The sample weight. Optional.
        """
        captured["X"] = np.asarray(X)
        captured["y"] = np.asarray(y)
        captured["sample_weight"] = np.asarray(sample_weight)
        return original_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(RandomForestRegressor, "fit", recording_fit)
    RandomForestModel(objective="poisson").fit(data, params={"n_estimators": 1})

    assert captured["X"].dtype == np.float32
    assert captured["y"].dtype == np.float32
    assert captured["sample_weight"].dtype == np.float32


def test_rf_gamma_fit_predict(gamma_parquet):
    """Verify rf gamma fit predict.

    Args:
        gamma_parquet (object): The gamma parquet.
    """
    data = _gamma(gamma_parquet)
    train = test = data
    fitted = RandomForestModel(objective="gamma").fit(
        train, params={"n_estimators": 10}
    )
    preds = fitted.predict(test, prediction_type="response")
    assert (preds > 0).all()


def test_rf_uses_model_data_objective_when_omitted(gamma_parquet):
    """Verify rf uses model data objective when omitted.

    Args:
        gamma_parquet (object): The gamma parquet.
    """
    data = _gamma(gamma_parquet)

    fitted = RandomForestModel().fit(data, params={"n_estimators": 5})

    assert fitted.objective == "gamma"


def test_rf_poisson_without_exposure_passes_no_exposure_weight(
    poisson_parquet, monkeypatch
):
    """Verify rf poisson without exposure passes no exposure weight.

    Args:
        poisson_parquet (object): The poisson parquet.
        monkeypatch (object): The monkeypatch.
    """
    data = replace(_poisson(poisson_parquet), exposure=None).validate()
    captured = {}

    from sklearn.ensemble import RandomForestRegressor

    original_fit = RandomForestRegressor.fit

    def recording_fit(self, X, y, **kwargs):
        """Recording fit.

        Args:
            X (object): The X.
            y (object): The y.
            kwargs (object): Additional keyword arguments forwarded to the pipeline run.
        """
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
    """Verify rf gamma rejects rate.

    Args:
        gamma_parquet (object): The gamma parquet.
    """
    data = _gamma(gamma_parquet)
    train = test = data
    fitted = RandomForestModel(objective="gamma").fit(
        train, params={"n_estimators": 10}
    )
    with pytest.raises(ValueError, match="(?i)rate.*gamma"):
        fitted.predict(test, prediction_type="rate")


def test_rf_capabilities_no_native_offset():
    """Verify rf capabilities no native offset."""
    caps = RandomForestModel(objective="poisson").capabilities()
    assert not caps.supports_offset


def test_rf_feature_importance(poisson_parquet):
    """Verify rf feature importance.

    Args:
        poisson_parquet (object): The poisson parquet.
    """
    data = _poisson(poisson_parquet)
    fitted = RandomForestModel(objective="poisson").fit(
        data, params={"n_estimators": 10}
    )
    imp = fitted.feature_importance()
    assert "feature" in imp.columns
    assert "importance" in imp.columns


def test_rf_search_space_keys():
    """Verify rf search space keys."""
    space = RandomForestModel(objective="poisson").default_search_space()
    assert "n_estimators" in space
    assert "max_depth" in space
    assert "min_samples_leaf" in space
