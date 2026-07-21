import numpy as np
import polars as pl
import pytest
from ins_gbm.data.loader import load_model_data
from ins_gbm.models.lightgbm import LightGBMModel


def _poisson_data(poisson_parquet):
    return load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
    )


def _gamma_data(gamma_parquet):
    return load_model_data(
        path=str(gamma_parquet),
        target="severity",
        weight="weight",
        feature_cols=["x1"],
        objective="gamma",
    )


def test_lgb_poisson_fit_predict(poisson_parquet):
    data = _poisson_data(poisson_parquet)
    train = test = data
    model = LightGBMModel(objective="poisson")
    fitted = model.fit(train, params={"n_estimators": 10, "verbose": -1})
    preds = fitted.predict(test, prediction_type="response")
    assert isinstance(preds, pl.Series)
    assert len(preds) == test.n_rows
    assert (preds > 0).all()


def test_lgb_poisson_rate_prediction(poisson_parquet):
    data = _poisson_data(poisson_parquet)
    train = test = data
    fitted = LightGBMModel(objective="poisson").fit(train, params={"n_estimators": 10, "verbose": -1})
    rate = fitted.predict(test, prediction_type="rate")
    response = fitted.predict(test, prediction_type="response")
    # rate = response / exposure
    expected = response / test.exposure
    np.testing.assert_allclose(rate.to_numpy(), expected.to_numpy(), rtol=1e-5)


def test_lgb_gamma_fit_predict(gamma_parquet):
    data = _gamma_data(gamma_parquet)
    train = test = data
    fitted = LightGBMModel(objective="gamma").fit(train, params={"n_estimators": 10, "verbose": -1})
    preds = fitted.predict(test, prediction_type="response")
    assert (preds > 0).all()
    assert len(preds) == test.n_rows


def test_lgb_gamma_rejects_rate(gamma_parquet):
    data = _gamma_data(gamma_parquet)
    train = test = data
    fitted = LightGBMModel(objective="gamma").fit(train, params={"n_estimators": 10, "verbose": -1})
    with pytest.raises(ValueError, match="(?i)rate.*gamma"):
        fitted.predict(test, prediction_type="rate")


def test_lgb_feature_importance(poisson_parquet):
    data = _poisson_data(poisson_parquet)
    fitted = LightGBMModel(objective="poisson").fit(data, params={"n_estimators": 10, "verbose": -1})
    imp = fitted.feature_importance()
    assert "feature" in imp.columns
    assert "importance" in imp.columns
    assert len(imp) == len(data.feature_names)


def test_lgb_feature_importance_accepts_native_type(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count", exposure="exposure",
        feature_cols=["x1", "x3"], objective="poisson",
    )
    fitted = LightGBMModel(objective="poisson").fit(
        data, params={"n_estimators": 5, "verbose": -1}
    )
    assert fitted.feature_importance("split").height == 2
    with pytest.raises(ValueError, match="importance_type"):
        fitted.feature_importance("not-a-lightgbm-importance")


def test_lgb_capabilities():
    caps = LightGBMModel(objective="poisson").capabilities()
    assert caps.supports_poisson
    assert caps.supports_gamma
    assert caps.supports_offset


def test_lgb_search_space_keys():
    space = LightGBMModel(objective="poisson").default_search_space()
    assert "n_estimators" in space
    assert "learning_rate" in space
    assert "num_leaves" in space
