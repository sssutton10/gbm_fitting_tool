import numpy as np
import polars as pl
import pytest
from ins_gbm.data.loader import load_model_data
from ins_gbm.models.xgboost import XGBoostModel

pytest.importorskip("xgboost")


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


def test_xgb_poisson_fit_predict(poisson_parquet):
    data = _poisson(poisson_parquet)
    train = test = data
    fitted = XGBoostModel(objective="poisson").fit(train, params={"n_estimators": 10})
    preds = fitted.predict(test, prediction_type="response")
    assert isinstance(preds, pl.Series)
    assert len(preds) == test.n_rows
    assert (preds > 0).all()


def test_xgb_poisson_rate_prediction(poisson_parquet):
    data = _poisson(poisson_parquet)
    train = test = data
    fitted = XGBoostModel(objective="poisson").fit(train, params={"n_estimators": 10})
    rate = fitted.predict(test, prediction_type="rate")
    response = fitted.predict(test, prediction_type="response")
    expected = response / test.exposure
    np.testing.assert_allclose(rate.to_numpy(), expected.to_numpy(), rtol=1e-5)


def test_xgb_gamma_fit_predict(gamma_parquet):
    data = _gamma(gamma_parquet)
    train = test = data
    fitted = XGBoostModel(objective="gamma").fit(train, params={"n_estimators": 10})
    preds = fitted.predict(test, prediction_type="response")
    assert (preds > 0).all()


def test_xgb_gamma_rejects_rate(gamma_parquet):
    data = _gamma(gamma_parquet)
    train = test = data
    fitted = XGBoostModel(objective="gamma").fit(train, params={"n_estimators": 10})
    with pytest.raises(ValueError, match="(?i)rate.*gamma"):
        fitted.predict(test, prediction_type="rate")


def test_xgb_feature_importance(poisson_parquet):
    data = _poisson(poisson_parquet)
    fitted = XGBoostModel(objective="poisson").fit(data, params={"n_estimators": 10})
    imp = fitted.feature_importance()
    assert "feature" in imp.columns
    assert "importance" in imp.columns
    assert len(imp) == len(data.feature_names)


def test_xgb_capabilities():
    caps = XGBoostModel(objective="poisson").capabilities()
    assert caps.supports_poisson
    assert caps.supports_gamma
    assert caps.supports_offset


def test_xgb_search_space_keys():
    space = XGBoostModel(objective="poisson").default_search_space()
    assert "n_estimators" in space
    assert "learning_rate" in space
    assert "max_depth" in space
