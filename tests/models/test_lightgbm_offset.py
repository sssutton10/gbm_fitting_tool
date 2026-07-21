"""Tests for base-model offset (init_score) support in LightGBMModel."""
import numpy as np
import polars as pl

from ins_gbm.data.loader import load_model_data
from ins_gbm.models.lightgbm import LightGBMModel


def _poisson_data(parquet_path):
    return load_model_data(
        path=str(parquet_path), target="claim_count", exposure="exposure",
        feature_cols=["x1", "x3"], objective="poisson",
    )


def _gamma_data(parquet_path):
    return load_model_data(
        path=str(parquet_path), target="severity", weight="weight",
        feature_cols=["x1"], objective="gamma",
    )


def test_poisson_predict_time_offset_doubles_response(poisson_parquet):
    data = _poisson_data(poisson_parquet)
    fitted = LightGBMModel(objective="poisson").fit(data, params={"n_estimators": 10, "verbose": -1})
    offset = data.with_offset(pl.Series([np.log(2.0)] * data.n_rows))
    ratio = (fitted.predict(offset, "response") / fitted.predict(data, "response")).to_numpy()
    np.testing.assert_allclose(ratio, 2.0, rtol=1e-4)


def test_poisson_predict_time_offset_rate(poisson_parquet):
    data = _poisson_data(poisson_parquet)
    fitted = LightGBMModel(objective="poisson").fit(data, params={"n_estimators": 10, "verbose": -1})
    offset = data.with_offset(pl.Series([np.log(3.0)] * data.n_rows))
    ratio = (fitted.predict(offset, "rate") / fitted.predict(data, "rate")).to_numpy()
    np.testing.assert_allclose(ratio, 3.0, rtol=1e-4)


def test_poisson_predict_time_offset_link(poisson_parquet):
    data = _poisson_data(poisson_parquet)
    fitted = LightGBMModel(objective="poisson").fit(data, params={"n_estimators": 10, "verbose": -1})
    offset = data.with_offset(pl.Series([1.5] * data.n_rows))
    delta = fitted.predict(offset, "link").to_numpy() - fitted.predict(data, "link").to_numpy()
    np.testing.assert_allclose(delta, 1.5, rtol=1e-4)


def test_gamma_predict_time_offset_doubles_response(gamma_parquet):
    data = _gamma_data(gamma_parquet)
    fitted = LightGBMModel(objective="gamma").fit(data, params={"n_estimators": 10, "verbose": -1})
    offset = data.with_offset(pl.Series([np.log(2.0)] * data.n_rows))
    ratio = (fitted.predict(offset, "response") / fitted.predict(data, "response")).to_numpy()
    np.testing.assert_allclose(ratio, 2.0, rtol=1e-4)


def test_gamma_link_prediction_no_offset(gamma_parquet):
    data = _gamma_data(gamma_parquet)
    fitted = LightGBMModel(objective="gamma").fit(data, params={"n_estimators": 10, "verbose": -1})
    response = fitted.predict(data, "response").to_numpy()
    np.testing.assert_allclose(fitted.predict(data, "link").to_numpy(), np.log(response), rtol=1e-4)
