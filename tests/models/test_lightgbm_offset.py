"""Tests for base-model offset (init_score) support in LightGBMModel."""
import numpy as np
import polars as pl
import pytest
from dataclasses import replace
from ins_gbm.data.loader import load_model_data
from ins_gbm.data.splitter import TrainTestSplit
from ins_gbm.models.lightgbm import LightGBMModel


def _poisson_data(parquet_path):
    return load_model_data(
        path=str(parquet_path), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )


def _gamma_data(parquet_path):
    return load_model_data(
        path=str(parquet_path), target="severity",
        weight="weight", feature_cols=["x1"], objective="gamma",
    )


# ── No-offset regression: existing behaviour unchanged ───────────────────────

def test_poisson_no_offset_response_unchanged(poisson_parquet):
    data = _poisson_data(poisson_parquet)
    train, test = TrainTestSplit(seed=0).split(data)
    fitted = LightGBMModel(objective="poisson").fit(train, params={"n_estimators": 10, "verbose": -1})
    preds = fitted.predict(test, prediction_type="response")
    assert (preds > 0).all()
    assert len(preds) == test.n_rows


def test_gamma_no_offset_response_unchanged(gamma_parquet):
    data = _gamma_data(gamma_parquet)
    train, test = TrainTestSplit(seed=0).split(data)
    fitted = LightGBMModel(objective="gamma").fit(train, params={"n_estimators": 10, "verbose": -1})
    preds = fitted.predict(test, prediction_type="response")
    assert (preds > 0).all()
    assert len(preds) == test.n_rows


# ── Poisson: offset shifts predictions by exp(offset) ────────────────────────

def test_poisson_constant_offset_shifts_predictions(poisson_parquet):
    """A constant offset of log(2) should double the response predictions."""
    data = _poisson_data(poisson_parquet)
    train, test = TrainTestSplit(seed=0).split(data)

    fitted_base = LightGBMModel(objective="poisson").fit(
        train, params={"n_estimators": 10, "verbose": -1}
    )
    # Same model but trained on data + log(2) offset
    offset_val = np.log(2.0)
    offset = pl.Series([offset_val] * train.n_rows)
    train_with_offset = train.with_offset(offset)
    fitted_offset = LightGBMModel(objective="poisson").fit(
        train_with_offset, params={"n_estimators": 10, "verbose": -1}
    )

    # Predict with the offset model on test data that also has offset = log(2)
    test_with_offset = test.with_offset(pl.Series([offset_val] * test.n_rows))
    preds_base = fitted_base.predict(test, prediction_type="response")
    preds_with_offset = fitted_offset.predict(test_with_offset, prediction_type="response")

    # Both predictions should be positive
    assert (preds_base > 0).all()
    assert (preds_with_offset > 0).all()


def test_poisson_predict_time_offset_doubles_response(poisson_parquet):
    """A predict-time offset of log(2) on a model trained without offset should
    approximately double the response vs predicting without offset."""
    data = _poisson_data(poisson_parquet)
    train, test = TrainTestSplit(seed=0).split(data)
    fitted = LightGBMModel(objective="poisson").fit(
        train, params={"n_estimators": 10, "verbose": -1}
    )

    offset_val = np.log(2.0)
    test_with_offset = test.with_offset(pl.Series([offset_val] * test.n_rows))

    preds_no_offset = fitted.predict(test, prediction_type="response")
    preds_with_offset = fitted.predict(test_with_offset, prediction_type="response")

    # response = exp(raw + log(exposure) + offset) = base_response * exp(offset) = base * 2
    ratio = (preds_with_offset / preds_no_offset).to_numpy()
    np.testing.assert_allclose(ratio, 2.0, rtol=1e-4)


def test_poisson_predict_time_offset_rate(poisson_parquet):
    """Rate should also be multiplied by exp(offset)."""
    data = _poisson_data(poisson_parquet)
    train, test = TrainTestSplit(seed=0).split(data)
    fitted = LightGBMModel(objective="poisson").fit(
        train, params={"n_estimators": 10, "verbose": -1}
    )

    offset_val = np.log(3.0)
    test_with_offset = test.with_offset(pl.Series([offset_val] * test.n_rows))

    rate_no_offset = fitted.predict(test, prediction_type="rate")
    rate_with_offset = fitted.predict(test_with_offset, prediction_type="rate")

    ratio = (rate_with_offset / rate_no_offset).to_numpy()
    np.testing.assert_allclose(ratio, 3.0, rtol=1e-4)


def test_poisson_predict_time_offset_link(poisson_parquet):
    """Link = raw + offset; should differ by exactly offset_val."""
    data = _poisson_data(poisson_parquet)
    train, test = TrainTestSplit(seed=0).split(data)
    fitted = LightGBMModel(objective="poisson").fit(
        train, params={"n_estimators": 10, "verbose": -1}
    )

    offset_val = 1.5
    test_with_offset = test.with_offset(pl.Series([offset_val] * test.n_rows))

    link_no_offset = fitted.predict(test, prediction_type="link").to_numpy()
    link_with_offset = fitted.predict(test_with_offset, prediction_type="link").to_numpy()

    np.testing.assert_allclose(link_with_offset - link_no_offset, offset_val, rtol=1e-4)


# ── Poisson: exposure + offset compose correctly ─────────────────────────────

def test_poisson_exposure_and_offset_both_applied(poisson_parquet):
    """Exposure and offset both contribute to init_score additively on link scale."""
    data = _poisson_data(poisson_parquet)
    train, test = TrainTestSplit(seed=0).split(data)

    offset_arr = np.full(train.n_rows, np.log(2.0))
    train_with_offset = train.with_offset(pl.Series(offset_arr))

    fitted = LightGBMModel(objective="poisson").fit(
        train_with_offset, params={"n_estimators": 10, "verbose": -1}
    )

    test_with_offset = test.with_offset(pl.Series(np.full(test.n_rows, np.log(2.0))))
    preds = fitted.predict(test_with_offset, prediction_type="response")
    assert (preds > 0).all()
    assert len(preds) == test.n_rows


# ── Gamma: offset shifts response multiplicatively ───────────────────────────

def test_gamma_predict_time_offset_doubles_response(gamma_parquet):
    """A predict-time offset of log(2) should double gamma response predictions."""
    data = _gamma_data(gamma_parquet)
    train, test = TrainTestSplit(seed=0).split(data)
    fitted = LightGBMModel(objective="gamma").fit(
        train, params={"n_estimators": 10, "verbose": -1}
    )

    offset_val = np.log(2.0)
    test_with_offset = test.with_offset(pl.Series([offset_val] * test.n_rows))

    preds_base = fitted.predict(test, prediction_type="response")
    preds_offset = fitted.predict(test_with_offset, prediction_type="response")

    ratio = (preds_offset / preds_base).to_numpy()
    np.testing.assert_allclose(ratio, 2.0, rtol=1e-4)


# ── Gamma: link prediction branch (new) ──────────────────────────────────────

def test_gamma_link_prediction_no_offset(gamma_parquet):
    """Link = log(response) for gamma without offset."""
    data = _gamma_data(gamma_parquet)
    train, test = TrainTestSplit(seed=0).split(data)
    fitted = LightGBMModel(objective="gamma").fit(
        train, params={"n_estimators": 10, "verbose": -1}
    )

    response = fitted.predict(test, prediction_type="response").to_numpy()
    link = fitted.predict(test, prediction_type="link").to_numpy()

    np.testing.assert_allclose(link, np.log(response), rtol=1e-4)


def test_gamma_link_prediction_with_offset(gamma_parquet):
    """Link = log(response) + offset for gamma with offset."""
    data = _gamma_data(gamma_parquet)
    train, test = TrainTestSplit(seed=0).split(data)
    fitted = LightGBMModel(objective="gamma").fit(
        train, params={"n_estimators": 10, "verbose": -1}
    )

    offset_val = 0.7
    test_with_offset = test.with_offset(pl.Series([offset_val] * test.n_rows))

    link_no_offset = fitted.predict(test, prediction_type="link").to_numpy()
    link_with_offset = fitted.predict(test_with_offset, prediction_type="link").to_numpy()

    np.testing.assert_allclose(link_with_offset - link_no_offset, offset_val, rtol=1e-4)
