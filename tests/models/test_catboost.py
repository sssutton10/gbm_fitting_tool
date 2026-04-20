import polars as pl
import pytest
from ins_gbm.data.loader import load_model_data
from ins_gbm.data.splitter import TrainTestSplit
from ins_gbm.models.catboost import CatBoostModel


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


def test_catboost_poisson_fit_predict(poisson_parquet):
    data = _poisson(poisson_parquet)
    train, test = TrainTestSplit().split(data)
    fitted = CatBoostModel(objective="poisson").fit(train, params={"iterations": 10})
    preds = fitted.predict(test, prediction_type="response")
    assert isinstance(preds, pl.Series)
    assert len(preds) == test.n_rows
    assert (preds > 0).all()


def test_catboost_gamma_fit_predict(gamma_parquet):
    data = _gamma(gamma_parquet)
    train, test = TrainTestSplit().split(data)
    fitted = CatBoostModel(objective="gamma").fit(train, params={"iterations": 10})
    preds = fitted.predict(test, prediction_type="response")
    assert (preds > 0).all()


def test_catboost_gamma_rejects_rate(gamma_parquet):
    data = _gamma(gamma_parquet)
    train, test = TrainTestSplit().split(data)
    fitted = CatBoostModel(objective="gamma").fit(train, params={"iterations": 10})
    with pytest.raises(ValueError, match="(?i)rate.*gamma"):
        fitted.predict(test, prediction_type="rate")


def test_catboost_feature_importance(poisson_parquet):
    data = _poisson(poisson_parquet)
    fitted = CatBoostModel(objective="poisson").fit(data, params={"iterations": 10})
    imp = fitted.feature_importance()
    assert "feature" in imp.columns
    assert "importance" in imp.columns
    assert len(imp) == len(data.feature_names)


def test_catboost_capabilities():
    caps = CatBoostModel(objective="poisson").capabilities()
    assert caps.supports_poisson
    # offset support depends on installed version — just check it's declared
    assert isinstance(caps.supports_offset, bool)


def test_catboost_search_space_keys():
    space = CatBoostModel(objective="poisson").default_search_space()
    assert "iterations" in space
    assert "learning_rate" in space
    assert "depth" in space
