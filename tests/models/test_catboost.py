import polars as pl
import pytest

from ins_gbm.data.loader import load_model_data
from ins_gbm.models.catboost import CatBoostModel

pytest.importorskip("catboost")


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


def test_catboost_poisson_fit_predict(poisson_parquet):
    """Verify catboost poisson fit predict.

    Args:
        poisson_parquet (object): The poisson parquet.
    """
    data = _poisson(poisson_parquet)
    train = test = data
    fitted = CatBoostModel(objective="poisson").fit(train, params={"iterations": 10})
    preds = fitted.predict(test, prediction_type="response")
    assert isinstance(preds, pl.Series)
    assert len(preds) == test.n_rows
    assert (preds > 0).all()


def test_catboost_gamma_fit_predict(gamma_parquet):
    """Verify catboost gamma fit predict.

    Args:
        gamma_parquet (object): The gamma parquet.
    """
    data = _gamma(gamma_parquet)
    train = test = data
    fitted = CatBoostModel(objective="gamma").fit(train, params={"iterations": 10})
    preds = fitted.predict(test, prediction_type="response")
    assert (preds > 0).all()


def test_catboost_uses_model_data_objective_when_omitted(gamma_parquet):
    """Verify catboost uses model data objective when omitted.

    Args:
        gamma_parquet (object): The gamma parquet.
    """
    data = _gamma(gamma_parquet)

    fitted = CatBoostModel().fit(data, params={"iterations": 5})

    assert fitted.objective == "gamma"


def test_catboost_gamma_rejects_rate(gamma_parquet):
    """Verify catboost gamma rejects rate.

    Args:
        gamma_parquet (object): The gamma parquet.
    """
    data = _gamma(gamma_parquet)
    train = test = data
    fitted = CatBoostModel(objective="gamma").fit(train, params={"iterations": 10})
    with pytest.raises(ValueError, match="(?i)rate.*gamma"):
        fitted.predict(test, prediction_type="rate")


def test_catboost_feature_importance(poisson_parquet):
    """Verify catboost feature importance.

    Args:
        poisson_parquet (object): The poisson parquet.
    """
    data = _poisson(poisson_parquet)
    fitted = CatBoostModel(objective="poisson").fit(data, params={"iterations": 10})
    imp = fitted.feature_importance()
    assert "feature" in imp.columns
    assert "importance" in imp.columns
    assert len(imp) == len(data.feature_names)


def test_catboost_capabilities():
    """Verify catboost capabilities."""
    caps = CatBoostModel(objective="poisson").capabilities()
    assert caps.supports_poisson
    # offset support depends on installed version — just check it's declared
    assert isinstance(caps.supports_offset, bool)


def test_catboost_search_space_keys():
    """Verify catboost search space keys."""
    space = CatBoostModel(objective="poisson").default_search_space()
    assert "iterations" in space
    assert "learning_rate" in space
    assert "depth" in space
