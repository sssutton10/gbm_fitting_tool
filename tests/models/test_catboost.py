import numpy as np
import polars as pl
import pytest

from ins_gbm.data.loader import load_model_data
from ins_gbm.data.model_data import ModelData
from ins_gbm.models.catboost import CatBoostModel
from ins_gbm.preprocessing.encoder import OneHotEncoder

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


def _categorical_data(features: pl.DataFrame) -> ModelData:
    return ModelData(
        features=features,
        target=pl.Series("claims", [0.0, 2.0, 0.0, 1.0] * (features.height // 4)),
        feature_names=list(features.columns),
        objective="poisson",
    )


def test_catboost_uses_native_categories_in_train_and_prediction_pools(monkeypatch):
    from catboost import Pool

    features = pl.DataFrame(
        {
            "territory": ["north", "south", None, "north"] * 5,
            "driver_age": [20, 45, 32, 57] * 5,
        }
    )
    data = _categorical_data(features)
    captured = []
    original_init = Pool.__init__

    def recording_init(self, *args, **kwargs):
        captured.append(kwargs.copy())
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(Pool, "__init__", recording_init)
    fitted = CatBoostModel().fit(data, params={"iterations": 5})
    score = ModelData(
        features=pl.DataFrame({"territory": ["west", None], "driver_age": [30, 40]}),
        target=pl.Series("claims", [0.0, 0.0]),
        feature_names=data.feature_names,
        objective="poisson",
    )
    predictions = fitted.predict(score)

    assert [kwargs["cat_features"] for kwargs in captured] == [
        ["territory"],
        ["territory"],
    ]
    assert captured[0]["data"][2, 0] == "-999999999"
    assert captured[1]["data"][0, 0] == "west"
    assert np.isfinite(predictions.to_numpy()).all()


def test_catboost_allows_explicit_numeric_categorical_features(monkeypatch):
    from catboost import Pool

    features = pl.DataFrame(
        {"territory_code": [10, 20, 10, 20] * 5, "driver_age": [20, 45, 32, 57] * 5}
    )
    data = _categorical_data(features)
    captured = []
    original_init = Pool.__init__

    def recording_init(self, *args, **kwargs):
        captured.append(kwargs.copy())
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(Pool, "__init__", recording_init)
    fitted = CatBoostModel(categorical_features=["territory_code"]).fit(
        data, params={"iterations": 5}
    )
    fitted.predict(data)

    assert [kwargs["cat_features"] for kwargs in captured] == [
        ["territory_code"],
        ["territory_code"],
    ]
    assert captured[0]["data"][0, 0] == "10"
    assert isinstance(captured[0]["data"][0, 1], float)


def test_catboost_auto_detects_categorical_enum_and_boolean_columns():
    features = pl.DataFrame(
        {
            "territory": pl.Series(["north", "south"] * 10).cast(pl.Categorical),
            "vehicle": pl.Series(["car", "truck"] * 10).cast(pl.Enum(["car", "truck"])),
            "new_business": [True, False] * 10,
        }
    )
    fitted = CatBoostModel().fit(_categorical_data(features), params={"iterations": 5})

    assert fitted.model.get_cat_feature_indices() == [0, 1, 2]


def test_catboost_one_hot_encoder_removes_native_categories():
    data = _categorical_data(pl.DataFrame({"territory": ["north", "south"] * 10}))
    fitted = CatBoostModel().fit(
        data, encoder=OneHotEncoder(), params={"iterations": 5}
    )
    assert fitted.feature_names == ["territory__north", "territory__south"]


def test_catboost_rejects_categorical_features_in_params():
    data = _categorical_data(pl.DataFrame({"territory": ["north", "south"] * 2}))
    with pytest.raises(ValueError, match="categorical_features"):
        CatBoostModel().fit(data, params={"cat_features": ["territory"]})
