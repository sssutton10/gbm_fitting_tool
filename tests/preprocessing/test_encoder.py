import polars as pl
import pytest
from gbm_fitting.data.schema import FeatureSchema
from gbm_fitting.preprocessing.encoder import OneHotEncoder


def _df():
    return pl.DataFrame({
        "num": [1.0, 2.0, 3.0],
        "cat": ["A", "B", "A"],
    })


def _schema():
    return FeatureSchema(numeric=["num"], categorical=["cat"])


def test_fit_and_transform_basic():
    df = _df()
    encoder = OneHotEncoder()
    fitted = encoder.fit(df, _schema())
    out = fitted.transform(df)
    assert "num" in out.columns
    assert "cat__A" in out.columns
    assert "cat__B" in out.columns
    assert "cat" not in out.columns


def test_output_feature_names_stable():
    df = _df()
    fitted = OneHotEncoder().fit(df, _schema())
    assert fitted.output_feature_names() == fitted.output_feature_names()


def test_unknown_category_produces_zero_row():
    df = _df()
    fitted = OneHotEncoder().fit(df, _schema())
    unseen = pl.DataFrame({"num": [9.0], "cat": ["Z"]})
    out = fitted.transform(unseen)
    assert out["cat__A"][0] == 0
    assert out["cat__B"][0] == 0


def test_missing_category_treated_as_explicit_level():
    df = pl.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["A", None, "B"]})
    schema = FeatureSchema(numeric=["num"], categorical=["cat"])
    fitted = OneHotEncoder().fit(df, schema)
    names = fitted.output_feature_names()
    assert any("__null" in n or "__missing" in n or "null" in n.lower() for n in names)


def test_numeric_passthrough():
    df = _df()
    fitted = OneHotEncoder().fit(df, _schema())
    out = fitted.transform(df)
    assert (out["num"].to_list()) == [1.0, 2.0, 3.0]


def test_column_order_stable_across_calls():
    df = _df()
    fitted = OneHotEncoder().fit(df, _schema())
    out1 = fitted.transform(df)
    out2 = fitted.transform(df)
    assert out1.columns == out2.columns
