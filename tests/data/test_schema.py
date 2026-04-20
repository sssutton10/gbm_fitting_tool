import polars as pl
from ins_gbm.data.schema import FeatureSchema, infer_schema


def test_feature_schema_defaults():
    s = FeatureSchema(numeric=["x1"], categorical=["x2"])
    assert s.numeric == ["x1"]
    assert s.categorical == ["x2"]
    assert s.ordinal == []
    assert s.passthrough == []


def test_feature_schema_all_features():
    s = FeatureSchema(numeric=["x1"], categorical=["x2"], ordinal=["x3"], passthrough=["id"])
    assert set(s.all_features()) == {"x1", "x2", "x3", "id"}


def test_infer_schema_numeric_and_categorical():
    df = pl.DataFrame({
        "num1": [1.0, 2.0],
        "int1": [1, 2],
        "cat1": ["a", "b"],
        "bool1": [True, False],
    })
    s = infer_schema(df, feature_cols=["num1", "int1", "cat1", "bool1"])
    assert set(s.numeric) == {"num1", "int1"}
    assert set(s.categorical) == {"cat1", "bool1"}


def test_infer_schema_unsupported_dtype():
    import pytest
    df = pl.DataFrame({"dt": [1, 2]}).with_columns(pl.col("dt").cast(pl.Date))
    with pytest.raises(ValueError, match="Unsupported dtype"):
        infer_schema(df, feature_cols=["dt"])
