import pytest
from ins_gbm.data.loader import load_model_data


def test_load_poisson_from_parquet(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x2", "x3"],
        objective="poisson",
    )
    assert data.n_rows == 400
    assert data.feature_names == ["x1", "x2", "x3"]
    assert data.exposure is not None
    assert data.objective == "poisson"
    assert data.schema is not None
    assert set(data.schema.numeric) == {"x1", "x3"}
    assert data.schema.categorical == ["x2"]


def test_load_gamma_from_parquet(gamma_parquet):
    data = load_model_data(
        path=str(gamma_parquet),
        target="severity",
        weight="weight",
        feature_cols=["x1", "x2"],
        objective="gamma",
    )
    assert data.n_rows == 300
    assert data.weight is not None
    assert data.exposure is None


def test_load_infers_feature_cols(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        objective="poisson",
    )
    assert "claim_count" not in data.feature_names
    assert "exposure" not in data.feature_names
    assert "x1" in data.feature_names
