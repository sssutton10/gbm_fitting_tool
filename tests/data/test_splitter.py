import pytest
from ins_gbm.data.loader import load_model_data
from ins_gbm.data.splitter import TrainTestSplit


def _load_poisson(path):
    return load_model_data(
        path=str(path),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x2", "x3"],
        objective="poisson",
    )


def test_default_split_70_30(poisson_parquet):
    data = _load_poisson(poisson_parquet)
    train, test = TrainTestSplit(train_ratio=0.7, seed=42).split(data)
    assert abs(train.n_rows / data.n_rows - 0.7) < 0.02
    assert train.n_rows + test.n_rows == data.n_rows


def test_split_preserves_exposure(poisson_parquet):
    data = _load_poisson(poisson_parquet)
    train, test = TrainTestSplit().split(data)
    assert train.exposure is not None
    assert test.exposure is not None
    assert train.exposure.len() == train.n_rows


def test_split_is_reproducible(poisson_parquet):
    data = _load_poisson(poisson_parquet)
    splitter = TrainTestSplit(train_ratio=0.7, seed=42)
    t1, _ = splitter.split(data)
    t2, _ = splitter.split(data)
    assert t1.target.to_list() == t2.target.to_list()


def test_split_different_seeds_differ(poisson_parquet):
    data = _load_poisson(poisson_parquet)
    t1, _ = TrainTestSplit(seed=1).split(data)
    t2, _ = TrainTestSplit(seed=2).split(data)
    assert t1.target.to_list() != t2.target.to_list()


def test_split_invalid_ratio(poisson_parquet):
    data = _load_poisson(poisson_parquet)
    with pytest.raises(ValueError, match="train_ratio"):
        TrainTestSplit(train_ratio=1.5).split(data)
