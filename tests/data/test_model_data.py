import polars as pl
import pytest
from ins_gbm.data.model_data import ModelData
from ins_gbm.data.schema import FeatureSchema


def test_model_data_poisson_valid(poisson_raw):
    data = ModelData(
        features=poisson_raw.select(["x1", "x2", "x3"]),
        target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"],
        weight=None,
        feature_names=["x1", "x2", "x3"],
        schema=FeatureSchema(numeric=["x1", "x3"], categorical=["x2"]),
        objective="poisson",
    ).validate()
    assert data.n_rows == 400
    assert data.feature_names == ["x1", "x2", "x3"]


def test_model_data_gamma_valid(gamma_raw):
    data = ModelData(
        features=gamma_raw.select(["x1", "x2"]),
        target=gamma_raw["severity"],
        exposure=None,
        weight=gamma_raw["weight"],
        feature_names=["x1", "x2"],
        objective="gamma",
    ).validate()
    assert data.n_rows == 300


def test_poisson_requires_exposure(poisson_raw):
    with pytest.raises(ValueError, match="exposure is required"):
        ModelData(
            features=poisson_raw.select(["x1"]),
            target=poisson_raw["claim_count"],
            exposure=None,
            weight=None,
            feature_names=["x1"],
            objective="poisson",
        ).validate()


def test_poisson_nonnegative_target(poisson_raw):
    bad = poisson_raw["claim_count"].clone()
    bad = pl.Series("claim_count", [-1.0] + bad[1:].to_list())
    with pytest.raises(ValueError, match="non-negative"):
        ModelData(
            features=poisson_raw.select(["x1"]),
            target=bad,
            exposure=poisson_raw["exposure"],
            weight=None,
            feature_names=["x1"],
            objective="poisson",
        ).validate()


def test_gamma_positive_target(gamma_raw):
    bad = pl.Series("severity", [0.0] + gamma_raw["severity"][1:].to_list())
    with pytest.raises(ValueError, match="strictly positive"):
        ModelData(
            features=gamma_raw.select(["x1"]),
            target=bad,
            exposure=None,
            weight=None,
            feature_names=["x1"],
            objective="gamma",
        ).validate()


def test_row_count_mismatch(poisson_raw):
    with pytest.raises(ValueError, match="row count"):
        ModelData(
            features=poisson_raw.select(["x1"]),
            target=poisson_raw["claim_count"].head(10),
            exposure=poisson_raw["exposure"],
            weight=None,
            feature_names=["x1"],
            objective="poisson",
        ).validate()


def test_duplicate_feature_names(poisson_raw):
    with pytest.raises(ValueError, match="unique"):
        ModelData(
            features=poisson_raw.select(["x1", "x3"]),
            target=poisson_raw["claim_count"],
            exposure=poisson_raw["exposure"],
            weight=None,
            feature_names=["x1", "x1"],
            objective="poisson",
        ).validate()


def test_positive_exposure_required(poisson_raw):
    bad_exposure = pl.Series("exposure", [0.0] + poisson_raw["exposure"][1:].to_list())
    with pytest.raises(ValueError, match="positive"):
        ModelData(
            features=poisson_raw.select(["x1"]),
            target=poisson_raw["claim_count"],
            exposure=bad_exposure,
            weight=None,
            feature_names=["x1"],
            objective="poisson",
        ).validate()


def test_with_features(poisson_raw):
    data = ModelData(
        features=poisson_raw.select(["x1", "x3"]),
        target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"],
        weight=None,
        feature_names=["x1", "x3"],
        objective="poisson",
    )
    new_features = poisson_raw.select(["x1"])
    updated = data.with_features(new_features)
    assert updated.feature_names == ["x1"]
    assert updated.target is data.target


def test_select_features_preserves_row_fields_and_filters_schema():
    data = ModelData(
        features=pl.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "c": [3.0, 4.0]}),
        target=pl.Series("target", [1.0, 2.0]),
        exposure=None,
        weight=None,
        feature_names=["a", "b", "c"],
        schema=FeatureSchema(numeric=["a", "c"], categorical=["b"]),
        objective="gamma",
        cv_fold=pl.Series("fold", [0, 1]),
    ).validate()

    selected = data.select_features(["c", "b"])

    assert selected.features.columns == ["c", "b"]
    assert selected.feature_names == ["c", "b"]
    assert selected.schema == FeatureSchema(numeric=["c"], categorical=["b"])
    assert selected.target is data.target
    assert selected.cv_fold is data.cv_fold


@pytest.mark.parametrize("names", [[], ["x1", "x1"], ["missing"]])
def test_select_features_rejects_invalid_names(poisson_raw, names):
    data = ModelData(
        features=poisson_raw.select(["x1", "x3"]),
        target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"],
        weight=None,
        feature_names=["x1", "x3"],
        objective="poisson",
    ).validate()

    with pytest.raises(ValueError):
        data.select_features(names)
