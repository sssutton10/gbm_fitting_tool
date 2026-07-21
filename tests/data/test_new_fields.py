"""Tests for offset, cv_fold, and comparisons fields on ModelData.

Written before implementation (TDD approach).
"""
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from ins_gbm.data.loader import load_model_data
from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.data.schema import FeatureSchema


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_data():
    """Minimal ModelData (10 rows, no optional fields) for field-level tests."""
    n = 10
    features = pl.DataFrame({"x": [float(i) for i in range(n)]})
    target = pl.Series("target", [float(i + 1) for i in range(n)])
    return ModelData(
        features=features,
        target=target,
        exposure=None,
        weight=None,
        feature_names=["x"],
    )


@pytest.fixture
def valid_offset(base_data):
    return pl.Series("offset", [float(i) * 0.1 for i in range(base_data.n_rows)])


@pytest.fixture
def valid_cv_fold(base_data):
    n = base_data.n_rows  # 10
    return pl.Series("cv_fold", [i % 3 for i in range(n)], dtype=pl.Int32)


@pytest.fixture
def valid_comparisons(base_data):
    n = base_data.n_rows
    return pl.DataFrame({
        "model_a": [float(i + 1) * 1.1 for i in range(n)],
        "model_b": [float(i + 1) * 0.9 for i in range(n)],
    })


# ---------------------------------------------------------------------------
# 1. validate() — offset
# ---------------------------------------------------------------------------

class TestValidateOffset:
    def test_valid_offset_passes(self, base_data, valid_offset):
        data = replace(base_data, offset=valid_offset)
        data.validate()  # should not raise

    def test_offset_wrong_length_raises(self, base_data):
        short_offset = pl.Series("offset", [0.1, 0.2, 0.3])
        data = replace(base_data, offset=short_offset)
        with pytest.raises(ValueError, match="offset"):
            data.validate()

    def test_offset_with_null_raises(self, base_data):
        offset_with_null = pl.Series("offset", [None] + [0.1] * (base_data.n_rows - 1))
        data = replace(base_data, offset=offset_with_null)
        with pytest.raises(ValueError, match="offset"):
            data.validate()

    def test_offset_with_inf_raises(self, base_data):
        vals = [0.1] * base_data.n_rows
        vals[2] = float("inf")
        offset_with_inf = pl.Series("offset", vals)
        data = replace(base_data, offset=offset_with_inf)
        with pytest.raises(ValueError, match="offset"):
            data.validate()

    def test_offset_non_numeric_raises(self, base_data):
        str_offset = pl.Series("offset", ["a"] * base_data.n_rows)
        data = replace(base_data, offset=str_offset)
        with pytest.raises(ValueError):
            data.validate()


# ---------------------------------------------------------------------------
# 2. validate() — cv_fold
# ---------------------------------------------------------------------------

class TestValidateCvFold:
    def test_valid_cv_fold_passes(self, base_data, valid_cv_fold):
        data = replace(base_data, cv_fold=valid_cv_fold)
        data.validate()

    def test_cv_fold_wrong_length_raises(self, base_data):
        short_fold = pl.Series("cv_fold", [0, 1, 2], dtype=pl.Int32)
        data = replace(base_data, cv_fold=short_fold)
        with pytest.raises(ValueError, match="cv_fold"):
            data.validate()

    def test_cv_fold_with_null_raises(self, base_data):
        fold_with_null = pl.Series("cv_fold", [None] + [0] * (base_data.n_rows - 1), dtype=pl.Int32)
        data = replace(base_data, cv_fold=fold_with_null)
        with pytest.raises(ValueError, match="cv_fold"):
            data.validate()

    def test_cv_fold_single_value_raises(self, base_data):
        single_fold = pl.Series("cv_fold", [0] * base_data.n_rows, dtype=pl.Int32)
        data = replace(base_data, cv_fold=single_fold)
        with pytest.raises(ValueError, match="cv_fold"):
            data.validate()

    def test_cv_fold_float_raises(self, base_data):
        float_fold = pl.Series("cv_fold", [0.5] * base_data.n_rows)
        data = replace(base_data, cv_fold=float_fold)
        with pytest.raises(ValueError, match="cv_fold"):
            data.validate()


# ---------------------------------------------------------------------------
# 3. validate() — comparisons
# ---------------------------------------------------------------------------

class TestValidateComparisons:
    def test_valid_comparisons_passes(self, base_data, valid_comparisons):
        data = replace(base_data, comparisons=valid_comparisons)
        data.validate()

    def test_comparisons_wrong_row_count_raises(self, base_data):
        short_comp = pl.DataFrame({"model_a": [1.0, 2.0, 3.0]})
        data = replace(base_data, comparisons=short_comp)
        with pytest.raises(ValueError, match="comparisons"):
            data.validate()

    def test_comparisons_negative_value_raises(self, base_data):
        n = base_data.n_rows
        bad_comp = pl.DataFrame({"model_a": [-1.0] + [1.0] * (n - 1)})
        data = replace(base_data, comparisons=bad_comp)
        with pytest.raises(ValueError, match="comparisons"):
            data.validate()

    def test_comparisons_zero_value_raises(self, base_data):
        n = base_data.n_rows
        bad_comp = pl.DataFrame({"model_a": [0.0] + [1.0] * (n - 1)})
        data = replace(base_data, comparisons=bad_comp)
        with pytest.raises(ValueError, match="comparisons"):
            data.validate()


# ---------------------------------------------------------------------------
# 4. slice_model_data — all three fields are sliced correctly
# ---------------------------------------------------------------------------

class TestSliceModelData:
    def test_slice_offset(self, base_data, valid_offset):
        data = replace(base_data, offset=valid_offset)
        indices = [0, 2, 4, 6, 8]
        sliced = slice_model_data(data, indices)
        assert sliced.offset is not None
        assert sliced.offset.to_list() == valid_offset[indices].to_list()

    def test_slice_cv_fold(self, base_data, valid_cv_fold):
        data = replace(base_data, cv_fold=valid_cv_fold)
        indices = [1, 3, 5, 7, 9]
        sliced = slice_model_data(data, indices)
        assert sliced.cv_fold is not None
        assert sliced.cv_fold.to_list() == valid_cv_fold[indices].to_list()

    def test_slice_comparisons(self, base_data, valid_comparisons):
        data = replace(base_data, comparisons=valid_comparisons)
        indices = [0, 1, 2, 3, 4]
        sliced = slice_model_data(data, indices)
        assert sliced.comparisons is not None
        assert sliced.comparisons.shape[0] == 5
        assert sliced.comparisons["model_a"].to_list() == valid_comparisons["model_a"][indices].to_list()

    def test_slice_none_fields_stay_none(self, base_data):
        sliced = slice_model_data(base_data, [0, 1, 2])
        assert sliced.offset is None
        assert sliced.cv_fold is None
        assert sliced.comparisons is None


# ---------------------------------------------------------------------------
# 5. with_offset convenience method
# ---------------------------------------------------------------------------

class TestWithOffset:
    def test_with_offset_sets_field(self, base_data, valid_offset):
        updated = base_data.with_offset(valid_offset)
        assert updated.offset is not None
        assert updated.offset.to_list() == valid_offset.to_list()

    def test_with_offset_original_unchanged(self, base_data, valid_offset):
        _ = base_data.with_offset(valid_offset)
        assert base_data.offset is None

    def test_with_offset_other_fields_preserved(self, base_data, valid_offset):
        updated = base_data.with_offset(valid_offset)
        assert updated.target.to_list() == base_data.target.to_list()
        assert updated.feature_names == base_data.feature_names


# ---------------------------------------------------------------------------
# 6. Explicit slicing preserves all three fields
# ---------------------------------------------------------------------------

class TestSliceModelDataNewFields:
    @pytest.fixture
    def rich_data(self):
        """200-row ModelData with all three new fields set."""
        n = 200
        rng = np.random.default_rng(99)
        features = pl.DataFrame({"x": rng.normal(size=n).tolist()})
        target = pl.Series("target", (rng.uniform(0.5, 2.0, n)).tolist())
        offset = pl.Series("offset", rng.normal(size=n).tolist())
        cv_fold = pl.Series("cv_fold", [i % 5 for i in range(n)], dtype=pl.Int32)
        comparisons = pl.DataFrame({
            "ext_model": (rng.uniform(0.1, 2.0, n)).tolist(),
        })
        return ModelData(
            features=features,
            target=target,
            exposure=None,
            weight=None,
            feature_names=["x"],
            offset=offset,
            cv_fold=cv_fold,
            comparisons=comparisons,
        )

    def test_slice_propagates_offset(self, rich_data):
        holdout = slice_model_data(rich_data, range(100))
        assert holdout.offset is not None
        assert holdout.offset.len() == holdout.n_rows

    def test_slice_propagates_cv_fold(self, rich_data):
        holdout = slice_model_data(rich_data, range(100))
        assert holdout.cv_fold is not None
        assert holdout.cv_fold.len() == holdout.n_rows

    def test_slice_propagates_comparisons(self, rich_data):
        holdout = slice_model_data(rich_data, range(100))
        assert holdout.comparisons is not None
        assert holdout.comparisons.shape[0] == holdout.n_rows

    def test_slice_none_fields_stay_none(self):
        """When fields are None, they remain None after split."""
        n = 50
        features = pl.DataFrame({"x": [float(i) for i in range(n)]})
        target = pl.Series("target", [1.0] * n)
        data = ModelData(
            features=features, target=target, exposure=None, weight=None,
            feature_names=["x"],
        )
        holdout = slice_model_data(data, range(25))
        assert holdout.offset is None
        assert holdout.cv_fold is None
        assert holdout.comparisons is None


# ---------------------------------------------------------------------------
# 7. load_model_data — new parameters cv_fold and comparison_cols
# ---------------------------------------------------------------------------

@pytest.fixture
def extended_parquet(tmp_path):
    """Parquet with target, exposure, features plus cv_fold and two comparison columns."""
    n = 100
    rng = np.random.default_rng(55)
    df = pl.DataFrame({
        "x1": rng.normal(size=n).tolist(),
        "x2": rng.choice(["A", "B"], n).tolist(),
        "claim_count": rng.poisson(1.0, n).astype(float).tolist(),
        "exposure": rng.uniform(0.5, 2.0, n).tolist(),
        "cv_fold": [i % 4 for i in range(n)],
        "model_ext": rng.uniform(0.1, 3.0, n).tolist(),
        "model_ext2": rng.uniform(0.2, 2.0, n).tolist(),
    })
    path = tmp_path / "extended.parquet"
    df.write_parquet(path)
    return path


class TestLoadModelDataNewParams:
    def test_cv_fold_loaded_correctly(self, extended_parquet):
        data = load_model_data(
            path=str(extended_parquet),
            target="claim_count",
            exposure="exposure",
            feature_cols=["x1", "x2"],
            cv_fold="cv_fold",
        )
        assert data.cv_fold is not None
        assert data.cv_fold.len() == data.n_rows
        assert data.cv_fold.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                       pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)

    def test_comparison_cols_loaded_correctly(self, extended_parquet):
        data = load_model_data(
            path=str(extended_parquet),
            target="claim_count",
            exposure="exposure",
            feature_cols=["x1", "x2"],
            comparison_cols=["model_ext", "model_ext2"],
        )
        assert data.comparisons is not None
        assert data.comparisons.shape == (data.n_rows, 2)
        assert list(data.comparisons.columns) == ["model_ext", "model_ext2"]

    def test_cv_fold_excluded_from_feature_cols(self, extended_parquet):
        """When cv_fold is specified but feature_cols is None, cv_fold must not appear in features."""
        data = load_model_data(
            path=str(extended_parquet),
            target="claim_count",
            exposure="exposure",
            cv_fold="cv_fold",
        )
        assert "cv_fold" not in data.feature_names

    def test_comparison_cols_excluded_from_feature_cols(self, extended_parquet):
        """When comparison_cols specified but feature_cols is None, they must not appear in features."""
        data = load_model_data(
            path=str(extended_parquet),
            target="claim_count",
            exposure="exposure",
            comparison_cols=["model_ext"],
        )
        assert "model_ext" not in data.feature_names

    def test_load_without_new_params_still_works(self, extended_parquet):
        """Backwards compatibility: omitting new params gives None fields."""
        data = load_model_data(
            path=str(extended_parquet),
            target="claim_count",
            exposure="exposure",
            feature_cols=["x1", "x2"],
        )
        assert data.cv_fold is None
        assert data.comparisons is None
