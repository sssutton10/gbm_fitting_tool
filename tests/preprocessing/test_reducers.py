import polars as pl
import pytest
from ins_gbm.preprocessing.pca import PCAReducer
from ins_gbm.preprocessing.pls import PLSReducer
from ins_gbm.preprocessing.umap import UMAPReducer


def _numeric_df(n=100):
    import numpy as np
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "c": rng.normal(0, 1, n),
        "d": rng.normal(0, 1, n),
    })


def _target(n=100):
    import numpy as np
    rng = np.random.default_rng(42)
    return pl.Series(rng.normal(0, 1, n))


# ── PCA ────────────────────────────────────────────────────────────────────────

def test_pca_reduces_dimensions():
    df = _numeric_df()
    fitted = PCAReducer(n_components=2).fit(df)
    out = fitted.transform(df)
    assert out.shape == (100, 2)


def test_pca_component_names():
    df = _numeric_df()
    fitted = PCAReducer(n_components=2).fit(df)
    names = fitted.output_feature_names()
    assert len(names) == 2
    assert all("pca_" in n for n in names)


def test_pca_component_mapping():
    df = _numeric_df()
    fitted = PCAReducer(n_components=2).fit(df)
    mapping = fitted.component_mapping()
    assert isinstance(mapping, dict)
    assert len(mapping) == 2


def test_pca_transform_matches_fit_dimensions():
    df = _numeric_df(100)
    test_df = _numeric_df(20)
    fitted = PCAReducer(n_components=3).fit(df)
    out = fitted.transform(test_df)
    assert out.shape == (20, 3)


# ── PLS ────────────────────────────────────────────────────────────────────────

def test_pls_reduces_dimensions():
    df = _numeric_df()
    target = _target()
    fitted = PLSReducer(n_components=2).fit(df, target=target)
    out = fitted.transform(df)
    assert out.shape == (100, 2)


def test_pls_requires_target_at_fit():
    df = _numeric_df()
    with pytest.raises((ValueError, TypeError)):
        PLSReducer(n_components=2).fit(df, target=None)


def test_pls_component_names():
    df = _numeric_df()
    target = _target()
    fitted = PLSReducer(n_components=2).fit(df, target=target)
    assert all("pls_" in n for n in fitted.output_feature_names())


# ── UMAP ───────────────────────────────────────────────────────────────────────

def test_umap_reduces_dimensions():
    df = _numeric_df()
    fitted = UMAPReducer(n_components=2, n_neighbors=5).fit(df)
    out = fitted.transform(df)
    assert out.shape == (100, 2)


def test_umap_component_names():
    df = _numeric_df()
    fitted = UMAPReducer(n_components=2, n_neighbors=5).fit(df)
    assert all("umap_" in n for n in fitted.output_feature_names())


def test_umap_transform_new_data():
    df = _numeric_df(80)
    test_df = _numeric_df(20)
    fitted = UMAPReducer(n_components=2, n_neighbors=5).fit(df)
    out = fitted.transform(test_df)
    assert out.shape == (20, 2)
