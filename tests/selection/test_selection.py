import polars as pl
import pytest
from ins_gbm.data.loader import load_model_data
from ins_gbm.data.splitter import TrainTestSplit
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.selection.boruta import BorutaSelector
from ins_gbm.selection.importance import ImportancePruner


# ── Boruta ─────────────────────────────────────────────────────────────────────

def test_boruta_returns_classification_dataframe(poisson_parquet):
    # Use only numeric features — in the full pipeline OHE encodes before Boruta
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    selector = BorutaSelector(base_estimator="lightgbm", max_iter=5, seed=42)
    fitted = selector.fit(data)
    clf = fitted.classification()
    assert "feature" in clf.columns
    assert "status" in clf.columns
    assert set(clf["feature"].to_list()) == {"x1", "x3"}
    assert all(s in {"confirmed", "tentative", "rejected"} for s in clf["status"].to_list())


def test_boruta_selected_features_subset(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    selector = BorutaSelector(base_estimator="lightgbm", max_iter=5, seed=42)
    fitted = selector.fit(data)
    selected = fitted.selected_features()
    assert set(selected).issubset({"x1", "x3"})


def test_boruta_rf_base_estimator(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    selector = BorutaSelector(base_estimator="random_forest", max_iter=5, seed=42)
    fitted = selector.fit(data)
    assert fitted.classification() is not None


def test_boruta_only_trained_on_given_data(poisson_parquet):
    """Boruta must not see data outside the ModelData passed to fit."""
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    train, test = TrainTestSplit().split(data)
    selector = BorutaSelector(max_iter=3, seed=42)
    # Should fit without error on training data only
    fitted = selector.fit(train)
    assert fitted is not None


# ── ImportancePruner ───────────────────────────────────────────────────────────

def test_importance_pruner_top_n(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    fitted_model = LightGBMModel(objective="poisson").fit(
        data, params={"n_estimators": 20, "verbose": -1}
    )
    pruner = ImportancePruner(top_n=2)
    fitted_pruner = pruner.fit(data, fitted_model)
    selected = fitted_pruner.selected_features()
    assert len(selected) == 2


def test_importance_pruner_percentile(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    fitted_model = LightGBMModel(objective="poisson").fit(
        data, params={"n_estimators": 20, "verbose": -1}
    )
    pruner = ImportancePruner(percentile=50.0)  # keep top 50%
    fitted_pruner = pruner.fit(data, fitted_model)
    selected = fitted_pruner.selected_features()
    assert 1 <= len(selected) <= 2


def test_importance_pruner_threshold(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    fitted_model = LightGBMModel(objective="poisson").fit(
        data, params={"n_estimators": 20, "verbose": -1}
    )
    # threshold=0 keeps everything
    pruner = ImportancePruner(threshold=0.0)
    fitted_pruner = pruner.fit(data, fitted_model)
    assert len(fitted_pruner.selected_features()) == 2
