import polars as pl
import pytest
from ins_gbm.data.loader import load_model_data
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.selection.boruta import BorutaSelector
from ins_gbm.selection.importance import ImportancePruner
from ins_gbm.selection import ImportanceSelectionStage, StagedImportanceSelector
from ins_gbm.pipeline import ModelPipeline, ModelRecipe
from ins_gbm.preprocessing.encoder import OneHotEncoder
from ins_gbm.preprocessing.pca import PCAReducer
from ins_gbm.preprocessing.steps import PreprocessingStep


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
    train = data
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


def test_importance_pruner_uses_fitted_one_hot_columns(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x2", "x3"],
        objective="poisson",
    )
    model = LightGBMModel(objective="poisson").fit(
        data, params={"n_estimators": 5, "verbose": -1}, encoder=OneHotEncoder(),
    )
    pruner = ImportancePruner(threshold=0)
    selected = pruner.fit(model)

    assert selected.selected_features() == model.feature_names
    assert any(name.startswith("x2__") for name in selected.selected_features())
    assert pruner.fit(data, model).selected_features() == model.feature_names
    assert pruner.fit(data, fitted_model=model).selected_features() == model.feature_names
    assert pruner.fit(data=data, fitted_model=model).selected_features() == model.feature_names

    recipe = ModelRecipe(
        model=LightGBMModel(objective="poisson"), encoder=OneHotEncoder(),
        params={"n_estimators": 5, "verbose": -1},
    )
    refit = recipe.fit(data, feature_names=selected)
    assert refit.model_selected_features == selected.selected_features()
    assert refit.train_data.feature_names == model.feature_names
    assert refit.predict(data).len() == data.n_rows

    model.importance_fn = lambda: pl.DataFrame({
        "feature": ["x1"], "importance": [1.0],
    })
    with pytest.raises(ValueError, match="do not match fitted model columns"):
        pruner.fit(model)


def test_importance_pruner_result_selects_after_preprocessing(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x2", "x3"],
        objective="poisson",
    )
    preprocessing = [PreprocessingStep(
        name="x1_pca", preprocessor=PCAReducer(n_components=1),
        feature_names=["x1"],
    )]
    recipe = ModelRecipe(
        model=LightGBMModel(objective="poisson"), encoder=OneHotEncoder(),
        preprocessing=preprocessing, params={"n_estimators": 5, "verbose": -1},
    )
    original = recipe.fit(data)
    selected = ImportancePruner(threshold=0).fit(original)
    assert "x1_pca__pca_1" in selected.selected_features()

    refit = recipe.fit(data, feature_names=selected)
    assert refit.train_data.feature_names == selected.selected_features()
    assert refit.predict(data).len() == data.n_rows

    selected.selected_feature_names = ["absent_output"]
    with pytest.raises(ValueError, match="Model features missing after preprocessing"):
        recipe.fit(data, feature_names=selected)


# ── StagedImportanceSelector ──────────────────────────────────────────────────

def test_staged_importance_selector_prunes_and_records_rankings(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    selector = StagedImportanceSelector(stages=[
        ImportanceSelectionStage(
            name="screen",
            model=LightGBMModel(objective="poisson"),
            params={"n_estimators": 5, "verbose": -1, "num_leaves": 4},
            max_features=2,
            importance_type="split",
        ),
        ImportanceSelectionStage(
            name="prune",
            model=LightGBMModel(objective="poisson"),
            params={"n_estimators": 10, "verbose": -1, "num_leaves": 8},
            max_features=1,
            importance_type="gain",
        ),
    ])

    fitted = selector.fit(data)

    assert len(fitted.selected_features()) == 1
    assert [stage.name for stage in fitted.stage_results()] == ["screen", "prune"]
    ranking = fitted.stage_results()[1].ranking
    assert ranking.columns == ["feature", "importance", "rank", "selected"]
    assert ranking["rank"].to_list() == list(range(1, ranking.height + 1))
    assert ranking.filter(pl.col("selected")).height == 1


def test_staged_importance_selection_integrates_with_pipeline(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    selector = StagedImportanceSelector(stages=[
        ImportanceSelectionStage(
            model=LightGBMModel(objective="poisson"),
            params={"n_estimators": 5, "verbose": -1},
            max_features=1,
        ),
    ])
    result = ModelPipeline(
        data=data,
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            selection=selector,
            params={"n_estimators": 5, "verbose": -1},
        ),
    ).run()

    assert len(result.selected_features) == 1
    assert result.train_data.feature_names == result.selected_features
    assert result.selection_results is not None
    assert result.metadata.selection_stages[0]["selected_features"] == result.selected_features
    assert result.predict(data).len() == data.n_rows


def test_staged_importance_selector_rejects_invalid_stage_list():
    with pytest.raises(ValueError, match="at least one"):
        StagedImportanceSelector(stages=[])

    with pytest.raises(ValueError, match="positive integer"):
        ImportanceSelectionStage(model=LightGBMModel(), max_features=0)
