import json

import numpy as np
import polars as pl
import pytest

from ins_gbm import EnsemblePipeline, ModelData, ModelRecipe, load_ensemble, load_model
from ins_gbm.ensemble.blending import BlendingEnsemble
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.persistence.io import load_pipeline, save_pipeline
from ins_gbm.preprocessing.encoder import OneHotEncoder
from ins_gbm.preprocessing.pca import PCAReducer
from ins_gbm.preprocessing.steps import PreprocessingStep


@pytest.fixture
def feature_pool():
    rng = np.random.default_rng(12)
    features = pl.DataFrame({
        "x": rng.normal(size=60), "y": rng.normal(size=60),
        "unused": rng.normal(size=60),
        "category": ["a", "b", "c"] * 20,
        "unused_category": ["secret", "other"] * 30,
    })
    return ModelData(
        features=features, feature_names=features.columns,
        target=pl.Series(rng.poisson(2, size=60)), objective="poisson",
    )


def fit(data, *, encoder=None, preprocessing=None, **kwargs):
    return ModelRecipe(
        model=LightGBMModel(), encoder=encoder,
        preprocessing=preprocessing or [],
        params={"n_estimators": 4, "num_threads": 1, "min_child_samples": 5},
    ).fit(data, **kwargs)


@pytest.mark.parametrize("include_training_data", [False, True])
def test_save_feature_contract_and_optional_training_rows(
    feature_pool, tmp_path, include_training_data,
):
    fitted = fit(
        feature_pool, encoder=OneHotEncoder(),
        feature_names=["category__b", "x"], feature_stage="encoded",
    )
    expected = fitted.predict(feature_pool)
    fitted.save(tmp_path, include_training_data=include_training_data)
    loaded = load_model(tmp_path)
    if include_training_data:
        assert loaded.input_feature_names == feature_pool.feature_names
        assert loaded.raw_train_data.features.equals(feature_pool.features)
        assert loaded.train_data.features.equals(fitted.train_data.features)
        assert loaded.encoder.levels == fitted.encoder.levels
        replacement = feature_pool.select_features(feature_pool.feature_names)
        reattached = load_pipeline(tmp_path, training_data=replacement)
        assert reattached.raw_train_data.features.equals(replacement.features)
    else:
        assert loaded.input_feature_names == ["x", "category"]
        assert loaded.input_schema.all_features() == ["x", "category"]
        assert loaded.raw_train_data is None
        assert loaded.encoder.levels == {"category": ["b"]}
        assert loaded.encoder.numeric_cols == ["x"]
        assert loaded.encoder.output_feature_names() == ["x", "category__b"]
        minimal = feature_pool.select_features(["x", "category"])
        np.testing.assert_allclose(loaded.predict(minimal), expected)
        np.testing.assert_allclose(loaded.predict_raw(minimal.features), expected)
        attached = load_pipeline(tmp_path, training_data=minimal)
        assert attached.raw_train_data.feature_names == ["x", "category"]
        assert attached.train_data.features.equals(fitted.train_data.features)
        with pytest.raises(ValueError, match="missing columns"):
            loaded.predict_raw(minimal.features.drop("category"))
        metadata = json.loads((tmp_path / "metadata.json").read_text())
        assert metadata["input_feature_names"] == ["x", "category"]
    # Saving must leave the in-memory fit and its encoder intact.
    assert fitted.input_feature_names == feature_pool.feature_names
    assert fitted.raw_train_data is feature_pool
    assert "unused_category" in fitted.encoder.levels
    assert fitted.encoder.numeric_cols == ["x", "y", "unused"]
    np.testing.assert_allclose(loaded.predict(feature_pool), expected)


def test_numeric_selection_without_encoder(feature_pool, tmp_path):
    data = feature_pool.select_features(["x", "y", "unused"])
    fitted = fit(data, feature_names=["y", "x"], feature_stage="model")
    save_pipeline(fitted, tmp_path)
    loaded = load_pipeline(tmp_path, training_data=data.select_features(["x", "y"]))
    assert loaded.input_feature_names == ["x", "y"]
    assert loaded.train_data.features.columns == ["y", "x"]
    np.testing.assert_allclose(loaded.predict_raw(data.features.select("x", "y")),
                               fitted.predict(data))
    ensemble = BlendingEnsemble(mode="oof", cv_folds=2).fit([loaded, loaded])
    assert ensemble.predict(data.select_features(["x", "y"])).len() == data.n_rows


def test_chained_components_keep_dependencies_and_drop_dead_steps(feature_pool, tmp_path):
    fitted = fit(
        feature_pool, encoder=OneHotEncoder(),
        preprocessing=[
            PreprocessingStep("dead", PCAReducer(1), ["unused", "y"]),
            PreprocessingStep("first", PCAReducer(2), ["x", "category__b"]),
            PreprocessingStep("second", PCAReducer(1), ["first__pca_1", "first__pca_2"]),
        ],
        feature_names=["second__pca_1"], feature_stage="model",
    )
    fitted.save(tmp_path)
    minimal = feature_pool.select_features(["x", "category"])
    loaded = load_pipeline(tmp_path, training_data=minimal)
    assert loaded.input_feature_names == ["x", "category"]
    assert [step.name for step in loaded.preprocessors] == ["first", "second"]
    assert [step.name for step in loaded.recipe.preprocessing] == ["first", "second"]
    np.testing.assert_allclose(loaded.predict(minimal), fitted.predict(feature_pool))
    assert loaded.train_data.features.equals(fitted.train_data.features)
    # Re-saving a compact artifact must preserve the same dependency contract.
    loaded.save(tmp_path / "again")
    again = load_pipeline(tmp_path / "again")
    np.testing.assert_allclose(again.predict(minimal), fitted.predict(feature_pool))
    # Both inputs to the retained PCA are still required even with one final component.
    with pytest.raises(ValueError, match="missing columns"):
        loaded.predict_raw(minimal.features.drop("x"))


def test_whole_frame_pca_keeps_all_component_inputs(feature_pool, tmp_path):
    data = feature_pool.select_features(["x", "y", "unused"])
    fitted = fit(data, preprocessing=[PCAReducer(2)],
                 feature_names=["pca_1"], feature_stage="model")
    fitted.save(tmp_path)
    loaded = load_pipeline(tmp_path)
    assert loaded.input_feature_names == data.feature_names
    np.testing.assert_allclose(loaded.predict(data), fitted.predict(data))


@pytest.mark.parametrize("include_training_data", [False, True])
def test_ensemble_members_use_same_save_policy(feature_pool, tmp_path, include_training_data):
    data = feature_pool.select_features(["x", "y", "unused"])
    fitted = fit(data, feature_names=["x"], feature_stage="encoded")
    result = EnsemblePipeline([fitted], blend_weights=[1.]).run()
    result.save(tmp_path, include_training_data=include_training_data)
    loaded = load_ensemble(tmp_path)
    member = loaded.base_pipelines[0]
    assert member.input_feature_names == (data.feature_names if include_training_data else ["x"])
    assert (member.raw_train_data is not None) == include_training_data
    scoring = data if include_training_data else data.select_features(["x"])
    np.testing.assert_allclose(loaded.predict(scoring), result.predict(data))
