import os

import cloudpickle
import pytest

from ins_gbm.data.loader import load_model_data
from ins_gbm.ensemble.blending import BlendingEnsemble
from ins_gbm.ensemble.stacking import StackingEnsemble
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.persistence.io import load_pipeline, save_pipeline
from ins_gbm.pipeline import ModelPipeline, ModelRecipe
from ins_gbm.preprocessing.encoder import OneHotEncoder
from ins_gbm.tuning.tuner import HyperparameterTuner


def test_save_load_preserves_predictions_without_metrics_artifact(poisson_parquet, tmp_path):
    data = load_model_data(path=str(poisson_parquet), target="claim_count", exposure="exposure", feature_cols=["x1", "x3"], objective="poisson")
    fitted = ModelPipeline(data=data, recipe=ModelRecipe(model=LightGBMModel(objective="poisson"))).run()
    save_pipeline(fitted, str(tmp_path))
    loaded = load_pipeline(str(tmp_path))

    assert fitted.predict(data).to_list() == pytest.approx(loaded.predict(data).to_list(), rel=1e-6)
    assert fitted.raw_train_data is data
    assert loaded.raw_train_data is None
    with pytest.raises(RuntimeError, match="training_data=original_training_data"):
        _ = loaded.train_data
    with pytest.raises(RuntimeError, match="training_data=original_training_data"):
        loaded.retune(
            HyperparameterTuner(
                n_trials=1,
                cv_folds=2,
                show_progress_bar=False,
            )
        )
    assert os.path.exists(tmp_path / "pipeline.pkl")
    assert os.path.exists(tmp_path / "metadata.json")
    assert not os.path.exists(tmp_path / "metrics.csv")


def test_compact_load_predict_raw_retains_input_schema(poisson_parquet, tmp_path):
    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x2", "x3"],
        objective="poisson",
    )
    fitted = ModelPipeline(
        data=data,
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            encoder=OneHotEncoder(),
            params={"n_estimators": 5},
        ),
    ).run()
    save_pipeline(fitted, str(tmp_path))

    loaded = load_pipeline(str(tmp_path))
    expected = fitted.predict_raw(data.features, exposure=data.exposure)
    actual = loaded.predict_raw(data.features, exposure=data.exposure)

    assert loaded.input_schema == data.schema
    assert actual.to_list() == pytest.approx(expected.to_list(), rel=1e-6)
    assert loaded.fitted_model.feature_importance().height > 0


def test_retuned_pipeline_preserves_predictions_and_history(poisson_parquet, tmp_path):
    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
    )
    original = ModelPipeline(
        data=data,
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            params={"n_estimators": 5},
        ),
    ).run(feature_names=["x1"], feature_stage="encoded")
    tuned = original.retune(
        HyperparameterTuner(
            n_trials=1,
            cv_folds=2,
            seed=19,
            show_progress_bar=False,
        )
    )

    save_pipeline(tuned, str(tmp_path))
    loaded = load_pipeline(str(tmp_path))

    assert loaded.predict(data).to_list() == pytest.approx(
        tuned.predict(data).to_list(),
        rel=1e-6,
    )
    assert loaded.tuning_history.equals(tuned.tuning_history)
    assert loaded.metadata.random_seeds["tuning"] == 19
    assert os.path.exists(tmp_path / "tuning_history.parquet")


def test_load_can_reattach_full_training_feature_pool(poisson_parquet, tmp_path):
    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x2", "x3"],
        objective="poisson",
    )
    fitted = ModelPipeline(
        data=data,
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            params={"n_estimators": 5},
        ),
    ).run(feature_names=["x1", "x3"])
    save_pipeline(fitted, str(tmp_path))

    loaded = load_pipeline(str(tmp_path), training_data=data)

    assert loaded.raw_train_data is not data
    assert loaded.raw_train_data.features.columns == ["x1", "x3"]
    assert loaded.train_data.n_rows == data.n_rows

    ensemble = BlendingEnsemble(mode="oof", cv_folds=2).fit([loaded, loaded])
    assert ensemble.predict(data).len() == data.n_rows


def test_compact_load_rejects_data_dependent_oof(poisson_parquet, tmp_path):
    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
    )
    fitted = ModelPipeline(
        data=data,
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            params={"n_estimators": 5},
        ),
    ).run()
    save_pipeline(fitted, str(tmp_path))
    loaded = load_pipeline(str(tmp_path))

    with pytest.raises(RuntimeError, match="training_data=original_training_data"):
        BlendingEnsemble(mode="oof", cv_folds=2).fit([loaded, loaded])
    with pytest.raises(RuntimeError, match="training_data=original_training_data"):
        StackingEnsemble(cv_folds=2).fit([loaded, loaded])


def test_load_rejects_incompatible_training_data(poisson_parquet, tmp_path):
    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
    )
    fitted = ModelPipeline(
        data=data,
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
    ).run()
    save_pipeline(fitted, str(tmp_path))

    incompatible_objective = data.__class__(
        features=data.features,
        target=data.target,
        exposure=data.exposure,
        weight=data.weight,
        feature_names=data.feature_names,
        schema=data.schema,
        objective="gamma",
    )
    with pytest.raises(ValueError, match="objective does not match"):
        load_pipeline(str(tmp_path), training_data=incompatible_objective)

    missing_feature = data.select_features(["x1"])
    with pytest.raises(ValueError, match="missing columns"):
        load_pipeline(str(tmp_path), training_data=missing_feature)


def test_load_legacy_full_data_artifact(poisson_parquet, tmp_path):
    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
    )
    fitted = ModelPipeline(
        data=data,
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            params={"n_estimators": 5},
        ),
    ).run()
    del fitted.input_schema  # Simulate an artifact created before this field existed.

    compact_dir = tmp_path / "compact"
    save_pipeline(fitted, str(compact_dir))

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    with open(legacy_dir / "pipeline.pkl", "wb") as artifact:
        cloudpickle.dump(fitted, artifact)

    loaded = load_pipeline(str(legacy_dir))

    assert (compact_dir / "pipeline.pkl").stat().st_size < (
        legacy_dir / "pipeline.pkl"
    ).stat().st_size
    assert loaded.raw_train_data.n_rows == data.n_rows
    assert loaded._input_schema() == data.schema
    assert loaded.predict(data).len() == data.n_rows
