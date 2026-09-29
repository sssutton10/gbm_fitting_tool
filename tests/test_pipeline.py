import polars as pl
import pytest

from ins_gbm.data.loader import load_model_data
from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.data.schema import infer_schema
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.persistence.io import load_pipeline
from ins_gbm.pipeline import FittedPipeline, ModelPipeline, ModelRecipe
from ins_gbm.preprocessing.encoder import OneHotEncoder
from ins_gbm.preprocessing.pca import PCAReducer
from ins_gbm.preprocessing.steps import PreprocessingStep
from ins_gbm.selection.importance import ImportancePruner
from ins_gbm.tuning.tuner import HyperparameterTuner


def _data(path):
    """Data.

    Args:
        path (object): Path to the input data file.
    """
    return load_model_data(
        path=str(path),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x3"],
        objective="poisson",
    )


def test_pipeline_fits_all_supplied_rows(poisson_parquet):
    """Verify pipeline fits all supplied rows.

    Args:
        poisson_parquet (object): The poisson parquet.
    """
    data = _data(poisson_parquet)
    result = ModelPipeline(
        data=data, recipe=ModelRecipe(model=LightGBMModel(objective="poisson"))
    ).run()
    assert isinstance(result, FittedPipeline)
    assert result.train_data.n_rows == data.n_rows
    assert "train_data" not in result.__dict__
    assert result.raw_train_data is data
    assert result.train_data is not result.train_data
    assert not hasattr(result, "test_data")
    assert not hasattr(result, "report")


def test_pipeline_manual_params_reach_model(poisson_parquet):
    """Verify pipeline manual params reach model.

    Args:
        poisson_parquet (object): The poisson parquet.
    """
    result = ModelPipeline(
        data=_data(poisson_parquet),
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            params={"n_estimators": 7, "learning_rate": 0.123},
        ),
    ).run()
    assert result.fitted_model.params.get("learning_rate") == 0.123


def test_explicit_evaluation_transforms_holdout_without_retaining_it(poisson_raw):
    """Verify explicit evaluation transforms holdout without retaining it.

    Args:
        poisson_raw (object): The poisson raw.
    """
    schema = infer_schema(poisson_raw, ["x1", "x2", "x3"])
    data = ModelData(
        features=poisson_raw.select(["x1", "x2", "x3"]),
        target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"],
        weight=None,
        feature_names=["x1", "x2", "x3"],
        schema=schema,
        objective="poisson",
    ).validate()
    result = ModelPipeline(
        data=data,
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"), encoder=OneHotEncoder()
        ),
    ).run()
    holdout = slice_model_data(data, range(100))
    report = result.evaluate(holdout)
    assert report.evaluation_data.n_rows == holdout.n_rows
    assert report.evaluation_data.features.columns != holdout.features.columns
    assert not hasattr(result, "evaluation_data")
    assert report.train_data is None
    assert "metric" in report.metrics().columns


def test_predict_raw_matches_explicit_evaluation(poisson_raw):
    """Verify predict raw matches explicit evaluation.

    Args:
        poisson_raw (object): The poisson raw.
    """
    schema = infer_schema(poisson_raw, ["x1", "x2", "x3"])
    data = ModelData(
        features=poisson_raw.select(["x1", "x2", "x3"]),
        target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"],
        weight=None,
        feature_names=["x1", "x2", "x3"],
        schema=schema,
        objective="poisson",
    ).validate()
    result = ModelPipeline(
        data=data,
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"), encoder=OneHotEncoder()
        ),
    ).run()
    holdout = slice_model_data(data, range(100))
    report = result.evaluate(holdout)
    raw_predictions = result.predict_raw(holdout.features, exposure=holdout.exposure)
    direct_predictions = result.fitted_model.predict(report.evaluation_data, "response")
    assert raw_predictions.to_list() == pytest.approx(
        direct_predictions.to_list(), rel=1e-6
    )


def test_tuning_still_uses_cv_and_stores_history(poisson_parquet):
    """Verify tuning still uses cv and stores history.

    Args:
        poisson_parquet (object): The poisson parquet.
    """
    result = ModelPipeline(
        data=_data(poisson_parquet),
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            tuning=HyperparameterTuner(n_trials=2, cv_folds=2, seed=42),
        ),
    ).run()
    assert result.train_data.n_rows == _data(poisson_parquet).n_rows
    assert len(result.tuning_history) == 2


def test_pipeline_defaults_to_feature_selection_before_tuning(poisson_parquet):
    """Verify pipeline defaults to feature selection before tuning.

    Args:
        poisson_parquet (object): The poisson parquet.
    """
    events = []

    class RecordingEncoder:
        """Configure RecordingEncoder."""

        def fit(self, features, schema):
            """Fit.

            Args:
                features (object): Input feature frame; rows align with the target and optional series.
                schema (object): Optional feature schema; inferred when omitted.
            """
            events.append("encode")
            return self

        def transform(self, features):
            """Transform.

            Args:
                features (object): Input feature frame; rows align with the target and optional series.
            """
            return features

    class SelectX3:
        """Configure SelectX3."""

        def fit(self, data):
            """Fit.

            Args:
                data (object): Model data to fit, transform, predict, or evaluate.
            """
            events.append("select")
            return self

        def selected_features(self):
            """Selected features."""
            return ["x3"]

    class RecordingTuner:
        """Configure RecordingTuner."""

        n_trials = 1
        seed = 17

        def tune(self, data, model, **kwargs):
            """Tune.

            Args:
                data (object): Model data to fit, transform, predict, or evaluate.
                model (object): Model wrapper or fitted model to use.
                kwargs (object): Additional keyword arguments forwarded to the pipeline run.
            """
            events.append("tune")
            assert data.feature_names == ["x3"]
            assert kwargs.get("encoder") is None
            assert kwargs.get("selector") is None
            return (
                {"n_estimators": 5, "verbose": -1},
                pl.DataFrame({"trial": [0], "value": [1.0]}),
            )

    result = ModelPipeline(
        data=_data(poisson_parquet),
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            encoder=RecordingEncoder(),
            selection=SelectX3(),
            tuning=RecordingTuner(),
        ),
    ).run()

    assert events == ["encode", "select", "tune"]
    assert result.selected_features == ["x3"]
    assert result.train_data.feature_names == ["x3"]
    assert len(result.tuning_history) == 1
    assert result.metadata.selection_scope == "fixed"


def test_pipeline_fold_scope_tuning_receives_fold_local_transforms(poisson_parquet):
    """Verify pipeline fold scope tuning receives fold local transforms.

    Args:
        poisson_parquet (object): The poisson parquet.
    """

    class RecordingTuner:
        """Configure RecordingTuner."""

        n_trials = 1
        seed = 17
        metric = None

        def tune(self, data, model, **kwargs):
            """Tune.

            Args:
                data (object): Model data to fit, transform, predict, or evaluate.
                model (object): Model wrapper or fitted model to use.
                kwargs (object): Additional keyword arguments forwarded to the pipeline run.
            """
            assert data.feature_names == ["x1", "x3"]
            assert kwargs["encoder"] is not None
            assert kwargs["selector"] is not None
            return {"n_estimators": 5}, pl.DataFrame({"trial": [0], "value": [1.0]})

    class IdentitySelector:
        """Configure IdentitySelector."""

        def fit(self, data):
            """Fit.

            Args:
                data (object): Model data to fit, transform, predict, or evaluate.
            """
            self.names = list(data.feature_names)
            return self

        def selected_features(self):
            """Selected features."""
            return self.names

    ModelRecipe(
        model=LightGBMModel(),
        encoder=OneHotEncoder(),
        selection=IdentitySelector(),
        tuning=RecordingTuner(),
        selection_scope="fold",
    ).fit(_data(poisson_parquet))


def test_comparison_predictions_are_taken_from_holdout(poisson_raw):
    """Verify comparison predictions are taken from holdout.

    Args:
        poisson_raw (object): The poisson raw.
    """
    data = _data_from_raw = ModelData(
        features=poisson_raw.select(["x1", "x3"]),
        target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"],
        weight=None,
        feature_names=["x1", "x3"],
        objective="poisson",
        comparisons=pl.DataFrame({"legacy": [1.0] * poisson_raw.height}),
    ).validate()
    result = ModelPipeline(
        data=data, recipe=ModelRecipe(model=LightGBMModel(objective="poisson"))
    ).run()
    report = result.evaluate(slice_model_data(data, range(100)))
    assert set(report.metrics()["model"].unique()) == {"GBM", "legacy"}


def test_pipeline_run_can_select_reusable_feature_subset(poisson_raw):
    """Verify pipeline run can select reusable feature subset.

    Args:
        poisson_raw (object): The poisson raw.
    """
    schema = infer_schema(poisson_raw, ["x1", "x2", "x3"])
    data = ModelData(
        features=poisson_raw.select(["x1", "x2", "x3"]),
        target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"],
        weight=None,
        feature_names=["x1", "x2", "x3"],
        schema=schema,
        objective="poisson",
    ).validate()

    result = ModelPipeline(
        data=data,
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            encoder=OneHotEncoder(),
            params={"n_estimators": 5},
        ),
    ).run(feature_names=["x1", "x2"])

    assert result.input_feature_names == ["x1", "x2"]
    assert result.raw_train_data.features.columns == ["x1", "x2"]
    assert result.metadata.input_feature_names == ["x1", "x2"]
    assert result.predict(data).len() == data.n_rows


def test_pipeline_run_can_select_fixed_encoded_features(poisson_raw):
    """Verify pipeline run can select fixed encoded features.

    Args:
        poisson_raw (object): The poisson raw.
    """
    schema = infer_schema(poisson_raw, ["x1", "x2", "x3"])
    data = ModelData(
        features=poisson_raw.select(["x1", "x2", "x3"]),
        target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"],
        weight=None,
        feature_names=["x1", "x2", "x3"],
        schema=schema,
        objective="poisson",
    ).validate()

    result = ModelPipeline(
        data=data,
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            encoder=OneHotEncoder(),
            params={"n_estimators": 5},
        ),
    ).run(
        feature_names=["x1", "x2__A"],
        feature_stage="encoded",
    )

    assert result.input_feature_names == ["x1", "x2", "x3"]
    assert result.raw_train_data is data
    assert result.selected_features == ["x1", "x2__A"]
    assert result.train_data.features.columns == ["x1", "x2__A"]
    assert result.metadata.selected_features == ["x1", "x2__A"]
    assert result.predict(data).len() == data.n_rows


def test_encoded_feature_stage_works_without_encoder(poisson_parquet):
    """Verify encoded feature stage works without encoder.

    Args:
        poisson_parquet (object): The poisson parquet.
    """
    data = _data(poisson_parquet)
    result = ModelPipeline(
        data=data,
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            params={"n_estimators": 5},
        ),
    ).run(feature_names=["x3"], feature_stage="encoded")

    assert result.input_feature_names == ["x1", "x3"]
    assert result.selected_features == ["x3"]
    assert result.train_data.features.columns == ["x3"]


@pytest.mark.parametrize("selection_scope", ["fold", "fixed"])
def test_pruned_model_columns_replay_through_tuning_cv_and_persistence(
    poisson_parquet,
    tmp_path,
    selection_scope,
):
    """Verify pruned model columns replay through tuning cv and persistence.

    Args:
        poisson_parquet (object): The poisson parquet.
        tmp_path (object): The tmp path.
        selection_scope (object): Selection scope: "full_data" or "fold_local".
    """
    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x2", "x3"],
        objective="poisson",
    )
    preprocessing = [
        PreprocessingStep(
            name="x1_pca",
            preprocessor=PCAReducer(n_components=1),
            feature_names=["x1"],
        )
    ]
    recipe = ModelRecipe(
        model=LightGBMModel(objective="poisson"),
        encoder=OneHotEncoder(),
        preprocessing=preprocessing,
        params={"n_estimators": 5, "verbose": -1},
        selection_scope=selection_scope,
    )
    original = recipe.fit(data)
    selection = ImportancePruner(threshold=0).fit(original)
    recipe.tuning = HyperparameterTuner(
        n_trials=1,
        cv_folds=2,
        show_progress_bar=False,
    )
    fitted = recipe.fit(data, feature_names=selection)
    assert fitted.train_data.feature_names == selection.selected_features()
    assert fitted.tuning_history.height == 1

    from ins_gbm.ensemble._utils import _apply_pipeline_recipe_fold_transforms

    fold_train, fold_val = _apply_pipeline_recipe_fold_transforms(
        fitted,
        slice_model_data(data, range(200)),
        slice_model_data(data, range(200, data.n_rows)),
    )
    assert fold_train.feature_names == selection.selected_features()
    assert fold_val.feature_names == selection.selected_features()

    cv = recipe.cross_validate(
        data,
        feature_names=selection,
    )
    assert cv.predictions.height == data.n_rows

    retuned = fitted.retune(
        HyperparameterTuner(
            n_trials=1,
            cv_folds=2,
            show_progress_bar=False,
        )
    )
    assert retuned.train_data.feature_names == selection.selected_features()

    fitted.save(str(tmp_path))
    loaded = load_pipeline(str(tmp_path))
    assert loaded.model_selected_features == selection.selected_features()
    assert loaded.predict(data).to_list() == pytest.approx(
        fitted.predict(data).to_list()
    )


def test_encoded_feature_stage_validates_configuration(poisson_parquet):
    """Verify encoded feature stage validates configuration.

    Args:
        poisson_parquet (object): The poisson parquet.
    """
    data = _data(poisson_parquet)
    recipe = ModelRecipe(model=LightGBMModel(objective="poisson"))

    with pytest.raises(ValueError, match="feature_names is required"):
        ModelPipeline(data=data, recipe=recipe).run(feature_stage="encoded")
    with pytest.raises(ValueError, match="at least one"):
        ModelPipeline(data=data, recipe=recipe).run(
            feature_names=[],
            feature_stage="encoded",
        )
    with pytest.raises(ValueError, match="must be unique"):
        ModelPipeline(data=data, recipe=recipe).run(
            feature_names=["x1", "x1"],
            feature_stage="encoded",
        )
    with pytest.raises(ValueError, match="missing after encoding"):
        ModelPipeline(data=data, recipe=recipe).run(
            feature_names=["not_a_feature"],
            feature_stage="encoded",
        )

    class Selector:
        """Configure Selector."""

        def fit(self, data):
            """Fit.

            Args:
                data (object): Model data to fit, transform, predict, or evaluate.
            """
            return self

        def selected_features(self):
            """Selected features."""
            return ["x1"]

    with pytest.raises(ValueError, match="cannot be combined"):
        ModelPipeline(
            data=data,
            recipe=ModelRecipe(
                model=LightGBMModel(objective="poisson"),
                selection=Selector(),
            ),
        ).run(feature_names=["x1"], feature_stage="encoded")

    with pytest.raises(ValueError, match="feature_stage"):
        ModelPipeline(data=data, recipe=recipe).run(
            feature_names=["x1"],
            feature_stage="invalid",
        )


def test_retune_freezes_encoder_and_selection_and_returns_new_pipeline(poisson_raw):
    """Verify retune freezes encoder and selection and returns new pipeline.

    Args:
        poisson_raw (object): The poisson raw.
    """
    schema = infer_schema(poisson_raw, ["x1", "x2", "x3"])
    data = ModelData(
        features=poisson_raw.select(["x1", "x2", "x3"]),
        target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"],
        weight=None,
        feature_names=["x1", "x2", "x3"],
        schema=schema,
        objective="poisson",
    ).validate()
    preprocessing = [
        PreprocessingStep(
            name="x1_pca",
            preprocessor=PCAReducer(n_components=1),
            feature_names=["x1"],
        ),
    ]
    original = ModelPipeline(
        data=data,
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            encoder=OneHotEncoder(),
            preprocessing=preprocessing,
            params={"n_estimators": 3},
        ),
    ).run(
        feature_names=["x1", "x2__A"],
        feature_stage="encoded",
    )

    class RecordingTuner:
        """Configure RecordingTuner."""

        n_trials = 1
        seed = 73

        def tune(self, tuning_data, model, **kwargs):
            """Tune.

            Args:
                tuning_data (object): Model data used for tuning.
                model (object): Model wrapper or fitted model to use.
                kwargs (object): Additional keyword arguments forwarded to the pipeline run.
            """
            assert tuning_data.features.columns == ["x1", "x2__A"]
            assert kwargs["preprocessors"] is preprocessing
            assert "encoder" not in kwargs
            assert "selector" not in kwargs
            return (
                {"n_estimators": 7, "learning_rate": 0.123},
                pl.DataFrame({"trial": [0], "value": [1.0]}),
            )

    original_model = original.fitted_model
    original_preprocessors = original.preprocessors
    tuned = original.retune(RecordingTuner())

    assert tuned is not original
    assert original.fitted_model is original_model
    assert original.preprocessors is original_preprocessors
    assert original.tuning_history is None
    assert original.recipe.tuning is None

    assert tuned.encoder is original.encoder
    assert tuned.selected_features == original.selected_features
    assert tuned.selection_results is original.selection_results
    assert tuned.raw_train_data is original.raw_train_data
    assert tuned.preprocessors is not original.preprocessors
    assert tuned.fitted_model.params["n_estimators"] == 7
    assert tuned.fitted_model.params["learning_rate"] == 0.123
    assert tuned.tuning_history.height == 1
    assert tuned.recipe.tuning.seed == 73
    assert tuned.metadata.random_seeds["tuning"] == 73
    assert tuned.metadata.model_params == tuned.fitted_model.params
    assert tuned.predict(data).len() == data.n_rows


def test_pipeline_targeted_preprocessor_retains_other_features(poisson_raw):
    """Verify pipeline targeted preprocessor retains other features.

    Args:
        poisson_raw (object): The poisson raw.
    """
    data = ModelData(
        features=poisson_raw.select(["x1", "x3"]),
        target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"],
        weight=None,
        feature_names=["x1", "x3"],
        objective="poisson",
    ).validate()

    result = ModelPipeline(
        data=data,
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"),
            preprocessing=[
                PreprocessingStep(
                    name="x1_pca",
                    preprocessor=PCAReducer(n_components=1),
                    feature_names=["x1"],
                ),
            ],
            params={"n_estimators": 5},
        ),
    ).run()

    assert result.train_data.features.columns == ["x1_pca__pca_1", "x3"]
    assert result.predict(data).len() == data.n_rows
