import numpy as np
import polars as pl
import pytest

from ins_gbm import (
    CVConfig,
    LightGBMModel,
    ModelData,
    ModelRecipe,
    OneHotEncoder,
    PCAReducer,
    PLSReducer,
    PreprocessingStep,
    cv_feature_importance,
)
from ins_gbm.models.base import FittedModel, ModelCapabilities


class RecordingModel:
    """Configure RecordingModel."""

    def __init__(self):
        """Init."""
        self.fits = []

    def capabilities(self):
        """Capabilities."""
        return ModelCapabilities(True, True, True, True, True)

    def fit(self, data, params=None):
        """Fit.

        Args:
            data (object): Model data to fit, transform, predict, or evaluate.
            params (object): Optional model or estimator parameter mapping.
        """
        fold = int(data.target[0])
        self.fits.append((data.target.to_list(), dict(params)))
        values = {
            1: {"weight": {"a": 2, "b": 0}, "gain": {"a": 4, "b": 0}},
            0: {"weight": {"b": 1}, "gain": {"b": 6}},
        }[fold]

        def importance(kind):
            """Importance.

            Args:
                kind (object): The kind.
            """
            return pl.DataFrame(
                {
                    "feature": list(values[kind]),
                    "importance": list(values[kind].values()),
                }
            )

        return FittedModel(
            model=None,
            params=params,
            framework="fake",
            objective="poisson",
            feature_names=data.feature_names,
            predict_fn=None,
            importance_fn=importance,
        )


def test_predefined_folds_include_zero_importance_and_average_all_folds():
    """Verify predefined folds include zero importance and average all folds."""
    data = ModelData(
        features=pl.DataFrame(
            {"a": [1, 2, 3, 4], "b": [1, 1, 1, 1], "c": [0, 0, 0, 0]}
        ),
        target=pl.Series([0, 0, 1, 1]),
        feature_names=["a", "b", "c"],
        cv_fold=pl.Series([10, 10, 20, 20]),
    )
    model = RecordingModel()
    result = cv_feature_importance(
        data, model=model, importance_types=("weight", "gain")
    )

    assert result.columns == ["feature", "n_folds_selected", "mean_weight", "mean_gain"]
    assert result["feature"].to_list() == ["a", "b", "c"]
    assert result["n_folds_selected"].to_list() == [1, 1, 0]
    assert result["mean_weight"].to_list() == [1, 0.5, 0]
    assert result["mean_gain"].to_list() == [2, 3, 0]
    assert [fit[0] for fit in model.fits] == [[1.0, 1.0], [0.0, 0.0]]


def test_random_folds_and_subset_use_requested_columns():
    """Verify random folds and subset use requested columns."""
    data = ModelData(
        features=pl.DataFrame({"a": range(12), "b": range(12), "c": range(12)}),
        target=pl.Series(np.arange(12) % 2),
        feature_names=["a", "b", "c"],
        cv_fold=pl.Series([0] * 6 + [1] * 6),
    )

    class RandomRecordingModel(RecordingModel):
        """Configure RandomRecordingModel."""

        def fit(self, data, params=None):
            """Fit.

            Args:
                data (object): Model data to fit, transform, predict, or evaluate.
                params (object): Optional model or estimator parameter mapping.
            """
            self.fits.append(
                (data.features["c"].to_list(), list(data.feature_names), dict(params))
            )

            def importance(kind):
                """Importance.

                Args:
                    kind (object): The kind.
                """
                return pl.DataFrame(
                    {
                        "feature": data.feature_names,
                        "importance": [1.0] * len(data.feature_names),
                    }
                )

            return FittedModel(
                model=None,
                params=params,
                framework="fake",
                objective="poisson",
                feature_names=data.feature_names,
                predict_fn=None,
                importance_fn=importance,
            )

    model = RandomRecordingModel()
    result = cv_feature_importance(
        data,
        model=model,
        cv=CVConfig(folds="random", n_splits=3, seed=7),
        feature_names=["c", "a"],
        importance_types="gain",
        params={"max_depth": 2},
    )
    assert result["feature"].to_list() == ["c", "a"]
    assert result.columns == ["feature", "n_folds_selected", "mean_gain"]
    assert result["n_folds_selected"].to_list() == [3, 3]
    assert len(model.fits) == 3
    assert all(fit[1:] == (["c", "a"], {"max_depth": 2}) for fit in model.fits)
    assert len({tuple(fit[0]) for fit in model.fits}) == 3


def test_invalid_importance_type_is_reported():
    """Verify invalid importance type is reported."""
    data = ModelData(
        features=pl.DataFrame({"a": [1, 2, 3, 4]}),
        target=pl.Series([0, 1, 0, 1]),
        feature_names=["a"],
    )
    with pytest.raises(ValueError, match="unique"):
        cv_feature_importance(data, importance_types=("gain", "gain"))
    with pytest.raises(ValueError, match="non-empty"):
        cv_feature_importance(data, importance_types=())


@pytest.mark.parametrize("with_preprocessing", [False, True])
def test_encoded_levels_are_ranked_and_can_be_used_for_final_fit(with_preprocessing):
    """Verify encoded levels are ranked and can be used for final fit."""

    class EncodedRecordingModel(RecordingModel):
        """Configure EncodedRecordingModel."""

        def fit(self, data, params=None):
            """Fit.

            Args:
                data (object): Model data to fit, transform, predict, or evaluate.
                params (object): Optional model or estimator parameter mapping.
            """
            self.fits.append(list(data.feature_names))

            def importance(kind):
                """Importance.

                Args:
                    kind (object): The kind.
                """
                return pl.DataFrame(
                    {
                        "feature": data.feature_names,
                        "importance": [1.0] * len(data.feature_names),
                    }
                )

            return FittedModel(
                model=None,
                params=params,
                framework="fake",
                objective="poisson",
                feature_names=data.feature_names,
                predict_fn=None,
                importance_fn=importance,
            )

    data = ModelData(
        features=pl.DataFrame(
            {
                "x": list(range(12)),
                "group": ["A"] * 6 + ["B"] * 6,
            }
        ),
        target=pl.Series([0, 1] * 6),
        feature_names=["x", "group"],
        objective="poisson",
        cv_fold=pl.Series([0] * 6 + [1] * 6),
    )
    model = EncodedRecordingModel()
    preprocessing = (
        [PreprocessingStep("numeric", PCAReducer(1), ["x"])]
        if with_preprocessing else []
    )
    numeric_name = "numeric__pca_1" if with_preprocessing else "x"
    ranking = cv_feature_importance(
        data,
        model=model,
        encoder=OneHotEncoder(),
        preprocessing=preprocessing,
        importance_types="gain",
    )
    assert ranking["feature"].to_list() == [numeric_name, "group__B", "group__A"]
    by_name = {row["feature"]: row for row in ranking.to_dicts()}
    assert by_name[numeric_name]["n_folds_selected"] == 2
    assert by_name["group__A"]["n_folds_selected"] == 1
    assert by_name["group__B"]["mean_gain"] == 0.5
    assert all(
        len([name for name in fit if name.startswith("group__")]) == 1
        for fit in model.fits
    )

    selected = ranking["feature"].to_list()
    if with_preprocessing:
        selected = selected[:2]
    recipe = ModelRecipe(
        model=LightGBMModel(objective="poisson"),
        encoder=OneHotEncoder(),
        preprocessing=preprocessing,
        params={"n_estimators": 3, "min_child_samples": 1},
    )
    fitted = recipe.fit(
        data,
        feature_names=selected,
        feature_stage="model" if with_preprocessing else "encoded",
    )
    assert fitted.train_data.feature_names == selected
    assert fitted.predict(data).len() == data.n_rows


def test_supervised_preprocessing_is_fold_local_and_steps_run_in_order():
    """Fit sequential transforms using only training rows and their targets."""
    data = ModelData(
        features=pl.DataFrame({"a": range(12), "b": [1, 3, 2] * 4, "unused": range(12)}),
        target=pl.Series([0, 2, 1, 3, 2, 4, 5, 3, 6, 4, 7, 5]),
        feature_names=["a", "b", "unused"],
        cv_fold=pl.Series([0] * 6 + [1] * 6),
    )
    fits = []

    class RecordingPLS:
        """Record training inputs while using a real supervised reducer."""

        def fit(self, features, target):
            """Fit PLS on the supplied fold."""
            fits.append((features.clone(), target.to_list()))
            return PLSReducer(1).fit(features, target)

    class TransformedModel(RecordingModel):
        """Assign known importance to each transformed column."""

        def fit(self, data, params=None):
            """Record the model matrix and report unit importance."""
            self.fits.append(data.features.clone())
            return FittedModel(
                model=None,
                params=params,
                framework="fake",
                objective="poisson",
                feature_names=data.feature_names,
                predict_fn=None,
                importance_fn=lambda kind: pl.DataFrame(
                    {
                        "feature": data.feature_names,
                        "importance": [1.0] * len(data.feature_names),
                    }
                ),
            )

    model = TransformedModel()
    ranking = cv_feature_importance(
        data,
        model=model,
        feature_names=["a", "b"],
        importance_types="gain",
        preprocessing=[
            PreprocessingStep("supervised", RecordingPLS(), ["a", "b"]),
            PreprocessingStep("reduced", PCAReducer(1), ["supervised__pls_1"]),
        ],
    )
    assert len(fits) == 2
    for (features, target), start in zip(fits, [6, 0]):
        assert features.equals(data.features.slice(start, 6).select("a", "b"))
        assert target == data.target.slice(start, 6).to_list()
    assert all(frame.columns == ["reduced__pca_1"] for frame in model.fits)
    assert ranking.to_dicts() == [
        {"feature": "reduced__pca_1", "n_folds_selected": 2, "mean_gain": 1.0}
    ]


def test_preprocessing_validation_is_shared():
    """Reject invalid targeted steps before fitting the model."""
    data = ModelData(
        features=pl.DataFrame({"a": range(12)}),
        target=pl.Series([0, 1] * 6),
        feature_names=["a"],
    )
    step = PreprocessingStep("duplicate", PCAReducer(1), ["a"])
    with pytest.raises(ValueError, match="names must be unique"):
        cv_feature_importance(data, preprocessing=[step, step])
    with pytest.raises(ValueError, match="missing features"):
        cv_feature_importance(
            data,
            preprocessing=[PreprocessingStep("missing", PCAReducer(1), ["absent"])],
        )


def test_default_xgboost_importances_when_installed():
    """Verify default xgboost importances when installed."""
    pytest.importorskip("xgboost")
    data = ModelData(
        features=pl.DataFrame({"a": range(12), "b": range(12)}),
        target=pl.Series(np.arange(12) % 2),
        feature_names=["a", "b"],
    )
    result = cv_feature_importance(
        data, cv=CVConfig(n_splits=3), params={"n_estimators": 2}
    )
    assert result.columns == [
        "feature",
        "n_folds_selected",
        "mean_weight",
        "mean_gain",
        "mean_cover",
    ]
