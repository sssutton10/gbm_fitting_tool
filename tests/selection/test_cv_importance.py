import numpy as np
import polars as pl
import pytest

from ins_gbm import (
    CVConfig, LightGBMModel, ModelData, ModelRecipe, OneHotEncoder,
    cv_feature_importance,
)
from ins_gbm.models.base import FittedModel, ModelCapabilities


class RecordingModel:
    def __init__(self):
        self.fits = []

    def capabilities(self):
        return ModelCapabilities(True, True, True, True, True)

    def fit(self, data, params=None):
        fold = int(data.target[0])
        self.fits.append((data.target.to_list(), dict(params)))
        values = {
            1: {"weight": {"a": 2, "b": 0}, "gain": {"a": 4, "b": 0}},
            0: {"weight": {"b": 1}, "gain": {"b": 6}},
        }[fold]

        def importance(kind):
            return pl.DataFrame({
                "feature": list(values[kind]),
                "importance": list(values[kind].values()),
            })

        return FittedModel(
            model=None, params=params, framework="fake", objective="poisson",
            feature_names=data.feature_names, predict_fn=None,
            importance_fn=importance,
        )


def test_predefined_folds_include_zero_importance_and_average_all_folds():
    data = ModelData(
        features=pl.DataFrame({"a": [1, 2, 3, 4], "b": [1, 1, 1, 1], "c": [0, 0, 0, 0]}),
        target=pl.Series([0, 0, 1, 1]), feature_names=["a", "b", "c"],
        cv_fold=pl.Series([10, 10, 20, 20]),
    )
    model = RecordingModel()
    result = cv_feature_importance(data, model=model, importance_types=("weight", "gain"))

    assert result.columns == ["feature", "n_folds_selected", "mean_weight", "mean_gain"]
    assert result["feature"].to_list() == ["a", "b", "c"]
    assert result["n_folds_selected"].to_list() == [1, 1, 0]
    assert result["mean_weight"].to_list() == [1, 0.5, 0]
    assert result["mean_gain"].to_list() == [2, 3, 0]
    assert [fit[0] for fit in model.fits] == [[1.0, 1.0], [0.0, 0.0]]


def test_random_folds_and_subset_use_requested_columns():
    data = ModelData(
        features=pl.DataFrame({"a": range(12), "b": range(12), "c": range(12)}),
        target=pl.Series(np.arange(12) % 2), feature_names=["a", "b", "c"],
        cv_fold=pl.Series([0] * 6 + [1] * 6),
    )
    class RandomRecordingModel(RecordingModel):
        def fit(self, data, params=None):
            self.fits.append((data.features["c"].to_list(), list(data.feature_names), dict(params)))

            def importance(kind):
                return pl.DataFrame({"feature": data.feature_names, "importance": [1.0] * len(data.feature_names)})

            return FittedModel(
                model=None, params=params, framework="fake", objective="poisson",
                feature_names=data.feature_names, predict_fn=None, importance_fn=importance,
            )

    model = RandomRecordingModel()
    result = cv_feature_importance(
        data, model=model,
        cv=CVConfig(folds="random", n_splits=3, seed=7),
        feature_names=["c", "a"], importance_types="gain", params={"max_depth": 2},
    )
    assert result["feature"].to_list() == ["c", "a"]
    assert result.columns == ["feature", "n_folds_selected", "mean_gain"]
    assert result["n_folds_selected"].to_list() == [3, 3]
    assert len(model.fits) == 3
    assert all(fit[1:] == (["c", "a"], {"max_depth": 2}) for fit in model.fits)
    assert len({tuple(fit[0]) for fit in model.fits}) == 3


def test_invalid_importance_type_is_reported():
    data = ModelData(
        features=pl.DataFrame({"a": [1, 2, 3, 4]}),
        target=pl.Series([0, 1, 0, 1]), feature_names=["a"],
    )
    with pytest.raises(ValueError, match="unique"):
        cv_feature_importance(data, importance_types=("gain", "gain"))
    with pytest.raises(ValueError, match="non-empty"):
        cv_feature_importance(data, importance_types=())


def test_encoded_levels_are_ranked_and_can_be_used_for_final_fit():
    class EncodedRecordingModel(RecordingModel):
        def fit(self, data, params=None):
            self.fits.append(list(data.feature_names))

            def importance(kind):
                return pl.DataFrame({
                    "feature": data.feature_names,
                    "importance": [1.0] * len(data.feature_names),
                })

            return FittedModel(
                model=None, params=params, framework="fake", objective="poisson",
                feature_names=data.feature_names, predict_fn=None,
                importance_fn=importance,
            )

    data = ModelData(
        features=pl.DataFrame({
            "x": list(range(12)),
            "group": ["A"] * 6 + ["B"] * 6,
        }),
        target=pl.Series([0, 1] * 6),
        feature_names=["x", "group"],
        objective="poisson",
        cv_fold=pl.Series([0] * 6 + [1] * 6),
    )
    model = EncodedRecordingModel()
    ranking = cv_feature_importance(
        data, model=model, encoder=OneHotEncoder(), importance_types="gain",
    )
    assert set(ranking["feature"].to_list()) == {"x", "group__A", "group__B"}
    by_name = {row["feature"]: row for row in ranking.to_dicts()}
    assert by_name["x"]["n_folds_selected"] == 2
    assert by_name["group__A"]["n_folds_selected"] == 1
    assert by_name["group__B"]["mean_gain"] == 0.5
    assert all(len([name for name in fit if name.startswith("group__")]) == 1 for fit in model.fits)

    selected = ranking["feature"].to_list()
    recipe = ModelRecipe(
        model=LightGBMModel(objective="poisson"), encoder=OneHotEncoder(),
        params={"n_estimators": 3, "min_child_samples": 1},
    )
    fitted = recipe.fit(data, feature_names=selected, feature_stage="encoded")
    assert fitted.train_data.feature_names == selected
    assert fitted.predict(data).len() == data.n_rows


def test_default_xgboost_importances_when_installed():
    pytest.importorskip("xgboost")
    data = ModelData(
        features=pl.DataFrame({"a": range(12), "b": range(12)}),
        target=pl.Series(np.arange(12) % 2), feature_names=["a", "b"],
    )
    result = cv_feature_importance(data, cv=CVConfig(n_splits=3), params={"n_estimators": 2})
    assert result.columns == [
        "feature", "n_folds_selected", "mean_weight", "mean_gain", "mean_cover",
    ]
