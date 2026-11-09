from dataclasses import dataclass, replace

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ins_gbm import (
    CVConfig, CatBoostModel, LightGBMModel, ModelData, ModelRecipe,
    NoiseScreeningResult, OneHotEncoder, PCAReducer, RandomForestModel, XGBoostModel,
)
from ins_gbm.models.base import FittedModel, ModelCapabilities


@dataclass
class ScoredModel:
    """Deterministic scores make exact shadow statistics independently testable."""

    objective: str = "poisson"

    def capabilities(self):
        return ModelCapabilities(True, True, True, True, True)

    def fit(self, data, params=None):
        def importance(kind=None):
            values = [1.0 if name.startswith("__noise_shadow_") else 5.0
                      for name in data.feature_names]
            return pl.DataFrame({"feature": data.feature_names, "importance": values})

        return FittedModel(
            model=None, params=dict(params or {}), framework="lightgbm",
            objective=self.objective, feature_names=list(data.feature_names),
            predict_fn=lambda held, kind: pl.Series(np.ones(held.n_rows)),
            importance_fn=importance,
        )


def make_data():
    rng = np.random.default_rng(4)
    x = rng.normal(size=90)
    return ModelData(
        features=pl.DataFrame({"x": x, "other": rng.normal(size=90)}),
        target=pl.Series(np.exp(x)), feature_names=["x", "other"], objective="poisson",
        exposure=pl.Series(np.linspace(0.5, 1.5, 90)),
        offset=pl.Series(np.linspace(-0.2, 0.2, 90)),
        weight=pl.Series(np.linspace(1.0, 2.0, 90)),
        cv_fold=pl.Series([4] * 30 + [8] * 30 + [12] * 30),
    )


def test_frozen_specification_exact_scores_and_no_prediction(monkeypatch):
    data = make_data()
    before = data.features.clone()
    fitted = ModelRecipe(model=ScoredModel(), params={"chosen": 7}).fit(
        data, feature_names=["x"]
    )
    fitted.raw_train_data = None
    fitted.recipe.params = {"chosen": 999}

    class MustNotRun:
        def __getattr__(self, name):
            raise AssertionError("Selection/tuning must not be accessed")

    fitted.recipe.selection = MustNotRun()
    fitted.recipe.tuning = MustNotRun()
    received, kinds = [], []
    original_fit = ScoredModel.fit

    def record(self, train, params=None):
        received.append((train, params))
        result = original_fit(self, train, params)
        callback = result.importance_fn

        def importance(kind=None):
            kinds.append(kind)
            return callback(kind)

        def predict(*args):
            raise AssertionError("Native importance should require no predictions")

        result.importance_fn = importance
        result.predict_fn = predict
        return result

    monkeypatch.setattr(ScoredModel, "fit", record)
    result = fitted.noise_screening_cv(data, n_shadow_repeats=2, show_progress_bar=False)
    assert isinstance(result, NoiseScreeningResult)
    assert len(received) == 6
    assert kinds == ["gain"] * 6
    assert result.settings["feature_groups"] == {"x": ["x"]}
    assert result.fit_details["fold"].to_list() == [4, 4, 8, 8, 12, 12]
    for i, (train, params) in enumerate(received):
        assert params == {"chosen": 7}
        assert train.feature_names == ["x", "__noise_shadow_0"]
        assert train.n_rows == 60
        assert [4, 8, 12][i // 2] not in train.cv_fold
        expected = data.features.filter(data.cv_fold != [4, 8, 12][i // 2])["x"]
        assert train.features["x"].equals(expected)
        assert train.features["x"].sort().to_list() == train.features["__noise_shadow_0"].sort().to_list()
        idx = np.where(data.cv_fold.to_numpy() != [4, 8, 12][i // 2])[0]
        for name in ("target", "exposure", "offset", "weight"):
            assert getattr(train, name).equals(getattr(data, name).gather(idx))
    row = result.summary.row(0, named=True)
    assert row["mean_importance"] == 5
    assert row["mean_shadow_importance"] == 1
    assert row["mean_importance_share"] == pytest.approx(5 / 6)
    assert row["mean_excess_over_shadow"] == pytest.approx(4 / 6)
    assert row["fraction_beating_shadow"] == 1
    assert row["fraction_beating_max_shadow"] == 1
    assert row["std_importance_share"] == 0
    assert_frame_equal(before, data.features)


def test_one_hot_columns_are_grouped_and_shuffled_together(monkeypatch):
    data = make_data()
    data = replace(data, features=pl.DataFrame({"cat": ["a", "b", "c"] * 30}),
                   feature_names=["cat"], schema=None)
    fitted = ModelRecipe(model=ScoredModel(), encoder=OneHotEncoder()).fit(data)
    original_fit = ScoredModel.fit

    def record(self, train, params=None):
        real = train.features.select("cat__a", "cat__b", "cat__c")
        shadow = train.features.select("__noise_shadow_0", "__noise_shadow_1", "__noise_shadow_2")
        assert shadow.sum_horizontal().to_list() == [1.0] * train.n_rows
        assert sorted(real.rows()) == sorted(shadow.rows())
        return original_fit(self, train, params)

    monkeypatch.setattr(ScoredModel, "fit", record)
    result = fitted.noise_screening_cv(data, n_shadow_repeats=1, show_progress_bar=False)
    assert result.summary["feature"].to_list() == ["cat"]
    assert result.settings["feature_stage"] == "raw"
    assert result.summary["mean_importance"][0] == 15
    assert result.summary["mean_shadow_importance"][0] == 3


def test_fold_local_preprocessing_and_frozen_component_selection(monkeypatch):
    data = make_data()
    fitted = ModelRecipe(model=ScoredModel(), preprocessing=[PCAReducer(2)]).fit(
        data, feature_names=["pca_1"], feature_stage="model"
    )
    sizes = []
    original = PCAReducer.fit

    def record(self, features, target=None):
        sizes.append((features.height, len(target)))
        return original(self, features, target)

    monkeypatch.setattr(PCAReducer, "fit", record)
    result = fitted.noise_screening_cv(data, n_shadow_repeats=2, show_progress_bar=False)
    assert sizes == [(60, 60)] * 3
    assert result.settings["feature_stage"] == "model"
    assert result.summary["feature"].to_list() == ["pca_1"]


@pytest.mark.parametrize("wrapper,params,default,override", [
    (LightGBMModel, {"n_estimators": 4, "num_threads": 1}, "gain", "split"),
    (XGBoostModel, {"n_estimators": 4, "nthread": 1}, "total_gain", "weight"),
    (CatBoostModel, {"iterations": 4, "thread_count": 1}, "PredictionValuesChange", "LossFunctionChange"),
    (RandomForestModel, {"n_estimators": 4, "n_jobs": 1}, "impurity", "impurity"),
])
def test_framework_defaults_and_explicit_overrides(wrapper, params, default, override):
    data = make_data()
    if wrapper is RandomForestModel:
        data = replace(data, offset=None)
    fitted = ModelRecipe(model=wrapper(), params=params).fit(data)
    first = fitted.noise_screening_cv(data, n_shadow_repeats=1, show_progress_bar=False)
    second = fitted.noise_screening_cv(data, importance_type=override,
                                     n_shadow_repeats=1, show_progress_bar=False)
    assert first.settings["importance_type"] == default
    assert second.settings["importance_type"] == override
    assert first.summary.height == 2
    assert np.isfinite(first.summary["mean_importance_share"].to_numpy()).all()


@pytest.mark.parametrize("wrapper,params", [
    (LightGBMModel, {"n_estimators": 3, "num_threads": 1}),
    (CatBoostModel, {"iterations": 3, "thread_count": 1}),
])
def test_native_categorical_shadows_with_explicit_category_lists(wrapper, params):
    data = make_data()
    features = data.features.with_columns(pl.Series("cat", ["a", "b", "c"] * 30))
    data = replace(data, features=features, feature_names=features.columns, schema=None)
    fitted = ModelRecipe(model=wrapper(categorical_features=["cat"]), params=params).fit(data)
    result = fitted.noise_screening_cv(data, n_shadow_repeats=1, show_progress_bar=False)
    assert result.summary.height == 3
    assert result.settings["feature_groups"]["cat"] == ["cat"]


def test_random_folds_reproducibility_and_fresh_shadows(monkeypatch):
    data = make_data()
    fitted = ModelRecipe(model=ScoredModel()).fit(data)
    draws = []
    original = ScoredModel.fit

    def record(self, train, params=None):
        draws.append(train.features.clone())
        return original(self, train, params)

    monkeypatch.setattr(ScoredModel, "fit", record)
    options = dict(cv=CVConfig(n_splits=2, folds="random", seed=8), seed=9,
                   n_shadow_repeats=2, show_progress_bar=False)
    first = fitted.noise_screening_cv(data, **options)
    second = fitted.noise_screening_cv(data, **options)
    assert_frame_equal(first.fit_importances, second.fit_importances)
    for a, b in zip(draws[:4], draws[4:]):
        assert_frame_equal(a, b)
    assert not draws[0]["__noise_shadow_0"].equals(draws[1]["__noise_shadow_0"])
    assert first.settings["cv"] == {"folds": "random", "n_splits": 2, "seed": 8}


@pytest.mark.parametrize("kwargs", [
    {"n_shadow_repeats": 0}, {"n_shadow_repeats": True}, {"n_shadow_repeats": 1.5},
    {"seed": -1}, {"importance_type": "unknown"}, {"importance_type": ""},
])
def test_invalid_options_fail_before_fitting(kwargs, monkeypatch):
    data = make_data()
    fitted = ModelRecipe(model=ScoredModel()).fit(data)

    def fail(*args, **kwargs):
        raise AssertionError("Should validate before fitting")

    monkeypatch.setattr(ScoredModel, "fit", fail)
    with pytest.raises(ValueError):
        fitted.noise_screening_cv(data, **kwargs)


@pytest.mark.parametrize("score", [0.0, -1.0])
def test_zero_and_signed_scores_and_ties(score, monkeypatch):
    data = make_data()
    fitted = ModelRecipe(model=ScoredModel()).fit(data)
    original = ScoredModel.fit

    def fit(self, train, params=None):
        model = original(self, train, params)
        model.importance_fn = lambda kind: pl.DataFrame({
            "feature": train.feature_names, "importance": [score] * len(train.feature_names),
        })
        return model

    monkeypatch.setattr(ScoredModel, "fit", fit)
    result = fitted.noise_screening_cv(data, n_shadow_repeats=1, show_progress_bar=False)
    assert result.summary["fraction_beating_shadow"].to_list() == [0.0, 0.0]
    assert result.summary["mean_excess_over_max_shadow"].to_list() == [0.0, 0.0]
    assert np.isfinite(result.summary["mean_importance_share"].to_numpy()).all()


@pytest.mark.parametrize("case", ["omitted", "nonfinite", "duplicate", "unknown"])
def test_backend_importance_validation(case, monkeypatch):
    data = make_data()
    fitted = ModelRecipe(model=ScoredModel()).fit(data)
    original = ScoredModel.fit

    def fit(self, train, params=None):
        model = original(self, train, params)
        names, scores = ["x"], [5.0]
        if case == "nonfinite":
            scores = [float("nan")]
        elif case == "duplicate":
            names, scores = ["x", "x"], [1.0, 2.0]
        elif case == "unknown":
            names = ["unrecognized"]
        model.importance_fn = lambda kind: pl.DataFrame({"feature": names, "importance": scores})
        return model

    monkeypatch.setattr(ScoredModel, "fit", fit)
    if case == "omitted":
        result = fitted.noise_screening_cv(data, n_shadow_repeats=1, show_progress_bar=False)
        other = result.summary.filter(pl.col("feature") == "other")
        assert other["mean_importance"][0] == 0
        assert other["fraction_beating_shadow"][0] == 0
    else:
        with pytest.raises(ValueError, match="Feature importance"):
            fitted.noise_screening_cv(data, n_shadow_repeats=1, show_progress_bar=False)


def test_missing_frozen_columns_and_fold_configuration():
    data = make_data()
    data = replace(data, features=pl.DataFrame({"cat": ["a"] * 30 + ["b"] * 60}),
                   feature_names=["cat"], schema=None)
    fitted = ModelRecipe(model=ScoredModel(), encoder=OneHotEncoder()).fit(data)
    with pytest.raises(ValueError, match="Frozen model features missing"):
        fitted.noise_screening_cv(data, n_shadow_repeats=1, show_progress_bar=False)
    with pytest.raises(ValueError, match="predefined"):
        fitted.noise_screening_cv(replace(data, cv_fold=None), cv=CVConfig(folds="predefined"))
    with pytest.raises(ValueError, match="objective"):
        fitted.noise_screening_cv(replace(data, objective="gamma"))
