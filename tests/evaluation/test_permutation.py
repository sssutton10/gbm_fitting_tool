from dataclasses import dataclass, replace

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ins_gbm import (
    CVConfig, ModelData, ModelRecipe, OneHotEncoder, PCAReducer,
    PermutationImportanceResult,
)
from ins_gbm.evaluation.metrics import compute_metrics
from ins_gbm.models.base import FittedModel


@dataclass
class FixedModel:
    """Predict a known relationship so exact scoring and signs are testable."""

    objective: str = "poisson"
    column: str = "signal"

    def fit(self, data, params=None):
        def predict(held, prediction_type):
            result = held.features[self.column].exp()
            if held.offset is not None:
                result = result * held.offset.exp()
            if self.objective == "poisson" and held.exposure is not None:
                result = result * held.exposure
            return result

        return FittedModel(
            model=None, params=dict(params or {}), framework="fixed",
            objective=self.objective, feature_names=list(data.feature_names),
            predict_fn=predict, importance_fn=lambda: None,
        )


def make_data(objective="poisson"):
    x = np.tile(np.linspace(-1, 1, 20), 3)
    exposure = pl.Series("exposure", np.linspace(0.5, 2, len(x)))
    offset = pl.Series("offset", np.linspace(-0.1, 0.1, len(x)))
    y = np.exp(x + offset.to_numpy())
    if objective == "poisson":
        y *= exposure.to_numpy()
    return ModelData(
        features=pl.DataFrame({"signal": x, "unused": x[::-1]}),
        target=pl.Series("y", y), feature_names=["signal", "unused"],
        objective=objective, exposure=exposure if objective == "poisson" else None,
        weight=pl.Series("weight", np.linspace(0.5, 1.5, len(x))), offset=offset,
        cv_fold=pl.Series("fold", [10] * 20 + [20] * 20 + [30] * 20),
    )


@pytest.mark.parametrize("objective", ["poisson", "gamma"])
def test_frozen_fits_scoring_signs_and_immutability(objective, monkeypatch):
    data = make_data(objective)
    fitted = ModelRecipe(model=FixedModel(objective), params={"chosen": 17}).fit(data)
    before = fitted.predict(data)
    original_features = data.features.clone()
    received, predictions = [], []
    original_fit = FixedModel.fit

    def record(self, train, params=None):
        received.append((train, params))
        result = original_fit(self, train, params)
        original_predict = result.predict_fn

        def capture(held, prediction_type):
            predictions.append(held)
            return original_predict(held, prediction_type)

        result.predict_fn = capture
        return result

    class MustNotRun:
        def __getattr__(self, name):
            raise AssertionError("Selection/tuning must not be accessed")

    monkeypatch.setattr(FixedModel, "fit", record)
    fitted.recipe.selection = MustNotRun()
    fitted.recipe.tuning = MustNotRun()
    fitted.recipe.params = {"chosen": 999}  # fitted params are the source of truth
    result = fitted.permutation_importance_cv(data, n_repeats=3, show_progress_bar=False)
    assert isinstance(result, PermutationImportanceResult)
    assert len(received) == 3
    assert [params for _, params in received] == [{"chosen": 17}] * 3
    assert all(train.n_rows == 40 for train, _ in received)
    assert len(predictions) == 3 * (1 + 2 * 3)
    assert result.baseline_scores["fold"].to_list() == [10, 20, 30]
    assert result.settings["cv"]["folds"] == "predefined"
    assert result.summary["feature"][0] == "signal"
    assert result.summary["mean_importance"][0] > 0
    unused = result.summary.filter(pl.col("feature") == "unused")
    assert unused["mean_importance"][0] == 0
    for i, (_, params) in enumerate(received):
        baseline = predictions[i * 7]
        train = received[i][0]
        assert set(baseline.cv_fold.to_list()).isdisjoint(train.cv_fold.to_list())
        score = compute_metrics(
            objective=objective, actual=baseline.target,
                predicted=original_fit(FixedModel(objective), train).predict(baseline),
            exposure=baseline.exposure, weight=baseline.weight,
        ).filter(pl.col("metric") == f"{objective}_deviance")["value"][0]
        assert result.baseline_scores["baseline_score"][i] == pytest.approx(score)
        for permuted in predictions[i * 7 + 1:i * 7 + 7]:
            for name in ("target", "exposure", "offset", "weight", "cv_fold"):
                a, b = getattr(baseline, name), getattr(permuted, name)
                assert a is None and b is None or a.equals(b)
        # Independently check every shuffled score against the existing evaluator.
        for j, permuted in enumerate(predictions[i * 7 + 1:i * 7 + 7]):
            score = compute_metrics(
                objective=objective, actual=permuted.target,
                predicted=original_fit(FixedModel(objective), train).predict(permuted),
                exposure=permuted.exposure, weight=permuted.weight,
            ).filter(pl.col("metric") == f"{objective}_deviance")["value"][0]
            row = result.repeat_importances.row(i * 6 + j, named=True)
            assert row["permuted_score"] == pytest.approx(score)
            assert row["importance"] == pytest.approx(score - row["baseline_score"])
    assert_frame_equal(data.features, original_features)
    assert fitted.predict(data).equals(before)
    assert result.settings["model_params"] == {"chosen": 17}


def test_sampling_random_folds_reproducibility_and_summary():
    data = make_data()
    fitted = ModelRecipe(model=FixedModel()).fit(data)
    kwargs = dict(cv=CVConfig(folds="random", n_splits=4, seed=7),
                  max_samples=8, n_repeats=2, seed=9, show_progress_bar=False)
    first = fitted.permutation_importance_cv(data, **kwargs)
    second = fitted.permutation_importance_cv(data, **kwargs)
    assert_frame_equal(first.repeat_importances, second.repeat_importances)
    assert first.baseline_scores["n_evaluated"].to_list() == [8] * 4
    assert first.baseline_scores["n_held"].sum() == data.n_rows
    assert first.settings["cv"] == {"folds": "random", "n_splits": 4, "seed": 7}
    for row in first.summary.iter_rows(named=True):
        values = first.fold_importances.filter(pl.col("feature") == row["feature"])
        means = values["mean_importance"].to_numpy()
        assert row["mean_importance"] == pytest.approx(means.mean())
        assert row["std_importance"] == pytest.approx(means.std())
        assert row["fraction_positive_folds"] == pytest.approx((means > 0).mean())


def test_negative_importance_is_preserved():
    data = make_data()
    data = replace(data, target=data.target.reverse())
    fitted = ModelRecipe(model=FixedModel()).fit(data)
    result = fitted.permutation_importance_cv(data, n_repeats=5, show_progress_bar=False)
    assert result.summary.filter(pl.col("feature") == "signal")["mean_importance"][0] < 0


@pytest.mark.parametrize("kwargs", [
    {"n_repeats": 0}, {"n_repeats": True}, {"n_repeats": 1.5},
    {"max_samples": 0}, {"max_samples": 0.5}, {"max_samples": True},
    {"seed": -1}, {"seed": True},
])
def test_invalid_options(kwargs):
    data = make_data()
    fitted = ModelRecipe(model=FixedModel()).fit(data)
    with pytest.raises(ValueError):
        fitted.permutation_importance_cv(data, **kwargs)


def test_objective_and_missing_predefined_folds():
    data = make_data()
    fitted = ModelRecipe(model=FixedModel()).fit(data)
    with pytest.raises(ValueError, match="objective"):
        fitted.permutation_importance_cv(replace(data, objective="gamma"))
    with pytest.raises(ValueError, match="predefined"):
        fitted.permutation_importance_cv(
            replace(data, cv_fold=None), cv=CVConfig(folds="predefined")
        )


def test_encoded_and_model_selections_and_fold_local_transforms(monkeypatch):
    from ins_gbm.models.lightgbm import LightGBMModel

    data = make_data()
    data = data.with_features(data.features.with_columns(
        pl.Series("category", ["a", "b"] * 30)
    ))
    data = replace(data, schema=None)
    fitted = ModelRecipe(
        model=LightGBMModel(), encoder=OneHotEncoder(),
        params={"n_estimators": 3, "num_threads": 1},
        preprocessing=[PCAReducer(n_components=1)],
    ).fit(data, feature_names=["signal", "category__a"], feature_stage="encoded")
    # Exercise simultaneous frozen selections before and after preprocessing.
    fitted.model_selected_features = ["pca_1"]
    model_names, encoder_rows, prep_rows = [], [], []
    original_fit = LightGBMModel.fit
    original_encoder = OneHotEncoder.fit
    original_pca = PCAReducer.fit

    def record_fit(self, train, params=None, **kwargs):
        model_names.append(train.feature_names)
        return original_fit(self, train, params=params, **kwargs)

    def record_encoder(self, features, schema):
        encoder_rows.append(features.height)
        return original_encoder(self, features, schema)

    def record_pca(self, features, target=None):
        prep_rows.append((features.height, features.columns, len(target)))
        return original_pca(self, features, target)

    monkeypatch.setattr(LightGBMModel, "fit", record_fit)
    monkeypatch.setattr(OneHotEncoder, "fit", record_encoder)
    monkeypatch.setattr(PCAReducer, "fit", record_pca)
    fitted.raw_train_data = None  # loaded artifacts need no attached training rows
    result = fitted.permutation_importance_cv(data, n_repeats=1, show_progress_bar=False)
    assert model_names == [["pca_1"]] * 3
    assert encoder_rows == [40] * 3
    assert prep_rows == [(40, ["signal", "category__a"], 40)] * 3
    assert "category" in result.summary["feature"]
    assert result.settings["feature_names"] == ["signal", "category"]
    assert result.repeat_importances.height == 6
    assert result.fold_importances["repeat_std"].to_list() == [0.0] * 6


def test_missing_frozen_encoded_level_is_explicit():
    data = make_data().with_features(pl.DataFrame({"category": ["a"] * 20 + ["b"] * 40}))
    data = replace(data, schema=None)
    fitted = ModelRecipe(model=FixedModel(column="category__a"), encoder=OneHotEncoder()).fit(
        data, feature_names=["category__a"], feature_stage="encoded"
    )
    with pytest.raises(ValueError, match="Frozen selected features missing"):
        fitted.permutation_importance_cv(data, show_progress_bar=False)


def test_raw_subset_and_categorical_grouping():
    data = make_data()
    features = data.features.with_columns(pl.Series("category", ["a", "b"] * 30))
    target = pl.Series("y", np.exp(np.array([1, 0] * 30, dtype=float)))
    data = replace(data, features=features, feature_names=features.columns,
                   target=target, schema=None, exposure=None, offset=None)
    fitted = ModelRecipe(
        model=FixedModel(column="category__a"), encoder=OneHotEncoder(),
    ).fit(data, feature_names=["category"])
    result = fitted.permutation_importance_cv(data, n_repeats=2, show_progress_bar=False)
    assert result.summary["feature"].to_list() == ["category"]
    assert result.summary["mean_importance"][0] > 0
    assert result.baseline_scores["baseline_score"].max() == pytest.approx(0)


def test_zero_weight_validation_sample_has_clear_error():
    data = make_data()
    data = replace(data, weight=pl.Series("w", [0.0] * 20 + [1.0] * 40))
    fitted = ModelRecipe(model=FixedModel()).fit(data)
    with pytest.raises(ValueError, match="weight must have a positive total"):
        fitted.permutation_importance_cv(data, show_progress_bar=False)


@pytest.mark.parametrize('stage', ['encoded', 'model'])
def test_only_selected_raw_sources_are_shuffled(stage, monkeypatch):
    data = make_data()
    data = replace(data, features=data.features.with_columns(
        pl.Series('category', ['a', 'b'] * 30),
    ), feature_names=['signal', 'unused', 'category'], schema=None)
    fitted = ModelRecipe(FixedModel(column='category__a'), encoder=OneHotEncoder()).fit(
        data, feature_names=['category__a'], feature_stage=stage,
    )
    original_fit = FixedModel.fit
    predictions = []

    def record(self, train, params=None):
        model = original_fit(self, train, params)
        original_predict = model.predict_fn

        def predict(held, prediction_type):
            predictions.append(held)
            return original_predict(held, prediction_type)

        model.predict_fn = predict
        return model

    monkeypatch.setattr(FixedModel, 'fit', record)
    result = fitted.permutation_importance_cv(data, n_repeats=2, show_progress_bar=False)
    assert result.settings['feature_names'] == ['category']
    assert result.summary['feature'].to_list() == ['category']
    assert result.repeat_importances.height == 6
    assert len(predictions) == 9  # baseline + two shuffles per fold
    assert all(held.feature_names == ['category__a'] for held in predictions)


@pytest.mark.parametrize(('selected', 'expected'), [
    (['category__a'], ['category']),
    (['group__pca_1'], ['signal', 'unused']),
])
def test_targeted_preprocessing_traces_only_retained_outputs(selected, expected):
    from ins_gbm import LightGBMModel, PreprocessingStep

    data = make_data().with_features(make_data().features.with_columns(
        pl.Series('category', ['a', 'b'] * 30),
    ))
    data = replace(data, schema=None)
    fitted = ModelRecipe(
        LightGBMModel(), encoder=OneHotEncoder(),
        preprocessing=[PreprocessingStep('group', PCAReducer(1), ['signal', 'unused'])],
        params={'n_estimators': 3, 'num_threads': 1, 'verbosity': -1},
    ).fit(data, feature_names=selected, feature_stage='model')
    result = fitted.permutation_importance_cv(data, n_repeats=1, show_progress_bar=False)
    assert result.settings['feature_names'] == expected
    assert set(result.summary['feature']) == set(expected)
    assert result.repeat_importances.height == 3 * len(expected)
