"""Contracts for raw-input permutation of the final ensemble."""

from dataclasses import dataclass, replace

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ins_gbm import (
    EnsembleResult, HoldoutPermutationImportanceResult, ModelData,
    ModelRecipe, OneHotEncoder,
)
from ins_gbm.ensemble.blending import FittedBlendingEnsemble
from ins_gbm.ensemble.stacking import FittedStackingEnsemble
from ins_gbm.evaluation.metrics import compute_metrics
from ins_gbm.models.base import FittedModel


@dataclass
class CategoryModel:
    objective: str
    encoded: bool = False
    multiplier: float = 1.0

    def fit(self, data, params=None):
        def predict(held, prediction_type):
            category = held.features['category__b'] if self.encoded else (
                held.features['category'] == 'b'
            ).cast(pl.Float64)
            result = (1 + 3 * category) * self.multiplier
            if held.offset is not None:
                result = result * held.offset.exp()
            if self.objective == 'poisson' and held.exposure is not None:
                result = result * held.exposure
            return result

        return FittedModel(None, {}, 'category', self.objective,
                           data.feature_names, predict, lambda *_: pl.DataFrame())


@dataclass
class NonlinearStacker:
    def predict(self, predictions):
        return np.sqrt(predictions[:, 0] * predictions[:, 1])


def fixture(objective, method):
    category = np.tile(['a', 'b'], 20)
    data = ModelData(
        pl.DataFrame({'category': category, 'unused': np.arange(40, dtype=float)}),
        pl.Series(np.where(category == 'b', 4., 1.)), ['category', 'unused'],
        objective=objective, offset=pl.Series(np.linspace(-.1, .1, 40)),
        weight=pl.Series(np.linspace(.5, 1.5, 40)),
        exposure=pl.Series(np.linspace(.4, 2., 40)) if objective == 'poisson' else None,
        cv_fold=pl.Series([0] * 40),  # a holdout need not span training folds
    )
    native = ModelRecipe(CategoryModel(objective)).fit(replace(data, cv_fold=None))
    encoded = ModelRecipe(CategoryModel(objective, encoded=True, multiplier=2),
                          encoder=OneHotEncoder()).fit(replace(data, cv_fold=None))
    members = [native, encoded]
    ensemble = FittedBlendingEnsemble([.3, .7], members) if method == 'blending' else (
        FittedStackingEnsemble(NonlinearStacker(), members, None)
    )
    result = EnsembleResult(ensemble, members)
    return data, result


@pytest.mark.parametrize('objective', ['poisson', 'gamma'])
@pytest.mark.parametrize('method', ['blending', 'stacking'])
def test_shared_raw_shuffle_and_final_predictions(objective, method, monkeypatch):
    data, ensemble = fixture(objective, method)
    before = data.features.clone()
    before_predictions = ensemble.predict(data)
    seen = [[], []]
    for index, member in enumerate(ensemble.base_pipelines):
        original = member.fitted_model.predict_fn

        def capture(held, prediction_type, original=original, index=index):
            seen[index].append(held)
            return original(held, prediction_type)

        monkeypatch.setattr(member.fitted_model, 'predict_fn', capture)

    def no_fit(*args, **kwargs):
        raise AssertionError('Permutation importance must not refit')

    monkeypatch.setattr(CategoryModel, 'fit', no_fit)
    monkeypatch.setattr(OneHotEncoder, 'fit', no_fit)
    report = ensemble.permutation_importance(data, n_repeats=3, show_progress_bar=False)
    assert isinstance(report, HoldoutPermutationImportanceResult)
    assert len(seen[0]) == len(seen[1]) == 7
    assert report.summary['feature'].to_list() == ['category', 'unused']
    assert report.summary['mean_importance'][0] > 0
    assert report.summary['mean_importance'][1] == 0
    for native, encoded in zip(*seen):
        expected = (native.features['category'] == 'b').cast(pl.Float64)
        assert encoded.features['category__b'].equals(expected.rename('category__b'))
        assert (encoded.features['category__a'] + encoded.features['category__b'] == 1).all()
        for name in ['target', 'exposure', 'weight', 'offset', 'cv_fold']:
            a, b = getattr(native, name), getattr(data, name)
            assert (a is None and b is None) or a.equals(b)
    # Independently score the full blend/stacker for each captured raw shuffle.
    snapshots = list(seen[0])
    for index, held in enumerate(snapshots):
        score = compute_metrics(
            objective=objective, actual=held.target, predicted=ensemble.predict(held),
            exposure=held.exposure, weight=held.weight,
        ).filter(pl.col('metric') == f'{objective}_deviance')['value'][0]
        if index == 0:
            assert report.baseline_score == pytest.approx(score)
        else:
            row = report.repeat_importances.row(index - 1, named=True)
            assert row['permuted_score'] == pytest.approx(score)
            assert row['importance'] == pytest.approx(score - report.baseline_score)
    assert_frame_equal(data.features, before)
    assert ensemble.predict(data).equals(before_predictions)


def test_sampling_reproducibility_summary_and_different_member_inputs():
    data, ensemble = fixture('poisson', 'blending')
    ensemble.base_pipelines[0].input_feature_names = ['category']
    options = dict(max_samples=12, n_repeats=4, seed=7, show_progress_bar=False)
    first = ensemble.permutation_importance(data, **options)
    second = ensemble.permutation_importance(data, **options)
    assert_frame_equal(first.repeat_importances, second.repeat_importances)
    assert first.settings['n_evaluated'] == 12
    assert first.settings['n_held'] == 40
    assert first.settings['feature_names'] == ['category', 'unused']
    for row in first.summary.iter_rows(named=True):
        values = first.repeat_importances.filter(pl.col('feature') == row['feature'])['importance'].to_numpy()
        assert row['mean_importance'] == pytest.approx(values.mean())
        assert row['std_importance'] == pytest.approx(values.std())
    single = ensemble.permutation_importance(data, n_repeats=1, show_progress_bar=False)
    assert single.summary['std_importance'].to_list() == [0., 0.]


@pytest.mark.parametrize('options', [
    {'n_repeats': 0}, {'n_repeats': True}, {'n_repeats': 1.5},
    {'max_samples': 0}, {'max_samples': True}, {'max_samples': .5},
    {'seed': -1}, {'seed': True},
])
def test_invalid_options(options):
    data, ensemble = fixture('poisson', 'blending')
    with pytest.raises(ValueError):
        ensemble.permutation_importance(data, **options)


def test_objective_validation_and_negative_importance():
    data, ensemble = fixture('poisson', 'blending')
    with pytest.raises(ValueError, match='objective'):
        ensemble.permutation_importance(replace(data, objective='gamma'))
    bad_target = pl.Series(np.where(data.features['category'].to_numpy() == 'b', 1., 10.))
    report = ensemble.permutation_importance(replace(data, target=bad_target, objective=None),
                                            n_repeats=5, show_progress_bar=False)
    assert report.summary.filter(pl.col('feature') == 'category')['mean_importance'][0] < 0


@pytest.mark.parametrize('method', ['blending', 'stacking', 'seed_average', 'bagging'])
def test_loaded_backend_ensemble(method, tmp_path):
    from ins_gbm import EnsemblePipeline, LightGBMModel, load_ensemble

    data, _ = fixture('poisson', 'blending')
    data = replace(data, cv_fold=None)
    params = {'n_estimators': 3, 'min_data_in_leaf': 1, 'num_threads': 1, 'verbosity': -1}
    encoded = ModelRecipe(LightGBMModel(), encoder=OneHotEncoder(), params=params).fit(data)
    members = [encoded]
    if method in ('blending', 'stacking'):
        members.append(ModelRecipe(LightGBMModel(), params=params).fit(data))
    options = {'blend_weights': [.3, .7]} if method == 'blending' else {}
    ensemble = EnsemblePipeline(members, method=method, n_members=2, **options).run()
    # Scoring data includes unseen row combinations, uses the same fitted encoders.
    held = data.with_features(data.features.with_columns(pl.col('unused') + .25))
    expected = ensemble.permutation_importance(held, n_repeats=2, show_progress_bar=False)
    ensemble.save(str(tmp_path))
    loaded = load_ensemble(tmp_path)
    assert all(member.raw_train_data is None for member in loaded.base_pipelines)
    actual = loaded.permutation_importance(held, n_repeats=2, show_progress_bar=False)
    assert_frame_equal(actual.repeat_importances, expected.repeat_importances)
