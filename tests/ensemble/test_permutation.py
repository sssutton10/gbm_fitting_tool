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


@pytest.mark.parametrize('objective', ['poisson', 'gamma'])
def test_cv_fixed_blend_fold_isolation_and_shared_encoding(objective, monkeypatch):
    from ins_gbm import CVConfig, EnsemblePipeline, PermutationImportanceResult
    from ins_gbm.data.dtypes import FIT_DTYPE

    data, ensemble = fixture(objective, 'blending')
    data = replace(data, cv_fold=pl.Series([10] * 20 + [20] * 20))
    before = data.features.clone()
    before_predictions = ensemble.predict(data)
    fitted_inputs, prediction_inputs, encoded_inputs = [], [], []
    original_fit = CategoryModel.fit
    original_encode = OneHotEncoder.fit

    def record_fit(self, train, params=None):
        fitted_inputs.append((self.encoded, train, params))
        fitted = original_fit(self, train, params)
        original_predict = fitted.predict_fn

        def predict(held, prediction_type):
            prediction_inputs.append((self.encoded, held))
            return original_predict(held, prediction_type)

        fitted.predict_fn = predict
        return fitted

    def record_encode(self, features, schema):
        encoded_inputs.append(features.clone())
        return original_encode(self, features, schema)

    class MustNotRun:
        def __getattr__(self, name):
            raise AssertionError('Selection/tuning must not run')

    for member in ensemble.base_pipelines:
        member.recipe.selection = MustNotRun()
        member.recipe.tuning = MustNotRun()
        member.fitted_model.params = {'frozen': 17}
        member.recipe.params = {'frozen': 99}
        member.raw_train_data = None
    monkeypatch.setattr(CategoryModel, 'fit', record_fit)
    monkeypatch.setattr(OneHotEncoder, 'fit', record_encode)
    candidate = EnsemblePipeline(ensemble.base_pipelines, blend_weights=[.3, .7])
    result = candidate.permutation_importance_cv(
        data, cv=CVConfig(folds='predefined'), n_repeats=3, show_progress_bar=False,
    )
    assert isinstance(result, PermutationImportanceResult)
    assert len(fitted_inputs) == 4  # two members, two folds, independent of repeats
    assert len(encoded_inputs) == 2
    assert all(params == {'frozen': 17} for _, _, params in fitted_inputs)
    assert result.settings['cv'] == {'folds': 'predefined', 'n_splits': 2, 'seed': None}
    assert result.baseline_scores['fold'].to_list() == [10, 20]
    assert result.summary['feature'].to_list() == ['category', 'unused']
    assert result.summary['mean_importance'][0] > 0
    assert result.summary['mean_importance'][1] == 0
    for i in range(2):
        _, trained, _ = fitted_inputs[2 * i]
        held = prediction_inputs[14 * i][1]
        assert set(trained.cv_fold.to_list()).isdisjoint(held.cv_fold.to_list())
        assert_frame_equal(encoded_inputs[i], data.features[20:40] if i == 0 else data.features[:20])
        for j in range(7):
            native = prediction_inputs[14 * i + 2 * j][1]
            encoded = prediction_inputs[14 * i + 2 * j + 1][1]
            expected = (native.features['category'] == 'b').cast(pl.Float64)
            assert encoded.features['category__b'].equals(expected.rename('category__b'))
            for name in ['target', 'exposure', 'weight', 'offset', 'cv_fold']:
                a, b = getattr(native, name), getattr(held, name)
                assert (a is None and b is None) or a.equals(b)
            # Independently score the complete fixed-weight blend.
            predictions = np.stack([
                original_fit(CategoryModel(objective), trained).predict(native).to_numpy(),
                original_fit(CategoryModel(objective, True, 2), trained).predict(encoded).to_numpy(),
            ], axis=1).astype(FIT_DTYPE)
            prediction = pl.Series(predictions @ np.asarray([.3, .7], dtype=FIT_DTYPE))
            score = compute_metrics(objective=objective, actual=native.target,
                                    predicted=prediction, exposure=native.exposure,
                                    weight=native.weight).filter(pl.col('metric') == f'{objective}_deviance')['value'][0]
            if j == 0:
                assert result.baseline_scores['baseline_score'][i] == pytest.approx(score)
            else:
                row = result.repeat_importances.row(i * 6 + j - 1, named=True)
                assert row['permuted_score'] == pytest.approx(score)
                assert row['importance'] == pytest.approx(score - row['baseline_score'])
    assert_frame_equal(data.features, before)
    assert ensemble.predict(data).equals(before_predictions)


@pytest.mark.parametrize('method', ['seed_average', 'bagging'])
@pytest.mark.parametrize('objective', ['poisson', 'gamma'])
def test_cv_resampling_fits_only_outer_training_rows(method, objective, monkeypatch):
    from ins_gbm import EnsemblePipeline, LightGBMModel

    data, _ = fixture(objective, 'blending')
    data = replace(data, cv_fold=pl.Series([10] * 20 + [20] * 20))
    params = {'n_estimators': 3, 'min_data_in_leaf': 1, 'num_threads': 1, 'verbosity': -1}
    template = ModelRecipe(LightGBMModel(objective=objective), encoder=OneHotEncoder(), params=params).fit(data)
    original_params = dict(template.fitted_model.params)
    received = []
    original_fit = LightGBMModel.fit

    def capture(self, train, params=None):
        received.append((train, dict(params)))
        return original_fit(self, train, params=params)

    monkeypatch.setattr(LightGBMModel, 'fit', capture)
    template.raw_train_data = None  # explicit training data suffices for loaded templates
    candidate = EnsemblePipeline([template], method=method, n_members=3,
                                 seed=12, sample_fraction=.5)
    result = candidate.permutation_importance_cv(data, n_repeats=2, show_progress_bar=False)
    assert len(received) == 6
    for fold_index, fold in enumerate([10, 20]):
        members = received[fold_index * 3:fold_index * 3 + 3]
        seeds = result.settings['fold_ensembles'][fold_index]['member_seeds']
        assert len(set(seeds)) == 3
        for (train, effective), seed in zip(members, seeds):
            assert set(train.cv_fold.to_list()) == ({10, 20} - {fold})
            assert train.n_rows == (10 if method == 'bagging' else 20)
            assert effective['seed'] == seed
            assert effective['n_estimators'] == original_params['n_estimators']
            indices = train.features['unused'].to_numpy().astype(int)
            for name in ['target', 'exposure', 'weight', 'offset', 'cv_fold']:
                a, b = getattr(train, name), getattr(data, name)
                assert (a is None and b is None) or a.equals(b.gather(indices))
    assert template.fitted_model.params == original_params
    assert result.baseline_scores['n_train'].to_list() == [20, 20]
    assert result.settings['ensemble_seed'] == 12
    assert result.settings['fold_ensembles'][0]['weights'] == [1/3] * 3
    assert_frame_equal(result.repeat_importances, candidate.permutation_importance_cv(
        data, n_repeats=2, show_progress_bar=False,
    ).repeat_importances)


@pytest.mark.parametrize('options', [
    {'method': 'stacking'},
    {'method': 'blending', 'blend_mode': 'oof'},
    {'method': 'blending', 'blend_mode': 'validation'},
    {'method': 'unknown'},
])
def test_cv_rejects_expensive_methods_before_fitting(options, monkeypatch):
    from ins_gbm import EnsemblePipeline

    data, ensemble = fixture('poisson', 'blending')

    def no_fit(*args, **kwargs):
        raise AssertionError('Unsupported configuration must fail before fitting')

    monkeypatch.setattr(CategoryModel, 'fit', no_fit)
    candidate = EnsemblePipeline(ensemble.base_pipelines, **options)
    with pytest.raises(ValueError, match='supports only fixed-weight blending'):
        candidate.permutation_importance_cv(data)


def test_cv_sampling_summary_and_different_member_inputs():
    from ins_gbm import CVConfig, EnsemblePipeline

    data, ensemble = fixture('poisson', 'blending')
    data = replace(data, cv_fold=None)
    ensemble.base_pipelines[0].input_feature_names = ['category']
    candidate = EnsemblePipeline(ensemble.base_pipelines, blend_weights=[.3, .7])
    options = dict(cv=CVConfig(folds='random', n_splits=4, seed=13),
                   n_repeats=3, max_samples=6, seed=14, show_progress_bar=False)
    first = candidate.permutation_importance_cv(data, **options)
    second = candidate.permutation_importance_cv(data, **options)
    assert_frame_equal(first.repeat_importances, second.repeat_importances)
    assert first.settings['feature_names'] == ['category', 'unused']
    assert first.settings['cv'] == {'folds': 'random', 'n_splits': 4, 'seed': 13}
    assert first.baseline_scores['n_evaluated'].to_list() == [6] * 4
    assert first.baseline_scores['n_train'].to_list() == [30] * 4
    for row in first.summary.iter_rows(named=True):
        folds = first.fold_importances.filter(pl.col('feature') == row['feature'])
        means = folds['mean_importance'].to_numpy()
        assert row['mean_importance'] == pytest.approx(means.mean())
        assert row['std_importance'] == pytest.approx(means.std())
        assert row['mean_repeat_std'] == pytest.approx(folds['repeat_std'].mean())
    more_repeats = candidate.permutation_importance_cv(data, **{**options, 'n_repeats': 5})
    assert_frame_equal(first.baseline_scores, more_repeats.baseline_scores)


@pytest.mark.parametrize('options', [
    {'n_repeats': 0}, {'n_repeats': True}, {'max_samples': 0},
    {'max_samples': True}, {'seed': -1}, {'seed': True},
])
def test_cv_invalid_options(options):
    from ins_gbm import EnsemblePipeline

    data, ensemble = fixture('poisson', 'blending')
    candidate = EnsemblePipeline(ensemble.base_pipelines, blend_weights=[.3, .7])
    with pytest.raises(ValueError):
        candidate.permutation_importance_cv(data, **options)


def test_cv_frozen_encoded_selection_missing_in_fold(monkeypatch):
    from ins_gbm import EnsemblePipeline

    data, ensemble = fixture('poisson', 'blending')
    # Each training fold lacks one of the originally fitted categorical levels.
    data = replace(data, cv_fold=pl.Series(np.tile([10, 20], 20)))
    ensemble.base_pipelines[1].selected_features = ['category__a', 'category__b']
    candidate = EnsemblePipeline(ensemble.base_pipelines, blend_weights=[.3, .7])
    with pytest.raises(RuntimeError, match='fold 10, member 2.*Frozen selected features missing'):
        candidate.permutation_importance_cv(data, show_progress_bar=False)


def test_cv_preprocessing_refitted_with_frozen_feature_stages(monkeypatch):
    from ins_gbm import CVConfig, EnsemblePipeline, LightGBMModel, PCAReducer

    data, _ = fixture('poisson', 'blending')
    data = replace(data, cv_fold=pl.Series([10] * 20 + [20] * 20))
    recipe = ModelRecipe(
        LightGBMModel(), encoder=OneHotEncoder(), preprocessing=[PCAReducer(2)],
        params={'n_estimators': 3, 'min_data_in_leaf': 1, 'num_threads': 1, 'verbosity': -1},
    )
    template = recipe.fit(data, feature_names=['unused', 'category__b'], feature_stage='encoded')
    template = replace(template, model_selected_features=['pca_1'])
    original_fit = PCAReducer.fit
    seen = []

    def capture(self, features, target=None):
        seen.append(features.clone())
        return original_fit(self, features, target)

    monkeypatch.setattr(PCAReducer, 'fit', capture)
    candidate = EnsemblePipeline([template], blend_weights=[1.])
    result = candidate.permutation_importance_cv(
        data, cv=CVConfig(folds='predefined'), n_repeats=1, show_progress_bar=False,
    )
    assert len(seen) == 2
    assert seen[0]['unused'].to_list() == list(range(20, 40))
    assert seen[1]['unused'].to_list() == list(range(20))
    assert all(frame.columns == ['unused', 'category__b'] for frame in seen)
    assert result.settings['selected_features'] == [['unused', 'category__b']]
    assert result.settings['model_selected_features'] == [['pca_1']]
    assert template.selected_features == ['unused', 'category__b']
    assert template.model_selected_features == ['pca_1']
