from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from ins_gbm import (
    CVConfig, CrossValidationReport, EnsemblePipeline, LightGBMModel, ModelData,
    ModelRecipe, OneHotEncoder, compare_cv_double_lift, compare_reports,
    cross_validate_candidates, load_cv_result,
)
from ins_gbm.data.folds import resolve_folds
from ins_gbm.data.model_data import slice_model_data
from ins_gbm.ensemble._utils import _refit_pipeline


@pytest.fixture
def portfolio():
    n = 48
    data = ModelData(
        features=pl.DataFrame({
            "row": np.arange(n), "x": np.sin(np.arange(n)), "cat": ["a", "b"] * 24,
        }),
        target=pl.Series([0., 1., 2., 4.] * 12),
        feature_names=["row", "x", "cat"],
        exposure=pl.Series(np.linspace(.5, 2, n)),
        weight=pl.Series(np.linspace(.5, 3, n)),
        offset=pl.Series(np.linspace(-.2, .2, n)),
        cv_fold=pl.Series(np.array([10, 30, 70] * 16)), objective="poisson",
    )
    recipe = ModelRecipe(
        LightGBMModel(), encoder=OneHotEncoder(),
        params={"n_estimators": 4, "min_data_in_leaf": 2, "num_threads": 1},
    )
    first = recipe.fit(data)
    second = replace(recipe, params={**recipe.params, "num_leaves": 3}).fit(data)
    return data, recipe, first, second


@pytest.mark.parametrize("method,mode", [
    ("blending", "fixed"), ("blending", "oof"), ("blending", "validation"),
    ("stacking", "fixed"), ("seed_average", "fixed"), ("bagging", "fixed"),
])
@pytest.mark.parametrize("folds", ["random", "predefined"])
def test_ensemble_cv_is_outer_heldout_and_returns_standard_results(portfolio, method, mode, folds):
    data, _, first, second = portfolio
    templates = [first] if method in {"seed_average", "bagging"} else [first, second]
    candidate = EnsemblePipeline(
        templates, method=method, blend_mode=mode, blend_weights=[.4, .6],
        metric="auto", n_members=2,
    )
    cv = CVConfig(n_splits=3, folds=folds)
    # Inner random folds also work with grouped outer CV.
    result = candidate.cross_validate(data, cv=cv, inner_cv=CVConfig(2, folds="random"), show_progress_bar=False)
    assert result.predictions.shape == (48, 1)
    assert result.predictions["gbm"].is_finite().all()
    assert result.predictions["gbm"].min() > 0
    assert result.fold_metrics["fold"].n_unique() == 3
    assert result.fold_params.keys() == set(result.row_folds.to_list())
    assert result.actual.equals(data.target)
    assert result.exposure.equals(data.exposure)
    assert result.weight.equals(data.weight)
    assert first.raw_train_data is data
    assert first.recipe.tuning is None
    assert result.cv_config["execution"]["pipeline_fits"] > 0
    if method in {"bagging", "seed_average"}:
        assert result.cv_config["execution"]["pipeline_fits"] == 6  # No unused template fits.


def test_shared_candidates_match_independent_and_reduce_actual_fits(portfolio, monkeypatch):
    data, recipe, first, second = portfolio
    candidates = {
        "blend": EnsemblePipeline([first, second], blend_mode="oof", metric="auto"),
        "stack": EnsemblePipeline([first, second], method="stacking"),
        "single": recipe,  # Intentionally listed last to exercise resolved-param sharing.
    }
    calls = []
    original_fit = LightGBMModel.fit

    def record(self, train, *args, **kwargs):
        calls.append(train.n_rows)
        return original_fit(self, train, *args, **kwargs)

    monkeypatch.setattr(LightGBMModel, "fit", record)
    shared = cross_validate_candidates(candidates, data, show_progress_bar=False)
    actual_shared = len(calls)
    assert actual_shared == 18  # 3 outer * 2 bases * (1 full + 2 inner).
    assert shared["stack"].cv_config["execution"]["pipeline_fits"] == actual_shared
    assert shared["stack"].cv_config["execution"]["fit_cache_hits"] > 0
    assert shared["stack"].cv_config["execution"]["prediction_cache_hits"] > 0
    calls.clear()
    independent = {}
    for name, candidate in candidates.items():
        independent[name] = cross_validate_candidates({name: candidate}, data, show_progress_bar=False)[name]
    assert actual_shared < len(calls)
    for name in candidates:
        np.testing.assert_allclose(shared[name].predictions, independent[name].predictions, rtol=1e-12)
        assert shared[name].fold_metrics.equals(independent[name].fold_metrics)
        assert shared[name].data_signature == independent[name].data_signature
    regular = recipe.cross_validate(data)
    np.testing.assert_allclose(shared["single"].predictions, regular.predictions)
    assert compare_reports(shared).width == 5
    assert compare_cv_double_lift(shared["single"], shared["stack"]).height > 0


@pytest.mark.parametrize("scale", ["rate", "count", "severity"])
def test_stacking_double_lift_matches_independent_bucket_calculation(portfolio, scale):
    data, recipe, _, _ = portfolio
    objective = "gamma" if scale == "severity" else "poisson"
    data = replace(
        data, objective=objective,
        exposure=data.exposure if scale == "count" else None,
        target=data.target + .5 if scale == "severity" else data.target,
    )
    recipe = replace(recipe, model=LightGBMModel(objective=objective))
    first = recipe.fit(data)
    second = replace(recipe, params={**recipe.params, "num_leaves": 3}).fit(data)
    candidate = EnsemblePipeline([first, second], method="stacking")
    cv = CVConfig(folds="predefined")
    single = CrossValidationReport(recipe, data, cv=cv, show_progress_bar=False).run()
    stack = candidate.cross_validate(data, cv=cv, show_progress_bar=False)

    # Independently rebuild each stack to check that the stored comparison
    # predictions are the final outer-held-out response, in original row order.
    _, splits = resolve_folds(data, cv)
    for train_rows, held_rows in splits:
        train = slice_model_data(data, train_rows)
        held = slice_model_data(data, held_rows)
        bases = [_refit_pipeline(p, train, p.fitted_model.params) for p in (first, second)]
        expected = replace(candidate, fitted_pipelines=bases, cv=cv).run().predict(held)
        np.testing.assert_allclose(stack.predictions["gbm"][held_rows], expected, rtol=1e-6)

    y = data.target.to_numpy().astype(float)
    a = single.predictions["gbm"].to_numpy().astype(float)
    b = stack.predictions["gbm"].to_numpy().astype(float)
    weights = data.weight.to_numpy().astype(float)
    if scale == "count":
        exposure = data.exposure.to_numpy()
        y, a, b = y / exposure, a / exposure, b / exposure
        weights = weights * exposure

    # Do not call the production double-lift helpers for the reference score.
    def manual_score(rows):
        order = rows[np.argsort((b / a)[rows], kind="stable")]
        w = weights[order]
        buckets = np.minimum(
            np.floor((np.cumsum(w) - w / 2) / w.sum() * 10).astype(int), 9,
        )
        score = 0.
        for bucket in np.unique(buckets):
            indices = order[buckets == bucket]
            actual, pred_a, pred_b = [
                np.average(values[indices], weights=weights[indices])
                for values in (y, a, b)
            ]
            score += abs(pred_a - actual) - abs(pred_b - actual)
        return score

    scores = compare_cv_double_lift(single, stack)
    expected = manual_score(np.arange(data.n_rows))
    assert scores["score"][0] == pytest.approx(expected, rel=1e-7, abs=1e-8)
    for row in scores.filter(pl.col("scope") == "fold").iter_rows(named=True):
        rows = np.flatnonzero(single.row_folds.to_numpy() == int(row["fold"]))
        assert row["score"] == pytest.approx(manual_score(rows), rel=1e-7, abs=1e-8)

    comparison = compare_reports({"single": single, "stack": stack})
    row = comparison.filter(pl.col("metric") == "double_lift_score").row(0, named=True)
    assert row["single"] is None
    assert row["stack"] == f"{expected:+.4f}"
    assert row["preferred"] == (
        "tie" if abs(expected) < 1e-6 else "stack" if expected > 0 else "single"
    )
    assert compare_cv_double_lift(stack, stack)["score"].to_list() == [0.] * 4


@pytest.mark.parametrize("mode", ["fixed", "oof", "validation"])
def test_against_manually_rebuilt_outer_ensemble(portfolio, mode):
    data, _, first, second = portfolio
    candidate = EnsemblePipeline([first, second], blend_mode=mode, blend_weights=[.3, .7], metric="auto")
    cv = CVConfig(folds="predefined")
    inner = CVConfig(2, seed=17, folds="random")
    result = candidate.cross_validate(data, cv=cv, inner_cv=inner, show_progress_bar=False)
    _, splits = resolve_folds(data, cv)
    from ins_gbm.evaluation.candidates import _validation_split
    for fold_number, (train_idx, held_idx) in enumerate(splits):
        train = slice_model_data(data, train_idx)
        held = slice_model_data(data, held_idx)
        validation = None
        if mode == "validation":
            fit_idx, val_idx = _validation_split(train, inner, .2, inner.seed + fold_number)
            validation = slice_model_data(train, val_idx)
            train = slice_model_data(train, fit_idx)
        bases = [_refit_pipeline(p, train, p.fitted_model.params) for p in (first, second)]
        expected = replace(candidate, fitted_pipelines=bases, cv=inner, validation_data=validation).run().predict(held)
        np.testing.assert_allclose(result.predictions["gbm"][held_idx], expected, rtol=1e-6)


def test_report_entrypoint_benchmarks_and_persistence(portfolio, tmp_path):
    data, _, first, second = portfolio
    data = replace(data, comparisons=pl.DataFrame({"benchmark": [1.] * data.n_rows}))
    candidate = EnsemblePipeline([first, second], method="stacking")
    result = CrossValidationReport(candidate, data, show_progress_bar=False).run()
    direct = candidate.cross_validate(data, show_progress_bar=False)
    assert result.predictions.equals(direct.predictions)
    assert result.fold_metrics["model"].unique().sort().to_list() == ["benchmark", "gbm"]
    result.save(tmp_path)
    loaded = load_cv_result(tmp_path)
    assert loaded.cv_config == result.cv_config
    assert loaded.fold_params == result.fold_params
    assert loaded.predictions.equals(result.predictions)
    assert compare_cv_double_lift(loaded, result, data=data).height > 0
    for kwargs in ({"feature_names": ["x"]}, {"feature_stage": "encoded"}):
        with pytest.raises(ValueError, match="feature definitions"):
            CrossValidationReport(candidate, data).run(**kwargs)


def test_report_legacy_fold_and_benchmark_columns_are_excluded(portfolio):
    data, _, first, _ = portfolio
    fields = data.features.with_columns(data.cv_fold.alias("fold"), pl.lit(1.).alias("benchmark"))
    data = replace(data, features=fields, feature_names=fields.columns, cv_fold=None)
    result = CrossValidationReport(
        EnsemblePipeline([first], blend_weights=[1.]), data,
        fold_col="fold", benchmark_col="benchmark", show_progress_bar=False,
    ).run()
    assert result.fold_col == "fold"
    assert "benchmark" in result.predictions.columns
    assert "fold" not in result.feature_names
    assert "benchmark" not in result.feature_names


def test_loaded_templates_do_not_use_original_models_or_transforms(portfolio, monkeypatch):
    data, _, first, second = portfolio

    def forbidden(*args, **kwargs):
        pytest.fail("reused original fitted model or transform")

    for p in (first, second):
        monkeypatch.setattr(p.fitted_model, "predict_fn", forbidden)
        monkeypatch.setattr(p.encoder, "transform", forbidden)
    templates = [replace(p, raw_train_data=None) for p in (first, second)]
    result = EnsemblePipeline(templates, method="stacking").cross_validate(data, show_progress_bar=False)
    assert result.predictions["gbm"].is_finite().all()


def test_insufficient_inner_groups_and_external_validation_fail_before_fit(portfolio, monkeypatch):
    data, _, first, _ = portfolio
    monkeypatch.setattr(LightGBMModel, "fit", lambda *a, **k: pytest.fail("must fail before fitting"))
    data = replace(data, cv_fold=pl.Series(np.arange(data.n_rows) % 2))
    with pytest.raises(ValueError, match="outer fold.*at least two fold IDs"):
        EnsemblePipeline([first], method="stacking").cross_validate(data, show_progress_bar=False)
    with pytest.raises(ValueError, match="external validation_data"):
        EnsemblePipeline([first], blend_mode="validation", validation_data=data).cross_validate(data)


@pytest.mark.parametrize("mode", ["oof", "validation"])
def test_outer_held_rows_never_enter_base_fit(portfolio, monkeypatch, mode):
    data, _, first, _ = portfolio
    calls = []
    original = LightGBMModel.fit

    def record(self, train, *args, **kwargs):
        calls.append(set(train.features["row"].to_list()))
        return original(self, train, *args, **kwargs)

    monkeypatch.setattr(LightGBMModel, "fit", record)
    EnsemblePipeline([first], blend_mode=mode, metric="auto").cross_validate(data, show_progress_bar=False)
    # Predefined outer groups and internal groups are disjoint on every fit.
    _, splits = resolve_folds(data, CVConfig())
    per_fold = 3 if mode == "oof" else 1
    assert len(calls) == len(splits) * per_fold
    for i, (outer_train, outer_held) in enumerate(splits):
        for training_rows in calls[i * per_fold:(i + 1) * per_fold]:
            assert training_rows <= set(outer_train)
            assert training_rows.isdisjoint(outer_held)
            if mode == "validation":
                # Validation split holds out a complete remaining fold group.
                assert len(training_rows) == 16


@pytest.mark.parametrize("outer_refit,inner_refit,expected_tunes", [
    ("fixed", "fixed", 0), ("retune", "fixed", 3),
    ("fixed", "retune", 6), ("retune", "retune", 9),
])
def test_outer_and_inner_tuning_policies_are_independent(portfolio, monkeypatch, outer_refit, inner_refit, expected_tunes):
    from ins_gbm import HyperparameterTuner
    data, _, first, _ = portfolio
    first = replace(first, recipe=replace(first.recipe, tuning=HyperparameterTuner(n_trials=1)))
    seen = []

    def tune(self, train, model, **kwargs):
        seen.append(set(train.features["row"].to_list()))
        return {**(kwargs["base_params"] or {}), "n_estimators": 2, "min_data_in_leaf": 2, "num_threads": 1}, None

    monkeypatch.setattr(HyperparameterTuner, "tune", tune)
    result = EnsemblePipeline([first], method="stacking", refit=inner_refit).cross_validate(
        data, base_refit=outer_refit, show_progress_bar=False,
    )
    assert len(seen) == expected_tunes
    assert result.cv_config["execution"]["tuned_pipeline_fits"] == expected_tunes
    if seen:
        _, folds = resolve_folds(data, CVConfig())
        per_fold = expected_tunes // 3
        for i, (train, held) in enumerate(folds):
            for tuned_rows in seen[i * per_fold:(i + 1) * per_fold]:
                assert tuned_rows <= set(train)
                assert tuned_rows.isdisjoint(held)
    assert first.recipe.tuning is not None
    assert first.fitted_model.params["n_estimators"] == 4


def test_actual_tuner_runs_only_inside_outer_training_rows(portfolio, monkeypatch):
    from ins_gbm import HyperparameterTuner
    data, _, first, _ = portfolio
    tuner = HyperparameterTuner(n_trials=1, cv=CVConfig(2, folds="random"), search_space={}, show_progress_bar=False)
    first = replace(first, recipe=replace(first.recipe, tuning=tuner))
    observed = []
    original = LightGBMModel.fit

    def record(self, train, *args, **kwargs):
        observed.append(set(train.features["row"].to_list()))
        return original(self, train, *args, **kwargs)

    monkeypatch.setattr(LightGBMModel, "fit", record)
    result = EnsemblePipeline([first], blend_weights=[1.]).cross_validate(
        data, base_refit="retune", show_progress_bar=False,
    )
    assert len(observed) == 9  # Two tuner folds + one final fit per outer fold.
    assert result.cv_config["execution"]["pipeline_fits"] == 3
    for i, (train, held) in enumerate(resolve_folds(data, CVConfig())[1]):
        for training in observed[i * 3:(i + 1) * 3]:
            assert training <= set(train)
            assert training.isdisjoint(held)


def test_fits_with_tuning_and_custom_configs_are_not_shared(portfolio, monkeypatch):
    from ins_gbm import HyperparameterTuner
    data, recipe, _, _ = portfolio
    recipe = replace(recipe, tuning=HyperparameterTuner(n_trials=1))
    tuned = []

    def tune(self, train, model, **kwargs):
        tuned.append(train.n_rows)
        return kwargs["base_params"], None

    monkeypatch.setattr(HyperparameterTuner, "tune", tune)
    result = cross_validate_candidates({"a": recipe, "b": recipe}, data, show_progress_bar=False)
    assert len(tuned) == 6
    assert result["a"].cv_config["execution"]["fit_cache_hits"] == 0

    class CustomLightGBM(LightGBMModel):
        pass

    custom = replace(recipe, model=CustomLightGBM(), tuning=None)
    result = cross_validate_candidates({"a": custom, "b": custom}, data, show_progress_bar=False)
    assert result["a"].cv_config["execution"]["pipeline_fits"] == 6
    assert result["a"].cv_config["execution"]["fit_cache_hits"] == 0


def test_cache_distinguishes_model_config_feature_stages_and_ordered_rows(portfolio, monkeypatch):
    from ins_gbm.evaluation.candidates import _FitContext, _FitSpec
    data, recipe, first, _ = portfolio
    counters = dict(pipeline_fits=0, tuned_pipeline_fits=0, fit_cache_hits=0, prediction_cache_hits=0)
    context = _FitContext(data, counters)
    rows = np.arange(24)
    spec = _FitSpec(recipe, data.feature_names, recipe.params)
    actual = []
    original = LightGBMModel.fit

    def record(self, train, *args, **kwargs):
        actual.append(train.features["row"].to_list())
        return original(self, train, *args, **kwargs)

    monkeypatch.setattr(LightGBMModel, "fit", record)
    entry = context.fit(spec, rows)
    assert context.fit(spec, rows) is entry
    context.fit(replace(spec, params={**spec.params, "seed": 123}), rows)
    context.fit(replace(spec, params={**spec.params, "n_estimators": 2}), rows)
    context.fit(replace(spec, input_names=["row", "x"]), rows)
    context.fit(spec, rows[::-1])
    context.fit(spec, np.array([0, 0, 1, 1, 2, 3, 4, 5]))
    template = replace(first, selected_features=["row", "x"])
    context.fit(_FitSpec(template.recipe, data.feature_names, template.fitted_model.params, template), rows)
    assert len(actual) == 7
    assert actual[-3] == rows[::-1].tolist()
    assert actual[-2][:4] == [0, 0, 1, 1]
    held = np.arange(24, 48)
    p = context.predict(entry, held)
    assert context.predict(entry, held) is p
    np.testing.assert_allclose(context.predict(entry, held[::-1]), p[::-1])
    assert counters["prediction_cache_hits"] == 1


@pytest.mark.parametrize("objective", ["poisson", "gamma"])
def test_nested_stacker_matches_reference_with_exposure_weights_and_offsets(portfolio, objective):
    data, recipe, _, _ = portfolio
    if objective == "gamma":
        data = replace(data, objective=objective, target=data.target + 1)
    recipe = replace(recipe, model=LightGBMModel(objective=objective))
    template = recipe.fit(data)
    candidate = EnsemblePipeline([template], method="stacking")
    result = candidate.cross_validate(data, show_progress_bar=False)
    for label, (train_rows, held_rows) in zip(*resolve_folds(data, CVConfig())):
        train = slice_model_data(data, train_rows)
        held = slice_model_data(data, held_rows)
        fitted = _refit_pipeline(template, train, template.fitted_model.params)
        reference = replace(candidate, fitted_pipelines=[fitted]).run()
        expected = reference.predict(held)
        np.testing.assert_allclose(result.predictions["gbm"][held_rows], expected, rtol=1e-12)
        params = result.fold_params[label]
        assert params["meta_learner"] == ("PoissonRegressor" if objective == "poisson" else "GammaRegressor")
        assert params["inner_cv"]["n_splits"] == 2
        changed = replace(held, exposure=held.exposure * 3, offset=held.offset + np.log(2))
        np.testing.assert_allclose(
            reference.predict(changed), expected.to_numpy() * (6 if objective == "poisson" else 2),
            rtol=2e-6,
        )


@pytest.mark.parametrize("refit", ["fixed", "retune"])
def test_feature_stages_and_transform_refits_stay_inside_outer_fold(portfolio, monkeypatch, refit):
    from ins_gbm import HyperparameterTuner, PCAReducer
    data, recipe, _, _ = portfolio
    recipe = replace(recipe, preprocessing=[PCAReducer(2)])
    template = recipe.fit(data, feature_names=["row", "x"], feature_stage="encoded")
    template = replace(template, model_selected_features=["pca_1"])
    if refit == "retune":
        template = replace(template, recipe=replace(template.recipe, tuning=HyperparameterTuner(
            n_trials=1, cv=CVConfig(2, folds="random"), search_space={}, show_progress_bar=False,
        )))
    seen = []
    fitted_names = []
    original_pca = PCAReducer.fit
    original_model = LightGBMModel.fit

    def fit_pca(self, features, *args, **kwargs):
        assert features.columns == ["row", "x"]
        seen.append(set(features["row"].to_list()))
        return original_pca(self, features, *args, **kwargs)

    def fit_model(self, train, *args, **kwargs):
        fitted_names.append(train.feature_names)
        return original_model(self, train, *args, **kwargs)

    monkeypatch.setattr(PCAReducer, "fit", fit_pca)
    monkeypatch.setattr(LightGBMModel, "fit", fit_model)
    result = EnsemblePipeline([template], method="stacking").cross_validate(
        data, base_refit=refit, show_progress_bar=False,
    )
    assert result.predictions["gbm"].is_finite().all()
    assert all(names == ["pca_1"] for names in fitted_names)
    per_fold = len(seen) // 3
    for i, (train, held) in enumerate(resolve_folds(data, CVConfig())[1]):
        for transformed_rows in seen[i * per_fold:(i + 1) * per_fold]:
            assert transformed_rows <= set(train)
            assert transformed_rows.isdisjoint(held)
    assert template.selected_features == ["row", "x"]
    assert template.preprocessors[0].input_names == ["row", "x"]


def test_candidate_alignment_and_config_validation(portfolio):
    data, recipe, first, _ = portfolio
    for candidates, kwargs, message in [
        ({}, {}, "nonempty"),
        ({"bad": object()}, {}, "configurations"),
        ({"bad": recipe}, {"base_refit": "invalid"}, "base_refit"),
        ({"val": EnsemblePipeline([first], blend_mode="validation")}, {"validation_fraction": 1.}, "validation_fraction"),
    ]:
        with pytest.raises((ValueError, TypeError), match=message):
            cross_validate_candidates(candidates, data, show_progress_bar=False, **kwargs)
    gamma_recipe = replace(recipe, model=LightGBMModel(objective="gamma"))
    with pytest.raises(ValueError, match="conflicts"):
        cross_validate_candidates({"p": recipe, "g": gamma_recipe}, data)
    candidate = EnsemblePipeline([first], blend_weights=[1.])
    a = candidate.cross_validate(data, cv=CVConfig(3, seed=10, folds="random"), show_progress_bar=False)
    b = candidate.cross_validate(data, cv=CVConfig(3, seed=11, folds="random"), show_progress_bar=False)
    with pytest.raises(ValueError, match="different evaluation rows"):
        compare_cv_double_lift(a, b)
