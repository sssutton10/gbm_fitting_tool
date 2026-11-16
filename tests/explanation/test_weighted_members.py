"""Contracts for weighted member raw-score diagnostics, distinct from final SHAP."""

from dataclasses import replace
import json

import numpy as np
import polars as pl
import pytest

from ins_gbm import (
    CatBoostModel, EnsemblePipeline, LightGBMModel, ModelData, ModelRecipe,
    OneHotEncoder, PCAReducer, PreprocessingStep, RandomForestModel, XGBoostModel,
    load_ensemble,
)

pytest.importorskip("shap")


def portfolio(objective="poisson"):
    rng = np.random.default_rng(14)
    x, z = rng.normal(size=(2, 80))
    frame = pl.DataFrame(dict(x=x, z=z, region=["A", "B", "C", "D"] * 20))
    return ModelData(frame, pl.Series(np.exp(.4 + .3 * x - .2 * z)),
                     frame.columns, objective=objective)


def native_blend(data):
    for backend in ["lightgbm", "xgboost", "catboost"]:
        pytest.importorskip(backend)
    members = [
        ModelRecipe(LightGBMModel(), params=dict(n_estimators=8, num_threads=1,
                                                 min_data_in_leaf=2, verbosity=-1)).fit(data),
        ModelRecipe(XGBoostModel(), encoder=OneHotEncoder(), params=dict(
            n_estimators=8, max_depth=2, nthread=1)).fit(data, feature_names=["x", "region"]),
        ModelRecipe(CatBoostModel(), params=dict(iterations=8, depth=2, thread_count=1),
                    preprocessing=[PreprocessingStep("pc", PCAReducer(1), ["x", "z"])]).fit(data),
    ]
    return EnsemblePipeline(members, blend_weights=[.5, .3, .2]).run()


@pytest.mark.parametrize("objective", ["poisson", "gamma"])
def test_native_weighted_mixed_pca_categories_and_subset_reconstruction(objective, tmp_path):
    pytest.importorskip("shap")
    data = portfolio(objective)
    ensemble = native_blend(data)
    report = ensemble.explain(data, batch_size=7)  # Background deliberately absent.
    grouped = report.weighted_member_shap(max_samples=9)
    indices = grouped.settings["row_indices"]
    selected = replace(data, features=data.features[indices], target=data.target[indices])
    effective = np.asarray(ensemble.ensemble.weights, dtype=np.float32).astype(float)
    actual = sum(w * p.predict(selected, "link").to_numpy().astype(float)
                 for w, p in zip(effective, ensemble.base_pipelines))
    np.testing.assert_allclose(grouped.predictions["prediction"], actual, atol=2e-6)
    np.testing.assert_allclose(grouped.predictions["residual"], 0, atol=2e-6)
    assert set(grouped.contributions["feature"]) == {"x", "z", "region", "member[2]::pc__pca_1"}
    assert grouped.settings["explanation_target"] == "weighted_member_raw"
    assert grouped.settings["perturbation"] == "tree_path_dependent"
    assert len(grouped.settings["limitations"]) >= 5
    # This target differs from log(final response blend).
    assert np.max(np.abs(actual - np.log(ensemble.predict(selected).to_numpy()))) > 1e-4
    # Recombining signed indicators preserves their member's total contribution.
    encoded = report.weighted_member_shap(max_samples=9, group_one_hot=False)
    indicator_names = [n for n in encoded.samples.columns if n.startswith("member[1]::region__")]
    assert len(indicator_names) == 4
    ungrouped_region = encoded.contributions.filter(pl.col("feature").is_in(indicator_names + ["region"]))
    summed = ungrouped_region.group_by("row").agg(pl.col("contribution").sum()).sort("row")
    np.testing.assert_allclose(summed["contribution"],
                               grouped.contributions.filter(pl.col("feature") == "region")["contribution"], atol=1e-7)
    # Scoring context and the parent scale/background cannot change the target.
    context = replace(data, exposure=pl.Series(np.linspace(.5, 2, 80)) if objective == "poisson" else None,
                      offset=pl.Series(np.linspace(-.5, .5, 80)))
    other = ensemble.explain(context, background=context, prediction_type="link", batch_size=7)
    np.testing.assert_allclose(other.weighted_member_shap(max_samples=9).contributions["contribution"],
                               grouped.contributions["contribution"], atol=1e-7)
    ensemble.save(str(tmp_path / "ensemble"))
    restored = load_ensemble(str(tmp_path / "ensemble"))
    loaded = restored.explain(data, batch_size=7).weighted_member_shap(max_samples=9)
    np.testing.assert_allclose(loaded.contributions["contribution"], grouped.contributions["contribution"], atol=1e-7)


def test_components_with_same_name_keep_member_bases_separate():
    pytest.importorskip("lightgbm")
    data = portfolio()
    recipe = ModelRecipe(LightGBMModel(), encoder=OneHotEncoder(), params=dict(
        n_estimators=4, num_threads=1, verbosity=-1, min_data_in_leaf=2),
        preprocessing=[PreprocessingStep("pc", PCAReducer(1), ["x", "z"])])
    first = recipe.fit(data)
    shifted = data.with_features(data.features.with_columns((pl.col("x") * 10).alias("x")))
    second = recipe.fit(shifted)
    ensemble = EnsemblePipeline([first, second], blend_weights=[.4, .6]).run()
    result = ensemble.explain(data).weighted_member_shap(max_samples=3)
    assert {n for n in result.samples.columns if "pc__pca_1" in n} == {
        "member[0]::pc__pca_1", "member[1]::pc__pca_1"}
    assert "x" not in result.samples.columns


def test_xgboost_training_margin_and_summary_weights():
    pytest.importorskip("xgboost")
    data = replace(portfolio(), exposure=pl.Series(np.linspace(.5, 2, 80)),
                   offset=pl.Series(np.linspace(-.2, .2, 80)))
    fitted = ModelRecipe(XGBoostModel(), encoder=OneHotEncoder(), params=dict(
        n_estimators=4, nthread=1)).fit(data)
    ensemble = EnsemblePipeline([fitted], blend_weights=[1.]).run()
    result = ensemble.explain(data, batch_size=5).weighted_member_shap(max_samples=8, weights=np.arange(1, 81))
    selected = result.settings["row_indices"]
    no_context = replace(data, exposure=None, offset=None)
    np.testing.assert_allclose(result.predictions["prediction"], fitted.predict(no_context, "link").to_numpy()[selected], atol=2e-6)
    np.testing.assert_allclose(result.weights, np.arange(1, 81)[selected])


def test_raw_diagnostic_exports_and_plot_labels(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = native_blend(portfolio()).explain(portfolio()).weighted_member_shap(max_samples=4)
    result.export(tmp_path)
    metadata = json.loads((tmp_path / "settings.json").read_text())
    assert metadata["explanation_target"] == "weighted_member_raw"
    assert "not the final" in metadata["limitations"][0]
    assert (tmp_path / "importance.svg").exists()
    for fig in [result.plot_importance(), result.plot_beeswarm(), result.plot_dependence("region"), result.plot_waterfall()]:
        assert "raw-score diagnostic" in fig.axes[0].get_title()
        plt.close(fig)


def test_invalid_ensemble_options_and_cancellation():
    from ins_gbm.progress import PipelineCancelled
    data = portfolio()
    ensemble = native_blend(data)
    report = ensemble.explain(data)
    for options in [dict(max_samples=0), dict(group_one_hot=1), dict(weights=[1.])]:
        with pytest.raises(ValueError):
            report.weighted_member_shap(**options)
    with pytest.raises(ValueError, match="fixed-weight"):
        ensemble.base_pipelines[0].explain(data).weighted_member_shap()
    forest = ModelRecipe(RandomForestModel(), encoder=OneHotEncoder(),
                         params=dict(n_estimators=2)).fit(data)
    mixed = EnsemblePipeline([forest, ensemble.base_pipelines[0]], blend_weights=[.5, .5]).run()
    with pytest.raises(ValueError, match="raw log scales"):
        mixed.explain(data).weighted_member_shap()
    with pytest.raises(PipelineCancelled):
        ensemble.explain(data, should_stop=lambda: True).weighted_member_shap()


def test_native_categories_use_fitted_maps_and_prepare_once(monkeypatch):
    data = portfolio()
    ensemble = native_blend(data)
    scoring = data.with_features(data.features.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(pl.lit("unseen"))
        .when(pl.int_range(pl.len()) == 1).then(None).otherwise(pl.col("region")).alias("region")))
    report = ensemble.explain(scoring)
    calls = []
    for i, pipeline in enumerate(ensemble.base_pipelines):
        prepare = pipeline._prepare_data

        def counted(data, original=prepare, index=i):
            calls.append(index)
            return original(data)

        monkeypatch.setattr(pipeline, "_prepare_data", counted)
    result = report.weighted_member_shap(max_samples=None)
    assert calls == [0, 1, 2]
    assert np.isfinite(result.predictions["prediction"].to_numpy()).all()
    fitted = ensemble.base_pipelines[0].fitted_model
    monkeypatch.setattr(fitted, "native_input_fn", None)
    with pytest.raises(ValueError, match="fitted native category input adapter"):
        report.weighted_member_shap(max_samples=2)
