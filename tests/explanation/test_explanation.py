"""Numerical contracts for explanations of individual and combined predictions."""

from dataclasses import replace
from itertools import combinations
import math
import sys

import numpy as np
import polars as pl
import pytest

from ins_gbm import (
    EnsembleResult, FittedPipeline, LightGBMModel, ModelData, ModelRecipe,
    OneHotEncoder, PCAReducer, PreprocessingStep, RandomForestModel,
)
from ins_gbm.ensemble.blending import FittedBlendingEnsemble
from ins_gbm.ensemble.stacking import FittedStackingEnsemble
from ins_gbm.models.base import FittedModel
from ins_gbm.progress import PipelineCancelled


def population():
    x, z = np.meshgrid(np.arange(4, dtype=float), np.arange(4, dtype=float))
    return ModelData(pl.DataFrame({"x": x.ravel(), "z": z.ravel()}),
                     pl.Series(np.ones(16)), ["x", "z"], objective="poisson")


def analytic(data, function, objective="poisson"):
    def predict(d, prediction_type):
        response = function(d.features)
        if objective == "poisson" and d.exposure is not None:
            response = response * d.exposure.to_numpy()
        if d.offset is not None:
            response = response * np.exp(d.offset.to_numpy())
        if prediction_type == "rate":
            return pl.Series(response / d.exposure.to_numpy() if d.exposure is not None else response)
        return pl.Series(np.log(response) if prediction_type == "link" else response)

    fitted = FittedModel(None, {}, "analytic", objective, data.feature_names,
                         predict, lambda *_: pl.DataFrame())
    return FittedPipeline(fitted, ModelRecipe(LightGBMModel()), data.feature_names,
                          None, None, None, None, None, [], None, input_schema=data.schema)


def fit(data, *, encoder=None, preprocessing=None, forest=False):
    data = replace(data, schema=None)
    model = RandomForestModel() if forest else LightGBMModel()
    params = {"n_estimators": 6, "n_jobs": 1, "min_samples_leaf": 1} if forest else {
        "n_estimators": 6, "min_data_in_leaf": 1, "num_threads": 1, "verbosity": -1}
    return ModelRecipe(model=model, encoder=encoder, preprocessing=preprocessing or [], params=params).fit(data)


def blend(pipelines):
    return EnsembleResult(FittedBlendingEnsemble([1 / len(pipelines)] * len(pipelines), pipelines), pipelines)


@pytest.mark.parametrize("scale", ["response", "rate", "link"])
def test_linear_pdp_ice_and_context(scale):
    data = population()
    data = replace(data, exposure=pl.Series(np.linspace(.4, 2, 16)),
                   offset=pl.Series(np.linspace(-.3, .4, 16)),
                   weight=pl.Series(np.arange(1, 17, dtype=float)))
    pipeline = analytic(data, lambda f: 2 + 3 * f["x"].to_numpy() + f["z"].to_numpy())
    report = pipeline.explain(data, prediction_type=scale)
    grid = [0., .5, 2.]
    pdp = report.pdp("x", grids={"x": grid})
    ice = report.ice("x", grid=grid)
    for value in grid:
        changed = data.with_features(data.features.with_columns(pl.lit(value).alias("x")))
        expected = pipeline.predict(changed, scale).to_numpy()
        selected = ice.table.filter(pl.col("value") == value)
        np.testing.assert_allclose(selected["prediction"], expected, rtol=1e-6)
        np.testing.assert_allclose(pdp.table.filter(pl.col("value") == value)["prediction"],
                                   np.average(expected, weights=data.weight), rtol=1e-6)
    centered = report.ice("x", grid=grid, centered=True, reference=.5)
    np.testing.assert_allclose(centered.table.filter(pl.col("value") == .5)["prediction"], 0)


def test_integer_numeric_interventions_are_not_truncated():
    data = population().with_features(population().features.cast(pl.Int32))
    pipeline = analytic(data, lambda f: 1 + f["x"].to_numpy())
    assert pipeline.explain(data).pdp("x", grids={"x": [.5]}).table["prediction"][0] == 1.5


def test_linear_ale_and_second_order_interaction():
    data = population()
    pipeline = analytic(data, lambda f: 2 + 3 * f["x"].to_numpy() + f["z"].to_numpy())
    result = pipeline.explain(data).ale("x", edges={"x": [0, 1, 2, 3]})
    expected = 3 * result.table["value"].to_numpy()
    expected -= np.average(expected, weights=result.support["weight"])
    np.testing.assert_allclose(result.table["effect"], expected)
    result = pipeline.explain(data).ale(["x", "z"], bins=2)
    np.testing.assert_allclose(result.table["effect"], 0, atol=1e-10)
    interaction = analytic(data, lambda f: 2 + f["x"].to_numpy() * f["z"].to_numpy())
    result = interaction.explain(data).ale(["x", "z"], bins=2)
    np.testing.assert_allclose(result.table["effect"], [.5625, -.5625, -.5625, .5625])
    assert abs(np.average(result.table["effect"], weights=result.support["weight"])) < 1e-10


def test_missing_constant_and_sparse_ale():
    data = population()
    pipeline = analytic(data, lambda f: np.ones(f.height))
    constant = data.with_features(data.features.with_columns(pl.lit(2.).alias("x")))
    np.testing.assert_allclose(pipeline.explain(constant).ale("x").table["effect"], 0)
    missing = data.with_features(data.features.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(-999999999.).otherwise(pl.col("x")).alias("x")))
    result = pipeline.explain(missing).ale("x", bins=2)
    assert result.settings["excluded_missing_rows"] == 1
    with pytest.raises(ValueError, match="Empty"):
        pipeline.explain(data).ale(["x", "z"], bins=10)
    with pytest.raises(ValueError, match="edges"):
        pipeline.explain(data).ale("x", edges={"x": [2, 1]})


def test_mixed_native_one_hot_and_selected_levels():
    data = population()
    data = data.with_features(data.features.with_columns(pl.Series("region", ["A", "B"] * 8)))
    data = replace(data, schema=None)
    native, encoded = fit(data), fit(data, encoder=OneHotEncoder())
    ensemble = blend([native, encoded])
    report = ensemble.explain(data)
    np.testing.assert_allclose(report.adapter.predict(report.frame, np.arange(16)), ensemble.predict(data), rtol=1e-6)
    result = report.pdp("region", grids={"region": ["A", "B", "unseen", None]})
    for row in result.table.iter_rows(named=True):
        changed = data.with_features(data.features.with_columns(pl.lit(row["value"], dtype=pl.String).alias("region")))
        np.testing.assert_allclose(row["prediction"], ensemble.predict(changed).mean(), rtol=1e-6)
    with pytest.raises(ValueError, match="numeric"):
        report.ale("region")
    selected = ModelRecipe(model=LightGBMModel(), encoder=OneHotEncoder(), params={
        "n_estimators": 3, "num_threads": 1, "min_data_in_leaf": 1}).fit(
            data, feature_names=["x", "region__A"], feature_stage="encoded")
    selected_report = blend([native, selected]).explain(data)
    np.testing.assert_allclose(selected_report.adapter.predict(selected_report.frame, np.arange(16)),
                               blend([native, selected]).predict(data), rtol=1e-6)


def test_pca_coordinates_compatibility_and_saved_models(tmp_path):
    data = population()
    preprocessing = [PreprocessingStep("ages", PCAReducer(2), ["x", "z"])]
    pipeline = fit(data, preprocessing=preprocessing)
    report = pipeline.explain(data)
    assert report.frame.columns == ["ages__pca_1", "ages__pca_2"]
    np.testing.assert_allclose(report.adapter.predict(report.frame, np.arange(16)), pipeline.predict(data), rtol=1e-6)
    report.pdp("ages__pca_1")
    shared = blend([pipeline, pipeline])
    shared.explain(data)
    selected = ModelRecipe(model=LightGBMModel(), preprocessing=preprocessing, params={
        "n_estimators": 3, "num_threads": 1, "min_data_in_leaf": 1}).fit(
            data, feature_names=["ages__pca_1"], feature_stage="model")
    assert blend([pipeline, selected]).explain(data).feature_names == ["ages__pca_1", "ages__pca_2"]
    shared.save(str(tmp_path / "ensemble"))
    from ins_gbm import load_ensemble
    restored = load_ensemble(str(tmp_path / "ensemble"))
    restored_report = restored.explain(data)
    np.testing.assert_allclose(restored_report.adapter.predict(restored_report.frame, np.arange(16)), shared.predict(data), rtol=1e-6)
    other = fit(data.with_features(data.features.with_columns((pl.col("x") * 2).alias("x"))), preprocessing=preprocessing)
    with pytest.raises(ValueError, match="Incompatible"):
        blend([pipeline, other]).explain(data)
    with pytest.raises(ValueError, match="raw/component"):
        blend([pipeline, fit(data)]).explain(data)


def exact_shap(report, row, background):
    names = report.frame.columns
    result = np.zeros(len(names))
    source = report.frame.row(row, named=True)
    n = len(names)
    for j, feature in enumerate(names):
        rest = [name for name in names if name != feature]
        for size in range(n):
            for subset in combinations(rest, size):
                left = background.with_columns([pl.lit(source[name]).alias(name) for name in subset])
                right = left.with_columns(pl.lit(source[feature]).alias(feature))
                context = np.full(background.height, row)
                difference = report.adapter.predict(right, context).mean() - report.adapter.predict(left, context).mean()
                result[j] += math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n) * difference
    return result


@pytest.mark.parametrize("scale", ["response", "rate", "link"])
def test_shap_exact_blend_and_context(scale):
    pytest.importorskip("shap")
    data = replace(population(), exposure=pl.Series(np.linspace(.4, 2, 16)),
                   offset=pl.Series(np.linspace(-.3, .4, 16)))
    first = analytic(data, lambda f: np.exp(.2 * f["x"].to_numpy() + .1 * f["z"].to_numpy()))
    second = analytic(data, lambda f: np.exp(-.4 * f["x"].to_numpy() + .3 * f["z"].to_numpy()))
    report = blend([first, second]).explain(data, background=data, prediction_type=scale)
    result = report.shap(engine="permutation", n_permutations=1)
    for row in [0, 7, 15]:
        values = result.contributions.filter(pl.col("row") == row)["contribution"]
        np.testing.assert_allclose(values, exact_shap(report, row, report.frame), atol=2e-7)
    np.testing.assert_allclose(result.predictions["residual"], 0, atol=2e-7)
    # Context-specific reference means differ when exposure/offset vary.
    if scale != "rate":
        assert result.predictions["base_value"].n_unique() > 1


def test_native_lightgbm_and_random_forest_shap():
    pytest.importorskip("shap")
    data = replace(population(), target=pl.Series(np.arange(16) + 1.))
    for forest, scale in [(False, "link"), (True, "response")]:
        pipeline = fit(data, forest=forest)
        report = pipeline.explain(data, background=data, prediction_type=scale)
        tree = report.shap(engine="tree", n_permutations=1)
        general = report.shap(engine="permutation", n_permutations=1)
        assert tree.settings["engine"] == "tree"
        np.testing.assert_allclose(tree.contributions["contribution"], general.contributions["contribution"], atol=2e-6)
    report = fit(data).explain(data, background=data)
    assert report.shap(n_permutations=1).settings["engine"] == "permutation"
    with pytest.raises(ValueError, match="ineligible"):
        report.shap(engine="tree")


def test_categorical_and_pca_shap():
    pytest.importorskip("shap")
    data = population().with_features(population().features.with_columns(pl.Series("region", ["A", "B"] * 8)))
    pipeline = fit(data, encoder=OneHotEncoder(), preprocessing=[PreprocessingStep("ages", PCAReducer(1), ["x", "z"])])
    report = pipeline.explain(data, background=data)
    result = report.shap(n_permutations=1)
    assert set(result.summary()["feature"]) == {"region", "ages__pca_1"}
    assert not any("region__" in name for name in result.summary()["feature"])
    assert result.settings["engine"] == "permutation"


def test_weighting_groups_repeats_segments_exports_and_plots(tmp_path):
    pytest.importorskip("shap")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = population()
    report = analytic(data, lambda f: 1 + f["x"].to_numpy() + f["z"].to_numpy()).explain(data, background=data)
    result = report.shap(engine="permutation", n_permutations=1, n_repeats=2)
    grouped = result.summary(groups={"all": ["x", "z"]})
    summed = result.contributions.group_by("repeat", "row").agg(pl.col("contribution").sum())
    np.testing.assert_allclose(grouped["mean_abs_contribution"][0], summed["contribution"].abs().mean())
    assert result.segment_summary(["a", "b"] * 8).height == 4
    result.export(tmp_path / "shap")
    result.plot_waterfall(output_path=tmp_path / "waterfall.svg")
    result.plot_dependence("x", color_by="z", output_path=tmp_path / "dependence.png")
    report.pdp(["x", "z"], grid_size=3).export(tmp_path / "pdp")
    report.ice("x", grid_size=3, centered=True).export(tmp_path / "ice")
    report.ale("x", bins=2).export(tmp_path / "ale")
    assert (tmp_path / "shap/settings.json").exists()
    assert (tmp_path / "pdp/support.csv").exists()
    plt.close("all")


def test_optional_dependency_validation_cancellation_and_sampling(monkeypatch):
    data = population()
    pipeline = analytic(data, lambda f: np.ones(f.height))
    report = pipeline.explain(data)
    with pytest.raises(ValueError, match="background"):
        report.shap()
    with pytest.raises(ValueError, match="weights"):
        report.pdp("x", weights=[-1] * 16)
    with pytest.raises(ValueError, match="Unknown"):
        report.pdp("missing")
    with pytest.raises(ValueError, match="positive integer"):
        report.pdp("x", max_samples=0)
    with pytest.raises(PipelineCancelled):
        pipeline.explain(data, should_stop=lambda: True).pdp("x")
    assert report.pdp("x", max_samples=5).settings["row_indices"] == report.pdp("x", max_samples=5).settings["row_indices"]
    monkeypatch.setitem(sys.modules, "shap", None)
    with pytest.raises(ImportError, match="explain"):
        pipeline.explain(data, background=data).shap()
    assert report.pdp("x").table.height > 0


@pytest.mark.parametrize("distribution_aware", [True, False])
@pytest.mark.parametrize("scale", ["response", "rate", "link"])
def test_stacking_diagnostics_follow_actual_meta_prediction(distribution_aware, scale):
    from sklearn.linear_model import PoissonRegressor, Ridge

    data = replace(population(), exposure=pl.Series(np.linspace(.4, 2, 16)),
                   offset=pl.Series(np.linspace(-.2, .2, 16)))
    pipelines = [analytic(data, lambda f: 1 + f["x"].to_numpy()),
                 analytic(data, lambda f: 2 + f["z"].to_numpy())]
    base = np.column_stack([p.predict(data) for p in pipelines])
    x = np.log(base / (data.exposure.to_numpy() * np.exp(data.offset.to_numpy()))[:, None]) if distribution_aware else base
    meta = (PoissonRegressor(alpha=.1) if distribution_aware else Ridge()).fit(x, np.arange(16) + 1.)
    fitted = FittedStackingEnsemble(meta, pipelines, None, distribution_aware)
    ensemble = EnsembleResult(fitted, pipelines)
    report = ensemble.explain(data, prediction_type=scale)
    for value in [.5, 2.5]:
        changed = data.with_features(data.features.with_columns(pl.lit(value).alias("x")))
        expected = ensemble.predict(changed).to_numpy()
        expected = np.log(expected) if scale == "link" else expected / data.exposure.to_numpy() if scale == "rate" else expected
        actual = report.pdp("x", grids={"x": [value]}).table["prediction"][0]
        np.testing.assert_allclose(actual, expected.mean(), rtol=2e-6)


@pytest.mark.parametrize("backend", ["lightgbm", "xgboost", "catboost", "random_forest"])
@pytest.mark.parametrize("objective", ["poisson", "gamma"])
def test_backend_prediction_parity_and_shap(backend, objective):
    pytest.importorskip("shap")
    if backend != "random_forest":
        pytest.importorskip(backend)
    from ins_gbm import CatBoostModel, XGBoostModel

    models = {"lightgbm": LightGBMModel, "xgboost": XGBoostModel,
              "catboost": CatBoostModel, "random_forest": RandomForestModel}
    params = {
        "lightgbm": {"n_estimators": 3, "min_data_in_leaf": 1, "num_threads": 1, "verbosity": -1},
        "xgboost": {"n_estimators": 3, "max_depth": 2, "nthread": 1},
        "catboost": {"iterations": 3, "depth": 2, "thread_count": 1, "verbose": False},
        "random_forest": {"n_estimators": 3, "max_depth": 2, "n_jobs": 1},
    }[backend]
    data = replace(population(), target=pl.Series(np.arange(16) + 1.), objective=objective,
                   exposure=pl.Series(np.linspace(.4, 2, 16)),
                   offset=None if backend == "random_forest" else pl.Series(np.linspace(-.2, .2, 16)))
    pipeline = ModelRecipe(models[backend](objective=objective), params=params).fit(data)
    for scale in (["response", "rate", "link"] if objective == "poisson" else ["response", "link"]):
        report = pipeline.explain(data, background=data, prediction_type=scale, batch_size=5)
        np.testing.assert_allclose(report.adapter.predict(report.frame, np.arange(16)), pipeline.predict(data, scale), rtol=1e-6)
        result = report.shap(max_samples=2, n_permutations=1)
        np.testing.assert_allclose(result.predictions["residual"], 0, atol=2e-5)
    if objective == "gamma":
        with pytest.raises(ValueError, match="invalid"):
            pipeline.explain(data, prediction_type="rate")


@pytest.mark.parametrize("method", ["blending", "seed_average", "bagging", "stacking"])
def test_real_ensemble_methods_parity_shap_and_saved_artifacts(method, tmp_path):
    pytest.importorskip("shap")
    from ins_gbm import EnsemblePipeline, load_ensemble

    data = replace(population(), target=pl.Series(np.arange(16) + 1.))
    pipeline = fit(data)
    kwargs = {"n_members": 2} if method in {"bagging", "seed_average"} else {}
    if method == "blending":
        kwargs["blend_weights"] = [1.]
    ensemble = EnsemblePipeline([pipeline], method=method, **kwargs).run()
    report = ensemble.explain(data, background=data, prediction_type="link")
    np.testing.assert_allclose(report.adapter.predict(report.frame, np.arange(16)), np.log(ensemble.predict(data)), rtol=1e-6)
    result = report.shap(max_samples=2, n_permutations=1)
    np.testing.assert_allclose(result.predictions["residual"], 0, atol=2e-6)
    ensemble.save(str(tmp_path / method))
    loaded = load_ensemble(str(tmp_path / method))
    loaded_report = loaded.explain(data, background=data, prediction_type="link")
    np.testing.assert_allclose(loaded_report.pdp("x").table["prediction"], report.pdp("x").table["prediction"])


def test_native_tree_context_corrections():
    pytest.importorskip("shap")
    data = replace(population(), target=pl.Series(np.arange(16) + 1.),
                   exposure=pl.Series(np.linspace(.4, 2, 16)),
                   offset=pl.Series(np.linspace(-.2, .2, 16)))
    pipeline = fit(data)
    report = pipeline.explain(data, background=data, prediction_type="link")
    native = report.shap(engine="tree", max_samples=3, n_permutations=1)
    general = report.shap(engine="permutation", max_samples=3, n_permutations=1)
    np.testing.assert_allclose(native.contributions["contribution"], general.contributions["contribution"], atol=2e-6)
    np.testing.assert_allclose(native.predictions["base_value"], general.predictions["base_value"], atol=2e-6)


def test_categorical_plot_and_missing_masking(tmp_path):
    pytest.importorskip("shap")
    import matplotlib.pyplot as plt

    data = replace(population().with_features(population().features.with_columns(
        pl.Series("region", ["A", "B", None, "unseen"] * 4))), schema=None)
    pipeline = fit(data, encoder=OneHotEncoder())
    report = pipeline.explain(data, background=data)
    result = report.shap(n_permutations=1)
    result.plot_dependence("region", color_by="x", output_path=tmp_path / "category.png")
    report.pdp("region").export(tmp_path / "category-pdp")
    report.ice("region", centered=True).export(tmp_path / "category-ice")
    plt.close("all")


def test_zero_weight_segments_are_explicitly_undefined():
    pytest.importorskip("shap")
    data = replace(population(), weight=pl.Series([0.] * 8 + [1.] * 8))
    report = analytic(data, lambda f: 1 + f["x"].to_numpy()).explain(data, background=data)
    result = report.shap(engine="permutation", n_permutations=1)
    segments = result.segment_summary(["zero"] * 8 + ["positive"] * 8)
    assert segments.filter(pl.col("segment") == "zero")["mean_abs_contribution"].null_count() == 2
    assert report.pdp("x", weights="uniform").settings["averaging_weights"] == [1.] * 16


@pytest.mark.parametrize("scale", ["response", "rate", "link"])
def test_clipped_custom_stacker_shap_matches_coalitions(scale):
    pytest.importorskip("shap")

    class Difference:
        def predict(self, x):
            return x[:, 0] - 2 * x[:, 1]

    data = replace(population(), exposure=pl.Series(np.linspace(.4, 2, 16)))
    pipelines = [analytic(data, lambda f: 1 + 3 * f["x"].to_numpy()),
                 analytic(data, lambda f: 1 + f["z"].to_numpy())]
    ensemble = EnsembleResult(FittedStackingEnsemble(Difference(), pipelines, None), pipelines)
    report = ensemble.explain(data, background=data, prediction_type=scale)
    result = report.shap(n_permutations=1)
    for row in [0, 7, 15]:
        np.testing.assert_allclose(result.contributions.filter(pl.col("row") == row)["contribution"],
                                   exact_shap(report, row, report.frame), atol=1e-6)


def test_progress_is_reported_and_unknown_grid_keys_fail():
    data = population()
    events = []
    report = analytic(data, lambda f: 1 + f["x"].to_numpy()).explain(data, progress_callback=events.append)
    report.pdp("x", grid_size=2)
    assert events and all(event.stage == "explain" for event in events)
    with pytest.raises(ValueError, match="Grid keys"):
        report.pdp("x", grids={"typo": [1, 2]})
    with pytest.raises(ValueError, match="Edge keys"):
        report.ale("x", edges={"typo": [1, 2]})
