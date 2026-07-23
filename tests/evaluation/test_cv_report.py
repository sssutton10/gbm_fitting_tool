import numpy as np
import polars as pl
import pytest

from ins_gbm.data.model_data import ModelData
from ins_gbm.data.schema import infer_schema
from ins_gbm.evaluation.cv_report import CrossValidationReport, CVResult
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import ModelRecipe


def _poisson_data(raw: pl.DataFrame) -> ModelData:
    schema = infer_schema(raw, ["x1", "x3"])
    return ModelData(
        features=raw.select(["x1", "x3"]),
        target=raw["claim_count"],
        exposure=raw["exposure"],
        weight=None,
        feature_names=["x1", "x3"],
        schema=schema,
        objective="poisson",
    ).validate()


def test_run_returns_cv_result(poisson_raw):
    data = _poisson_data(poisson_raw)
    report = CrossValidationReport(
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        data=data,
        n_folds=3,
        seed=0,
    )
    result = report.run()
    assert isinstance(result, CVResult)


def test_random_folds_fold_metrics_row_count(poisson_raw):
    data = _poisson_data(poisson_raw)
    result = CrossValidationReport(
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        data=data,
        n_folds=3,
        seed=0,
    ).run()
    # 3 folds × 4 metrics = 12 rows for "gbm"
    gbm_rows = result.fold_metrics.filter(pl.col("model") == "gbm")
    assert gbm_rows.height == 12


def test_summary_has_mean_and_std(poisson_raw):
    data = _poisson_data(poisson_raw)
    result = CrossValidationReport(
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        data=data,
        n_folds=3,
        seed=0,
    ).run()
    gbm_summary = result.summary.filter(pl.col("model") == "gbm")
    assert "mean" in gbm_summary.columns
    assert "std" in gbm_summary.columns
    assert gbm_summary["std"].null_count() == 0


def test_run_passes_recipe_params_to_each_fold(poisson_raw, monkeypatch):
    data = _poisson_data(poisson_raw)
    params = {"n_estimators": 7, "learning_rate": 0.05}
    received_params = []
    original_fit = LightGBMModel.fit

    def recording_fit(self, fold_data, params=None):
        received_params.append(params)
        return original_fit(self, fold_data, params=params)

    monkeypatch.setattr(LightGBMModel, "fit", recording_fit)

    CrossValidationReport(
        recipe=ModelRecipe(
            model=LightGBMModel(objective="poisson"), params=params
        ),
        data=data,
        n_folds=3,
        seed=0,
    ).run()

    assert received_params == [params, params, params]


def test_run_selects_features_without_losing_special_columns(
    poisson_raw,
    monkeypatch,
):
    bench = (poisson_raw["x1"].abs() + 0.1).alias("bench_pred")
    folds = pl.Series("fold_id", [i % 3 for i in range(poisson_raw.height)])
    raw = poisson_raw.with_columns(bench, folds)
    schema = infer_schema(raw, ["x1", "x3"])
    data = ModelData(
        features=raw.select(["x1", "x3", "bench_pred", "fold_id"]),
        target=raw["claim_count"],
        exposure=raw["exposure"],
        weight=None,
        feature_names=["x1", "x3", "bench_pred", "fold_id"],
        schema=schema,
        objective="poisson",
    ).validate()
    fitted_features = []
    original_fit = LightGBMModel.fit

    def recording_fit(self, fold_data, params=None):
        fitted_features.append(fold_data.feature_names)
        return original_fit(self, fold_data, params=params)

    monkeypatch.setattr(LightGBMModel, "fit", recording_fit)
    result = CrossValidationReport(
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        data=data,
        benchmark_col="bench_pred",
        fold_col="fold_id",
    ).run(feature_names=["x1"])

    assert fitted_features == [["x1"], ["x1"], ["x1"]]
    assert result.feature_names == ["x1"]
    assert result.predictions.columns == ["gbm", "benchmark"]


def test_predefined_fold_col_uses_exact_fold_ids(poisson_raw):
    fold_series = pl.Series("fold_id", [i % 3 for i in range(poisson_raw.height)])
    raw = poisson_raw.with_columns(fold_series)
    schema = infer_schema(raw, ["x1", "x3"])
    data = ModelData(
        features=raw.select(["x1", "x3", "fold_id"]),
        target=raw["claim_count"],
        exposure=raw["exposure"],
        weight=None,
        feature_names=["x1", "x3", "fold_id"],
        schema=schema,
        objective="poisson",
    ).validate()
    result = CrossValidationReport(
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        data=data,
        n_folds=99,        # ignored when fold_col is set
        fold_col="fold_id",
        seed=0,
    ).run()
    fold_ids_used = result.fold_metrics["fold"].unique().sort().to_list()
    assert fold_ids_used == [0, 1, 2]
    assert result.fold_col == "fold_id"


def test_fold_col_dropped_before_fitting(poisson_raw):
    """fold_id must not appear as a model feature."""
    fold_series = pl.Series("fold_id", [i % 3 for i in range(poisson_raw.height)])
    raw = poisson_raw.with_columns(fold_series)
    schema = infer_schema(raw, ["x1", "x3"])
    data = ModelData(
        features=raw.select(["x1", "x3", "fold_id"]),
        target=raw["claim_count"],
        exposure=raw["exposure"],
        weight=None,
        feature_names=["x1", "x3", "fold_id"],
        schema=schema,
        objective="poisson",
    ).validate()
    # Should complete without error — LightGBM would error on a non-numeric fold_id
    # if it weren't dropped first
    result = CrossValidationReport(
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        data=data,
        fold_col="fold_id",
        seed=0,
    ).run()
    assert isinstance(result, CVResult)


def test_benchmark_col_adds_benchmark_rows(poisson_raw):
    # Use x1 as a dummy benchmark prediction (positive values after clipping)
    bench = poisson_raw["x1"].abs() + 0.1
    raw = poisson_raw.with_columns(bench.alias("bench_pred"))
    schema = infer_schema(raw, ["x1", "x3"])
    data = ModelData(
        features=raw.select(["x1", "x3", "bench_pred"]),
        target=raw["claim_count"],
        exposure=raw["exposure"],
        weight=None,
        feature_names=["x1", "x3", "bench_pred"],
        schema=schema,
        objective="poisson",
    ).validate()
    result = CrossValidationReport(
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        data=data,
        n_folds=3,
        benchmark_col="bench_pred",
        seed=0,
    ).run()
    models_in_metrics = result.fold_metrics["model"].unique().sort().to_list()
    assert "benchmark" in models_in_metrics
    assert "gbm" in models_in_metrics
    assert "double_lift_score" in result.fold_metrics["metric"]
    assert np.isfinite(result.double_lift_score())
    assert result.plot_double_lift() is not None


def test_n_folds_less_than_2_raises(poisson_raw):
    data = _poisson_data(poisson_raw)
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        CrossValidationReport(
            recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
            data=data,
            n_folds=1,
        ).run()


def test_missing_fold_col_raises(poisson_raw):
    data = _poisson_data(poisson_raw)
    with pytest.raises(ValueError, match="fold_col"):
        CrossValidationReport(
            recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
            data=data,
            fold_col="nonexistent",
        ).run()


def test_fold_col_equals_benchmark_col_raises(poisson_raw):
    fold_series = pl.Series("fold_id", [i % 3 for i in range(poisson_raw.height)])
    raw = poisson_raw.with_columns(fold_series)
    schema = infer_schema(raw, ["x1", "x3"])
    data = ModelData(
        features=raw.select(["x1", "x3", "fold_id"]),
        target=raw["claim_count"],
        exposure=raw["exposure"],
        weight=None,
        feature_names=["x1", "x3", "fold_id"],
        schema=schema,
        objective="poisson",
    ).validate()
    with pytest.raises(ValueError, match="same column"):
        CrossValidationReport(
            recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
            data=data,
            fold_col="fold_id",
            benchmark_col="fold_id",
        ).run()
