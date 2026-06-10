"""
ins_gbm example usage
=====================

Demonstrates the full pipeline for a Poisson frequency model and a Gamma
severity model, including the expected missing-value convention.

Missing value convention
------------------------
Before calling any ins_gbm API, replace missing values in your data:

    numeric / ordinal columns  →  -999999999   (float or int)
    categorical columns        →  "-999999999" (string)

The library is designed to receive data in this state.  Internally:
  - OneHotEncoder treats "-999999999" as an explicit category level.
  - LightGBM and CatBoost convert -999999999.0 back to NaN before training
    so their native missing-value branch logic applies.
  - XGBoost declares missing=-999999999.0 on DMatrix for the same effect.
  - Random Forest receives the sentinel as a real number (sklearn limitation).
"""

import polars as pl

# ── Missing value sentinels ───────────────────────────────────────────────────
from ins_gbm.preprocessing.encoder import _NUMERIC_FILL, _MISSING_LEVEL

NUMERIC_MISSING = _NUMERIC_FILL   # -999999999.0
CAT_MISSING     = _MISSING_LEVEL  # "-999999999"


# ── 1. Load raw data and apply missing value convention ───────────────────────

def prepare_frequency_data(raw: pl.DataFrame) -> pl.DataFrame:
    """Replace nulls with the library's expected sentinels."""
    return raw.with_columns([
        pl.col("x1").fill_null(NUMERIC_MISSING),
        pl.col("x3").fill_null(NUMERIC_MISSING),
        pl.col("x2").cast(pl.Utf8).fill_null(CAT_MISSING),
    ])


# ── 2. Frequency (Poisson) pipeline ──────────────────────────────────────────

from ins_gbm.data.schema import FeatureSchema
from ins_gbm.data.model_data import ModelData
from ins_gbm.data.loader import ParquetLoader
from ins_gbm.data.splitter import TrainTestSplit
from ins_gbm.preprocessing.encoder import OneHotEncoder
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import ModelRecipe, ModelPipeline
from ins_gbm.persistence.io import save_pipeline, load_pipeline


def run_frequency_example(parquet_path: str, output_dir: str = "output/frequency"):
    # Load
    raw = ParquetLoader(path=parquet_path).load()
    raw = prepare_frequency_data(raw)

    schema = FeatureSchema(
        numeric=["x1", "x3"],
        categorical=["x2"],
    )

    data = ModelData(
        features=raw.select(["x1", "x2", "x3"]),
        target=raw["claim_count"],
        exposure=raw["exposure"],
        schema=schema,
        objective="poisson",
    )
    data.validate()

    recipe = ModelRecipe(
        model=LightGBMModel(objective="poisson"),
        encoder=OneHotEncoder(),
    )

    result = ModelPipeline(
        data=data,
        split=TrainTestSplit(test_size=0.3, seed=42),
        recipe=recipe,
    ).run()

    print(result.report.metrics())
    save_pipeline(result, output_dir)
    print(f"Pipeline saved to {output_dir!r}")
    return result


# ── 3. Severity (Gamma) pipeline ─────────────────────────────────────────────

from ins_gbm.models.xgboost import XGBoostModel


def run_severity_example(parquet_path: str, output_dir: str = "output/severity"):
    raw = ParquetLoader(path=parquet_path).load()
    raw = raw.with_columns([
        pl.col("x1").fill_null(NUMERIC_MISSING),
        pl.col("x2").cast(pl.Utf8).fill_null(CAT_MISSING),
    ])

    schema = FeatureSchema(numeric=["x1"], categorical=["x2"])

    data = ModelData(
        features=raw.select(["x1", "x2"]),
        target=raw["severity"],
        weight=raw["weight"],
        schema=schema,
        objective="gamma",
    )
    data.validate()

    recipe = ModelRecipe(
        model=XGBoostModel(objective="gamma"),
        encoder=OneHotEncoder(),
    )

    result = ModelPipeline(
        data=data,
        split=TrainTestSplit(test_size=0.3, seed=42),
        recipe=recipe,
    ).run()

    print(result.report.metrics())
    save_pipeline(result, output_dir)
    print(f"Pipeline saved to {output_dir!r}")
    return result


# ── 4. Load a saved pipeline and predict ─────────────────────────────────────

def predict_example(output_dir: str, new_data: ModelData) -> pl.Series:
    """Load a persisted pipeline and generate predictions on new data."""
    loaded = load_pipeline(output_dir)
    return loaded.fitted_model.predict(new_data, prediction_type="response")


# ── 5. Ensemble example ───────────────────────────────────────────────────────

from ins_gbm.ensemble.pipeline import EnsemblePipeline


def run_ensemble_example(fitted_pipeline_a, fitted_pipeline_b):
    """Blend two pre-fitted pipelines with OOF-optimised weights."""
    ensemble_result = EnsemblePipeline(
        fitted_pipelines=[fitted_pipeline_a, fitted_pipeline_b],
        method="blending",
        blend_mode="oof",
    ).run()

    print(ensemble_result.report.metrics())
    return ensemble_result


if __name__ == "__main__":
    # Replace these paths with real parquet files.
    run_frequency_example("data/frequency.parquet")
    run_severity_example("data/severity.parquet")
