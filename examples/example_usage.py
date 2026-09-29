"""Runnable end-to-end examples for the public ins_gbm API.

Run from the repository root after installing the LightGBM extra:
    python examples/example_usage.py --output output/example
Add --cross-validate to save CV reports and compare a new candidate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from ins_gbm import (
    CVConfig, LightGBMModel, ModelData, ModelRecipe,
    compare_cv_double_lift, compare_reports, load_cv_result, load_model,
)


def frequency_data(seed: int = 42) -> tuple[ModelData, ModelData]:
    rng = np.random.default_rng(seed)
    n = 300
    age = rng.normal(45, 12, n)
    exposure = rng.uniform(0.2, 1.0, n)
    offset = rng.normal(0, 0.08, n)
    rate = np.exp(-2.0 + 0.015 * (age - 45) + offset)
    target = rng.poisson(rate * exposure).astype(float)
    benchmark = np.maximum(target.mean() * exposure / exposure.mean(), 1e-6)
    data = ModelData(
        features=pl.DataFrame({"driver_age": age}),
        target=pl.Series("claim_count", target),
        exposure=pl.Series("exposure", exposure),
        offset=pl.Series("offset", offset),
        comparisons=pl.DataFrame({"portfolio_average": benchmark}),
        cv_fold=pl.Series("fold", np.arange(n) % 3),
        feature_names=["driver_age"],
        objective="poisson",
    ).validate()
    train_idx, holdout_idx = np.arange(240), np.arange(240, n)
    from ins_gbm.data.model_data import slice_model_data
    return slice_model_data(data, train_idx), slice_model_data(data, holdout_idx)


def severity_data(seed: int = 7) -> tuple[ModelData, ModelData]:
    rng = np.random.default_rng(seed)
    n = 240
    vehicle_age = rng.uniform(0, 15, n)
    severity = rng.gamma(3.0, np.exp(6.0 + 0.025 * vehicle_age) / 3.0)
    data = ModelData(
        features=pl.DataFrame({"vehicle_age": vehicle_age}),
        target=pl.Series("severity", severity),
        weight=pl.Series("weight", np.ones(n)),
        feature_names=["vehicle_age"],
        objective="gamma",
    ).validate()
    from ins_gbm.data.model_data import slice_model_data
    return slice_model_data(data, np.arange(190)), slice_model_data(data, np.arange(190, n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/example"))
    parser.add_argument("--cross-validate", action="store_true")
    args = parser.parse_args()

    frequency_train, frequency_holdout = frequency_data()
    frequency_recipe = ModelRecipe(
        model=LightGBMModel(),
        params={"n_estimators": 30, "num_leaves": 8, "verbose": -1},
    )
    if args.cross_validate:
        cv = frequency_recipe.cross_validate(
            frequency_train,
            cv=CVConfig(folds="auto"),
        )
        print("Frequency CV")
        print(cv.summary)

    frequency_fit = frequency_recipe.fit(frequency_train)
    print("Frequency holdout")
    print(frequency_fit.evaluate(frequency_holdout).metrics())
    rates = frequency_fit.predict(frequency_holdout, prediction_type="rate")
    counts = frequency_fit.predict(
        frequency_holdout.features,
        exposure=frequency_holdout.exposure,
        offset=frequency_holdout.offset,
    )
    np.testing.assert_allclose(
        counts.to_numpy(), rates.to_numpy() * frequency_holdout.exposure.to_numpy()
    )

    args.output.mkdir(parents=True, exist_ok=True)
    frequency_fit.save(str(args.output / "frequency"))
    restored = load_model(str(args.output / "frequency"))
    np.testing.assert_allclose(restored.predict(frequency_holdout), counts)

    if args.cross_validate:
        cv_path = args.output / "frequency" / "cv_report"
        cv.save(str(cv_path))
        saved_cv = load_cv_result(str(cv_path))
        candidate_recipe = ModelRecipe(
            model=LightGBMModel(),
            params={"n_estimators": 10, "num_leaves": 16, "verbose": -1},
        )
        candidate_cv = candidate_recipe.cross_validate(
            frequency_train, cv=CVConfig(folds="auto"),
        )
        print("Saved versus candidate CV metrics")
        print(compare_reports({"saved": saved_cv, "candidate": candidate_cv}))
        print("CV double lift by fold (positive favors candidate)")
        print(compare_cv_double_lift(saved_cv, candidate_cv))

    severity_train, severity_holdout = severity_data()
    severity_fit = ModelRecipe(
        model=LightGBMModel(),
        params={"n_estimators": 30, "num_leaves": 8, "verbose": -1},
    ).fit(severity_train)
    print("Severity holdout")
    print(severity_fit.evaluate(severity_holdout).metrics())


if __name__ == "__main__":
    main()
