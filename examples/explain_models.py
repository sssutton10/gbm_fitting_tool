"""Run post-fit diagnostics on a synthetic portfolio without external files.

From the repository root:
    .venv/bin/python examples/explain_models.py
    .venv/bin/python examples/explain_models.py --shap --native

The default uses Random Forest and core dependencies. --shap requires the
explain extra; --native adds a LightGBM member with native categories.
See docs/explanations.md for settings, interpretation, and output schemas.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import polars as pl

from ins_gbm import (
    EnsemblePipeline, LightGBMModel, ModelData, ModelRecipe, OneHotEncoder,
    PCAReducer, PreprocessingStep, RandomForestModel, load_model,
)
from ins_gbm.data.model_data import slice_model_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/explanation_example"))
    parser.add_argument("--shap", action="store_true", help="Also calculate SHAP (requires explain extra)")
    parser.add_argument("--native", action="store_true", help="Use a native-categorical LightGBM ensemble member")
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    n = 240
    age, vehicle_age = rng.uniform(18, 80, n), rng.uniform(0, 20, n)
    territory = rng.choice(["north", "south", "west"], n)
    mean = np.exp(-2 + .012 * (age - 45) + .025 * vehicle_age + .2 * (territory == "north"))
    data = ModelData(
        features=pl.DataFrame({"driver_age": age, "vehicle_age": vehicle_age, "territory": territory}),
        target=pl.Series("claim_rate", rng.poisson(mean).astype(float)),
        feature_names=["driver_age", "vehicle_age", "territory"], objective="poisson",
    ).validate()
    training = slice_model_data(data, np.arange(180))
    population = slice_model_data(data, np.arange(180, n))
    params = {"n_estimators": 20, "min_samples_leaf": 4, "n_jobs": 1, "random_state": 42}
    recipe = ModelRecipe(model=RandomForestModel(), encoder=OneHotEncoder(), params=params)
    fitted = recipe.fit(training)
    if args.native:
        second = ModelRecipe(model=LightGBMModel(), params={
            "n_estimators": 20, "num_leaves": 8, "num_threads": 1, "verbosity": -1,
        }).fit(training)
    else:
        second = ModelRecipe(model=RandomForestModel(), encoder=OneHotEncoder(),
                             params={**params, "random_state": 7}).fit(training)
    ensemble = EnsemblePipeline([fitted, second], blend_weights=[.6, .4]).run()

    # Original ModelData goes in, even though the models use different encodings.
    for name, model in [("single", fitted), ("ensemble", ensemble)]:
        report = model.explain(population, background=training, prediction_type="response", seed=42)
        print(name, "diagnostic features:", report.feature_names)
        report.pdp("driver_age", grid_size=8).export(args.output / name / "age_pdp")
        report.pdp(["driver_age", "territory"], grid_size=8).export(args.output / name / "joint_pdp")
        report.ice("driver_age", grid_size=8, centered=True).export(args.output / name / "age_ice")
        report.ale("driver_age", bins=4).export(args.output / name / "age_ale")
        # Two bins per axis keep this small example's joint cells populated.
        report.ale(["driver_age", "vehicle_age"], bins=2).export(args.output / name / "interaction_ale")
        if args.shap:
            import matplotlib.pyplot as plt

            values = report.shap(max_samples=12, max_background=20, n_permutations=2)
            print(name, values.summary())
            values.export(args.output / name / "shap")
            plt.close(values.plot_waterfall(row=0, output_path=args.output / name / "waterfall.svg"))
            plt.close(values.plot_dependence("driver_age", color_by="vehicle_age",
                                            output_path=args.output / name / "dependence.svg"))

    pca_fit = ModelRecipe(
        model=RandomForestModel(), encoder=OneHotEncoder(), params=params,
        preprocessing=[PreprocessingStep("ages", PCAReducer(2), ["driver_age", "vehicle_age"])],
    ).fit(training)
    pca_report = pca_fit.explain(population, background=training)
    print("PCA diagnostic features:", pca_report.feature_names)
    pca_report.pdp("ages__pca_1", grid_size=8).export(args.output / "pca" / "component_pdp")
    if args.shap:
        pca_report.shap(max_samples=12, max_background=20, n_permutations=2).export(args.output / "pca" / "shap")

    # Compact saved pipelines need explicit explanation/reference data, not a refit.
    fitted.save(str(args.output / "saved_model"))
    restored = load_model(str(args.output / "saved_model"))
    restored.explain(population).pdp("territory").export(args.output / "restored" / "territory_pdp")
    print("Wrote tables, settings, and figures to", args.output.resolve())


if __name__ == "__main__":
    main()
