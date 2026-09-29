"""Compare a tuned existing feature set with a staged-selection challenger.

Example (replace paths and RAW column names with your own):
    python examples/refit_vs_feature_selection.py \
        --train training.parquet --holdout holdout.parquet \
        --target claim_count --exposure exposure \
        --existing driver_age vehicle_age \
        --candidates driver_age vehicle_age territory annual_miles \
        --stage-sizes 6 3 --cross-validate

Uses the September 28, 2026 API. Both models are trained from scratch: the
baseline reuses the existing model's features, not its fitted trees. Holdout
rows must be separate from training, selection, and tuning. If rows are related
(e.g. repeated policies), supply appropriate training fold IDs with --fold-col.
For severity use --objective gamma and omit --exposure.
--cross-validate saves both nested CV reports and compares their standard
metrics and double lift; it adds one complete outer CV run per recipe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from ins_gbm import (
    CVConfig,
    HyperparameterTuner,
    ImportanceSelectionStage,
    LightGBMModel,
    ModelRecipe,
    OneHotEncoder,
    StagedImportanceSelector,
    compare_cv_double_lift,
    compare_reports,
    load_cv_result,
    load_model_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--objective", choices=["poisson", "gamma"], default="poisson")
    parser.add_argument("--exposure")
    parser.add_argument("--weight")
    parser.add_argument("--fold-col", help="Predefined folds in the training file only")
    parser.add_argument("--existing", nargs="+", required=True, help="Existing RAW features")
    parser.add_argument("--candidates", nargs="+", required=True, help="Full RAW candidate pool")
    parser.add_argument("--stage-sizes", nargs=2, type=int, default=[100, 30])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument(
        "--cross-validate", action="store_true",
        help="Save nested CV reports and compare the two sets of OOF predictions",
    )
    parser.add_argument("--output", type=Path, default=Path("output/refit_comparison"))
    args = parser.parse_args()
    if not set(args.existing).issubset(args.candidates):
        parser.error("--candidates must include all --existing features")
    if not 0 < args.stage_sizes[1] <= args.stage_sizes[0]:
        parser.error("--stage-sizes must be positive and non-increasing")
    if args.trials < 1:
        parser.error("--trials must be positive")

    # Explicit predictor lists exclude IDs, targets, and benchmark predictions.
    columns = dict(
        target=args.target, exposure=args.exposure, weight=args.weight,
        objective=args.objective, feature_cols=args.candidates,
    )
    training = load_model_data(args.train, cv_fold=args.fold_col, **columns)
    holdout = load_model_data(args.holdout, **columns)

    # Both searches use the same folds, seed, metric, search space and trial budget.
    # "auto" honors training.cv_fold, otherwise uses seeded shuffled K-folds.
    cv = CVConfig(n_splits=5, seed=42, folds="auto")
    base_params = {"seed": 42, "num_threads": 2, "verbose": -1}

    def make_recipe(selection=None):
        return ModelRecipe(
            model=LightGBMModel(),
            encoder=OneHotEncoder(),
            selection=selection,
            selection_scope="fold",  # Relearn selection within every tuning fold.
            params=dict(base_params),  # Trial suggestions override overlapping keys.
            tuning=HyperparameterTuner(
                n_trials=args.trials, cv=cv, seed=42, metric=None, n_jobs=1,
            ),  # metric=None infers Poisson/Gamma deviance; lower is better.
        )

    baseline_recipe = make_recipe()
    selector = StagedImportanceSelector(stages=[
        ImportanceSelectionStage(
            name="broad_screen", model=LightGBMModel(),
            max_features=args.stage_sizes[0], importance_type="gain",
            params={**base_params, "n_estimators": 100, "num_leaves": 16},
        ),
        ImportanceSelectionStage(
            name="refined_screen", model=LightGBMModel(),
            max_features=args.stage_sizes[1], importance_type="gain",
            params={**base_params, "n_estimators": 250, "num_leaves": 31},
        ),
    ])
    challenger_recipe = make_recipe(selector)

    # Stage settings are illustrative, not tuned by the final model's tuner.
    # Caps count ENCODED columns (including individual one-hot indicators).
    baseline = baseline_recipe.fit(training, feature_names=args.existing)
    challenger = challenger_recipe.fit(training, feature_names=args.candidates)

    reports = []
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cross_validate:
        # Outer folds refit the whole recipe, including inner tuning and selection.
        # Predefined outer folds need at least three IDs for inner CV.
        baseline_cv = baseline_recipe.cross_validate(
            training, cv=cv, feature_names=args.existing,
        )
        baseline_cv.save(str(args.output / "existing_features" / "cv_report"))
        saved_baseline_cv = load_cv_result(
            str(args.output / "existing_features" / "cv_report")
        )
        challenger_cv = challenger_recipe.cross_validate(
            training, cv=cv, feature_names=args.candidates,
        )
        challenger_cv.save(str(args.output / "staged_selection" / "cv_report"))
        comparison = compare_reports({
            "existing_features": saved_baseline_cv,
            "staged_selection": challenger_cv,
        })
        double_lift = compare_cv_double_lift(saved_baseline_cv, challenger_cv)
        print(comparison)  # Includes pooled double lift.
        print("CV double lift by fold (positive favors staged_selection)")
        print(double_lift)
        comparison.write_csv(args.output / "cv_metric_comparison.csv")
        double_lift.write_csv(args.output / "cv_double_lift.csv")

    # Do not compare best tuning-trial scores as unbiased test estimates.
    # fitted.retune(...) freezes selection; it does not revalidate discovery.
    for name, fitted in [("existing_features", baseline), ("staged_selection", challenger)]:
        reports.append(fitted.evaluate(holdout).metrics().with_columns(pl.lit(name).alias("model")))
        fitted.save(str(args.output / name))
        fitted.tuning_history.write_csv(args.output / f"{name}_tuning.csv")
    metrics = pl.concat(reports).select("model", "metric", "value")
    print(metrics)
    metrics.write_csv(args.output / "holdout_metrics.csv")

    # Audit exactly what survived each stage; these are post-encoding names.
    for stage in challenger.selection_results:
        print(f"{stage.name}: {stage.selected_feature_names}")
        stage.ranking.write_csv(args.output / f"{stage.name}_ranking.csv")
    print("Existing encoded columns removed:", sorted(
        set(baseline.fitted_model.feature_names) - set(challenger.selected_features)
    ))
    # Poisson predict(holdout) returns counts; prediction_type="rate" returns rates.
    # Use a validation set / nested CV for further experimentation, keeping a fresh
    # final test set if this holdout has already guided repeated model changes.


if __name__ == "__main__":
    main()
