"""Compare original/optimized permutation SHAP for a heterogeneous PCA blend.

Run: .venv/bin/python examples/benchmark_shap.py --features 12 --samples 10
Requires lightgbm, xgboost, catboost and explain extras. Each mode runs in a
fresh process so peak RSS measurements are independent (including imports/fit).
"""

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np
import polars as pl

from ins_gbm import (
    CatBoostModel, EnsemblePipeline, LightGBMModel, ModelData, ModelRecipe,
    OneHotEncoder, PCAReducer, PreprocessingStep, XGBoostModel,
)


def run(args):
    rng = np.random.default_rng(42)
    names = [f"x{i}" for i in range(args.features)]
    x = rng.normal(size=(args.rows, args.features))
    frame = pl.DataFrame(x, schema=names).with_columns(
        pl.Series("region", rng.choice(["A", "B", "C"], args.rows)))
    data = ModelData(frame, pl.Series(rng.poisson(np.exp(.4 + .3 * x[:, 0])).astype(float)),
                     frame.columns, objective="poisson")
    half = max(2, args.features // 2)
    pca = [PreprocessingStep("pc", PCAReducer(2), names[:half])]
    configurations = [
        (LightGBMModel(), names + ["region"], pca,
         dict(n_estimators=args.trees, num_threads=1, verbosity=-1, min_data_in_leaf=5)),
        (XGBoostModel(), names[::2] + ["region"], [],
         dict(n_estimators=args.trees, nthread=1, max_depth=4)),
        (CatBoostModel(), names[1::2] + ["region"], [],
         dict(iterations=args.trees, thread_count=1, depth=4, verbose=False, allow_writing_files=False)),
    ]
    members = [ModelRecipe(model=model, encoder=OneHotEncoder(), preprocessing=steps,
                           params=params).fit(data, feature_names=columns)
               for model, columns, steps, params in configurations]
    ensemble = EnsemblePipeline(members, blend_weights=[.4, .3, .3]).run()
    # Warm up SHAP/Numba in both subprocesses before measuring.
    ensemble.explain(data, background=data).shap(
        engine="permutation", optimize=args.mode == "optimized", max_samples=1,
        max_background=2, n_permutations=1)
    report = ensemble.explain(data, background=data)
    work = {"actual_member_rows": 0}
    for member in members:
        original_predict = member.fitted_model.predict

        def counted_predict(data, prediction_type="response", original=original_predict):
            work["actual_member_rows"] += data.n_rows
            return original(data, prediction_type)

        member.fitted_model.predict = counted_predict
    started = perf_counter()
    if args.mode == "weighted":
        result = report.weighted_member_shap(max_samples=args.samples, n_threads=args.native_threads)
        # Native model calls bypass the ordinary prediction-counter wrapper.
        work["actual_member_rows"] = None
    else:
        result = report.shap(engine="permutation", optimize=args.mode == "optimized",
                             max_samples=args.samples, max_background=args.background,
                             n_permutations=args.cycles)
    elapsed = perf_counter() - started
    stats = result.settings.get("cache_statistics", {})
    result.contributions.write_parquet(args.output / f"{args.mode}.parquet")
    return dict(mode=args.mode, seconds=elapsed,
                native_threads=result.settings.get("n_threads"),
                explanation_target=result.settings.get("explanation_target", "final_prediction"),
                peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                **work, **stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--background", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("output/shap_benchmark"))
    parser.add_argument("--include-weighted", action="store_true",
                        help="Also time the different weighted member raw-score target; no SHAP parity comparison")
    parser.add_argument("--native-threads", type=int, default=None,
                        help="Native threads for weighted mode only; None preserves backend defaults")
    parser.add_argument("--mode", choices=["original", "optimized", "weighted"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.features < 4 or min(args.rows, args.samples, args.background, args.cycles, args.trees) < 1:
        parser.error("features must be at least 4; all row counts and cycles must be positive")
    if args.native_threads is not None and args.native_threads < 1:
        parser.error("--native-threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode:
        print(json.dumps(run(args)))
        return
    results = []
    for mode in ["original", "optimized"] + (["weighted"] if args.include_weighted else []):
        process = subprocess.run([sys.executable, __file__, *sys.argv[1:], "--mode", mode],
                                 check=True, capture_output=True, text=True)
        result = json.loads(process.stdout.strip().splitlines()[-1])
        results.append(result)
        print(json.dumps(result), flush=True)
    original = pl.read_parquet(args.output / "original.parquet")["contribution"].to_numpy()
    optimized = pl.read_parquet(args.output / "optimized.parquet")["contribution"].to_numpy()
    np.testing.assert_allclose(original, optimized, atol=3e-6, rtol=2e-5)
    summary = dict(runs=results, speedup=results[0]["seconds"] / results[1]["seconds"],
                   max_contribution_difference=float(np.max(np.abs(original - optimized))))
    (args.output / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
