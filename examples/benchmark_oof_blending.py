"""Benchmark OOF blend CPU layouts and aggregate process-tree RSS on Linux.

Run: ./.venv/bin/python examples/benchmark_oof_blending.py --rows 100000 --trees 300
Each layout runs in a fresh process with an eight-core affinity mask. RSS sums
shared pages across processes, so the memory measurement is conservative.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def process_tree_rss(pid: int) -> int:
    """Sample Linux process RSS, including descendants, tolerating exits."""
    try:
        root = Path(f"/proc/{pid}")
        pages = int((root / "statm").read_text().split()[1])
        children = (root / "task" / str(pid) / "children").read_text().split()
        return pages * os.sysconf("SC_PAGE_SIZE") + sum(process_tree_rss(int(child)) for child in children)
    except (OSError, IndexError, ValueError):
        return 0


def run_case(args):
    # Apply limits before numerical-library imports, including the base fits.
    cpus = sorted(os.sched_getaffinity(0))[:args.cores]
    os.sched_setaffinity(0, cpus)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ[name] = str(args.cores)

    import numpy as np
    import polars as pl

    from ins_gbm.data.model_data import ModelData
    from ins_gbm.ensemble.pipeline import EnsemblePipeline
    from ins_gbm.models.lightgbm import LightGBMModel
    from ins_gbm.pipeline import ModelRecipe

    timings = {}

    class Timings(logging.Handler):
        def emit(self, record):
            if record.msg.startswith("OOF prediction generation completed"):
                timings["oof_seconds"] = record.args[0]
            elif record.msg.startswith("Blend weight optimization completed"):
                timings["solver_seconds"] = record.args[0]

    logger = logging.getLogger("ins_gbm.ensemble.blending")
    logger.setLevel(logging.INFO)
    logger.addHandler(Timings())
    rng = np.random.default_rng(42)
    values = rng.normal(size=(args.rows, 16)).astype(np.float32)
    names = [f"x{i}" for i in range(values.shape[1])]
    data = ModelData(
        features=pl.DataFrame(values, schema=names), feature_names=names,
        target=pl.Series(rng.poisson(np.exp(values[:, 0] * .4 + values[:, 1] * .2)).astype(float)),
        objective="poisson",
    )
    # Both serial and parallel use the full CPU budget overall.
    pipelines = [ModelRecipe(
        LightGBMModel(), params={
            "n_estimators": args.trees, "num_leaves": leaves, "seed": 42,
            "num_threads": len(cpus), "force_col_wise": True,
        },
    ).fit(data) for leaves in (15, 23, 31, 39)]
    started = time.perf_counter()
    result = EnsemblePipeline(
        pipelines, blend_mode="oof", cv_folds=4, metric="auto",
        n_jobs=args.case[0], threads_per_worker=args.case[1], show_progress_bar=False,
    ).run()
    print(json.dumps({
        "workers": args.case[0], "threads_per_worker": args.case[1],
        "total_seconds": time.perf_counter() - started, **timings,
        "effective_workers": result.configuration["effective_n_jobs"],
        "effective_threads": result.configuration["effective_threads_per_worker"],
        "weights": result.ensemble.weights,
    }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--case", type=int, nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.rows, args.trees, args.cores) < 1:
        parser.error("rows, trees, and cores must be positive")
    if not sys.platform.startswith("linux"):
        parser.error("the affinity and process-tree memory sampler require Linux")
    if args.case:
        run_case(args)
        return
    for workers, threads in ((1, 8), (4, 2), (8, 1)):
        with tempfile.TemporaryFile(mode="w+") as output:
            process = subprocess.Popen([
                sys.executable, str(Path(__file__).resolve()), "--rows", str(args.rows),
                "--trees", str(args.trees), "--cores", str(args.cores),
                "--case", str(workers), str(threads),
            ], stdout=output)
            peak = 0
            try:
                while process.poll() is None:
                    peak = max(peak, process_tree_rss(process.pid))
                    time.sleep(.05)
            except BaseException:
                process.terminate()
                process.wait()
                raise
            if process.returncode:
                raise RuntimeError(f"Benchmark case {workers} workers failed")
            output.seek(0)
            result = json.loads(output.read())
        result["peak_aggregate_rss_gib"] = peak / 1024 ** 3
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
