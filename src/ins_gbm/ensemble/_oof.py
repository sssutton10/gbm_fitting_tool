"""Independent OOF fold tasks and bounded process execution."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Iterator

import cloudpickle
import numpy as np
from loky import ProcessPoolExecutor, cpu_count

from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.ensemble._utils import _apply_pipeline_recipe_fold_transforms

if TYPE_CHECKING:
    from ins_gbm.pipeline import FittedPipeline, ModelRecipe


_THREAD_PARAMS = {
    "lightgbm": ("num_threads", "num_thread", "nthread", "nthreads", "n_jobs"),
    "xgboost": ("nthread", "n_jobs"),
    "catboost": ("thread_count",),
    "random_forest": ("n_jobs",),
}
_THREAD_ENV = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS",
)


def _worker_settings(n_jobs: int, threads_per_worker: int, tasks: int) -> dict:
    for name, value in (("n_jobs", n_jobs), ("threads_per_worker", threads_per_worker)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    available = cpu_count()
    workers = min(int(n_jobs), tasks, available)
    # Serial is the compatibility path: do not override model thread parameters.
    threads = (
        None if n_jobs == 1
        else min(int(threads_per_worker), max(1, available // workers))
    )
    return {
        "n_jobs": int(n_jobs), "threads_per_worker": int(threads_per_worker),
        "effective_n_jobs": workers, "effective_threads_per_worker": threads,
    }


def _limited_params(params: dict | None, framework: str, threads: int) -> dict:
    result = dict(params or {})
    aliases = _THREAD_PARAMS.get(framework, ())
    if aliases:
        for alias in aliases:
            result.pop(alias, None)
        result[aliases[0]] = threads
    return result


@dataclass
class _FoldRecipe:
    """The small subset of pipeline state needed to refit fold transforms."""

    recipe: ModelRecipe
    params: dict
    framework: str
    selected_features: list[str] | None
    model_selected_features: list[str] | None

    @classmethod
    def from_pipeline(cls, pipeline: FittedPipeline) -> _FoldRecipe:
        return cls(
            pipeline.recipe, pipeline.fitted_model.params,
            pipeline.fitted_model.framework, pipeline.selected_features,
            pipeline.model_selected_features,
        )


def _prepare_recipe(template: _FoldRecipe, threads: int | None) -> _FoldRecipe:
    template = deepcopy(template)
    if threads is None:
        return template
    recipe = template.recipe
    template.params = _limited_params(template.params, template.framework, threads)
    recipe.params = _limited_params(recipe.params, template.framework, threads)
    if recipe.tuning is not None:
        tuner = recipe.tuning
        tuner.n_jobs = 1
        tuner.backend = "thread"
        tuner.show_progress_bar = False
        space = (
            recipe.model.default_search_space()
            if tuner.search_space is None else tuner.search_space
        )
        aliases = _THREAD_PARAMS.get(template.framework, ())
        tuner.search_space = {
            name: dist for name, dist in space.items() if name not in aliases
        }
    # Importance selection stages have their own independent model parameters.
    from ins_gbm.models.catboost import CatBoostModel
    from ins_gbm.models.lightgbm import LightGBMModel
    from ins_gbm.models.random_forest import RandomForestModel
    from ins_gbm.models.xgboost import XGBoostModel

    frameworks = {
        LightGBMModel: "lightgbm", XGBoostModel: "xgboost",
        CatBoostModel: "catboost", RandomForestModel: "random_forest",
    }
    for stage in getattr(recipe.selection, "stages", ()):
        framework = next(
            (value for model, value in frameworks.items() if isinstance(stage.model, model)),
            "",
        )
        stage.params = _limited_params(stage.params, framework, threads)
    return template


def _fit_fold(
    template: _FoldRecipe,
    data: ModelData,
    model_idx: int,
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    refit: str,
    threads: int | None,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Fit and predict one fold, without retaining fitted estimators."""
    try:
        template = _prepare_recipe(template, threads)
        fold_train = slice_model_data(data, train_idx)
        fold_val = slice_model_data(data, val_idx)
        recipe = template.recipe
        if refit == "retune" and recipe.tuning is not None:
            from ins_gbm.pipeline import ModelPipeline

            model_selected = template.model_selected_features
            run_kwargs = (
                {"feature_names": model_selected, "feature_stage": "model"}
                if model_selected is not None
                else {
                    "feature_names": template.selected_features,
                    "feature_stage": "encoded",
                }
                if recipe.selection is None and template.selected_features is not None
                else {}
            )
            fitted = ModelPipeline(fold_train, recipe).run(**run_kwargs)
            predictions = fitted.predict(fold_val, "response")
        else:
            current_train, current_val = _apply_pipeline_recipe_fold_transforms(
                template, fold_train, fold_val
            )
            fitted = recipe.model.fit(current_train, params=template.params)
            predictions = fitted.predict(current_val, "response")
        return model_idx, fold_idx, val_idx, predictions.to_numpy()
    except Exception as exc:
        raise RuntimeError(
            f"OOF model {model_idx + 1}, fold {fold_idx + 1} failed: {exc}"
        ) from exc


def _run_pickled_fold(payload: bytes):
    return _fit_fold(*cloudpickle.loads(payload))


def _parallel_folds(tasks: Iterator[tuple], workers: int, threads: int):
    """Keep at most one task per worker in flight, including serialized data."""
    executor = ProcessPoolExecutor(
        max_workers=workers, env={name: str(threads) for name in _THREAD_ENV}
    )
    pending = {}

    def submit(task):
        try:
            payload = cloudpickle.dumps(task)
        except Exception as exc:
            raise TypeError(
                f"OOF model {task[2] + 1}, fold {task[3] + 1} inputs cannot be "
                "serialized. Use n_jobs=1 for unsupported custom components."
            ) from exc
        pending[executor.submit(_run_pickled_fold, payload)] = (task[2], task[3])

    try:
        for _ in range(workers):
            task = next(tasks, None)
            if task is not None:
                submit(task)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                model_idx, fold_idx = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"OOF model {model_idx + 1}, fold {fold_idx + 1} worker failed: {exc}"
                    ) from exc
                yield result
                task = next(tasks, None)
                if task is not None:
                    submit(task)
    except BaseException:
        executor.shutdown(wait=True, kill_workers=True)
        raise
    else:
        executor.shutdown(wait=True)
