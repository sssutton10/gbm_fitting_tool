from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Literal, Optional

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.data.folds import CVConfig
from ins_gbm.data.schema import FeatureSchema
from ins_gbm.evaluation.metrics import (
    _poisson_rate_metric_inputs,
    poisson_deviance,
    gamma_deviance,
    rmse,
    mae,
)
from ins_gbm.progress import ProgressCallback, ProgressEvent, PipelineCancelled
from ins_gbm.preprocessing.chain import fit_transform_chain


_METRIC_FN = {
    "poisson_deviance": poisson_deviance,
    "gamma_deviance": gamma_deviance,
    "rmse": rmse,
    "mae": mae,
}


def _create_journal_storage(file_path: str) -> Any:
    """Create process-safe JournalStorage with Windows-compatible locking."""
    from optuna.storages import JournalStorage
    from optuna.storages.journal import (
        JournalFileBackend,
        JournalFileOpenLock,
    )

    lock = (
        JournalFileOpenLock(file_path)
        if platform.system() == "Windows"
        else None
    )
    return JournalStorage(
        JournalFileBackend(file_path=file_path, lock_obj=lock)
    )


@dataclass
class _ObjectiveConfig:
    """Serializable inputs shared by local and subprocess trial objectives."""

    tuning_data: ModelData
    model: Any
    encoder: Optional[Any]
    selector: Optional[Any]
    preprocessing_chain: list[Any]
    encoder_schema: Optional[Any]
    model_selected_features: Optional[list[str]]
    fold_splits: list[tuple[np.ndarray, np.ndarray]]
    search_space: dict[str, Any]
    metric: str
    base_params: dict[str, Any]
    cancellation_path: Optional[str] = None


def _select_schema(
    schema: Optional[FeatureSchema],
    feature_names: Optional[list[str]],
) -> Optional[FeatureSchema]:
    """Restrict an explicit encoder schema to a runtime feature subset."""
    if schema is None or feature_names is None:
        return schema
    selected = set(feature_names)
    return FeatureSchema(
        numeric=[name for name in schema.numeric if name in selected],
        categorical=[name for name in schema.categorical if name in selected],
        ordinal=[name for name in schema.ordinal if name in selected],
        passthrough=[name for name in schema.passthrough if name in selected],
    )


def _suggest_from_distribution(trial: Any, name: str, dist: Any) -> Any:
    import optuna
    if isinstance(dist, optuna.distributions.IntDistribution):
        return trial.suggest_int(name, dist.low, dist.high, log=dist.log)
    elif isinstance(dist, optuna.distributions.FloatDistribution):
        return trial.suggest_float(name, dist.low, dist.high, log=dist.log)
    elif isinstance(dist, optuna.distributions.CategoricalDistribution):
        return trial.suggest_categorical(name, dist.choices)
    else:
        raise ValueError(f"Unsupported distribution type: {type(dist)}")


def _evaluate_trial(
    trial: Any,
    config: _ObjectiveConfig,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> float:
    """Evaluate one Optuna trial from serializable configuration."""
    import optuna

    params = dict(config.base_params)
    params.update({
        name: _suggest_from_distribution(trial, name, dist)
        for name, dist in config.search_space.items()
    })
    metric_fn = _METRIC_FN[config.metric]

    fold_scores: list[float] = []
    for fold_idx, (train_idx, val_idx) in enumerate(config.fold_splits):
        cancelled = (
            stop_requested is not None and stop_requested()
        ) or (
            config.cancellation_path is not None
            and os.path.exists(config.cancellation_path)
        )
        if cancelled:
            raise PipelineCancelled("cancelled during CV fold")

        train_data = slice_model_data(config.tuning_data, train_idx)
        val_data = slice_model_data(config.tuning_data, val_idx)

        transform_result = fit_transform_chain(
            train_data,
            encoder=config.encoder,
            selector=config.selector,
            preprocessing=config.preprocessing_chain,
            schema=config.encoder_schema,
            model_selected_features=config.model_selected_features,
        )
        train_data = transform_result.data
        val_data = transform_result.chain.transform(val_data)

        fitted_model = config.model.fit(train_data, params=params)
        preds = fitted_model.predict(val_data, prediction_type="response")

        metric_actual = val_data.target
        metric_predicted = preds
        if (
            val_data.objective == "poisson"
            and config.metric == "poisson_deviance"
        ):
            metric_actual, metric_predicted, weights = (
                _poisson_rate_metric_inputs(
                    val_data.target,
                    preds,
                    val_data.exposure,
                    val_data.weight,
                )
            )
        else:
            # Response predictions are expected counts for frequency models,
            # so exposure is not also a count-error weight.
            weights = val_data.weight
        score = metric_fn(
            metric_actual,
            metric_predicted,
            weights=weights,
        )
        fold_scores.append(score)

        trial.report(float(np.mean(fold_scores)), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


def _study_history(study: Any) -> pl.DataFrame:
    rows = []
    for trial in study.trials:
        if trial.value is not None:
            row: dict = {"trial": trial.number, "value": trial.value}
            row.update(trial.params)
            rows.append(row)

    if rows:
        return pl.DataFrame(rows)
    return pl.DataFrame({
        "trial": pl.Series([], dtype=pl.Int64),
        "value": pl.Series([], dtype=pl.Float64),
    })


@dataclass
class HyperparameterTuner:
    """Optuna-based hyperparameter tuner with CV fold evaluation.

    For each trial, encoder/selector/preprocessor are refit independently on
    each training fold to prevent target leakage. ``backend="thread"`` retains
    Optuna's in-process concurrency; ``backend="process"`` coordinates
    subprocess workers through JournalStorage.
    """
    n_trials: int = 20
    cv_folds: int = 5
    metric: Optional[str] = None
    seed: int = 42
    use_data_folds: Optional[bool] = None
    cv: Optional[CVConfig] = None
    n_jobs: int = 1
    backend: Literal["thread", "process"] = "thread"
    journal_path: Optional[str | os.PathLike[str]] = None
    show_progress_bar: bool = True

    def tune(
        self,
        data: ModelData,
        model: Any,
        encoder: Optional[Any] = None,
        selector: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        preprocessors: Optional[list[Any]] = None,
        schema: Optional[Any] = None,
        *,
        feature_names: Optional[list[str]] = None,
        model_selected_features: Optional[list[str]] = None,
        progress: Optional[ProgressCallback] = None,
        should_stop: Optional[Any] = None,
        base_params: Optional[dict] = None,
    ) -> tuple[dict, pl.DataFrame]:
        """Run hyperparameter search and return (best_params, trial_history).

        Parameters
        ----------
        data : ModelData
            Training data. Must not include the test set.
        model : BaseModel
            Unfitted model providing ``default_search_space()`` and ``fit()``.
        encoder : optional
            Unfitted encoder (e.g. OneHotEncoder). Fit on each fold's train split.
        selector : optional
            Unfitted feature selector. Fit on each fold's train split.
        preprocessor : optional
            Deprecated singular preprocessor retained for compatibility.
        preprocessors : optional
            Unfitted preprocessing chain. Each item is fit on each fold's train
            split and then applied to both train and validation data.
        schema : optional
            FeatureSchema passed to encoder.fit() when encoder is provided.
        feature_names : optional
            Ordered subset of raw features to use for every trial and fold.

        Returns
        -------
        best_params : dict
            Hyperparameters from the best trial.
        trial_history : pl.DataFrame
            One row per completed trial with columns ``trial``, ``value``,
            plus one column per hyperparameter.
        """
        import optuna
        from ins_gbm.data.folds import resolve_folds

        if (
            not isinstance(self.n_jobs, int)
            or isinstance(self.n_jobs, bool)
            or self.n_jobs == 0
            or self.n_jobs < -1
        ):
            raise ValueError("n_jobs must be -1 or a positive integer")
        if self.backend not in {"thread", "process"}:
            raise ValueError("backend must be 'thread' or 'process'")
        if self.backend == "thread" and self.journal_path is not None:
            raise ValueError(
                "journal_path is only supported with backend='process'"
            )
        metric = self.metric or (
            "gamma_deviance" if (getattr(model, "objective", None) or data.objective) == "gamma"
            else "poisson_deviance"
        )
        if metric not in _METRIC_FN:
            raise ValueError(
                f"Unknown metric: {metric!r}. Choose from {list(_METRIC_FN)}"
            )
        objective = getattr(model, "objective", None) or data.objective or "poisson"
        if metric in {"poisson_deviance", "gamma_deviance"} and not metric.startswith(objective):
            raise ValueError(f"metric {metric!r} is incompatible with objective {objective!r}")

        tuning_data = (
            data.select_features(feature_names)
            if feature_names is not None
            else data
        )
        encoder_schema = _select_schema(
            schema if schema is not None else tuning_data.schema,
            feature_names,
        )

        search_space = model.default_search_space()
        if preprocessors is not None and preprocessor is not None:
            raise ValueError("Pass either preprocessor or preprocessors, not both")
        preprocessing_chain = (
            list(preprocessors)
            if preprocessors is not None
            else ([preprocessor] if preprocessor is not None else [])
        )
        from ins_gbm.preprocessing.steps import validate_preprocessing_steps

        validate_preprocessing_steps(preprocessing_chain)

        if self.cv is not None and self.use_data_folds is not None:
            raise ValueError("Pass either cv or use_data_folds, not both")
        cv_config = self.cv or CVConfig(
            n_splits=self.cv_folds,
            seed=self.seed,
            folds=(
                "auto" if self.use_data_folds is None
                else ("predefined" if self.use_data_folds else "random")
            ),
        )
        _, fold_splits = resolve_folds(tuning_data, cv_config)

        config = _ObjectiveConfig(
            tuning_data=tuning_data,
            model=model,
            encoder=encoder,
            selector=selector,
            preprocessing_chain=preprocessing_chain,
            encoder_schema=encoder_schema,
            model_selected_features=model_selected_features,
            fold_splits=fold_splits,
            search_space=search_space,
            metric=metric,
            base_params=dict(base_params or {}),
        )

        stop_lock = Lock()

        def stop_requested() -> bool:
            if should_stop is None:
                return False
            with stop_lock:
                return bool(should_stop())

        trial_progress = tqdm(
            total=self.n_trials,
            desc="Hyperparameter tuning",
            unit="trial",
            disable=not self.show_progress_bar,
        )

        try:
            if self.backend == "thread":
                study = self._optimize_threads(
                    optuna=optuna,
                    config=config,
                    trial_progress=trial_progress,
                    progress=progress,
                    stop_requested=stop_requested,
                    has_should_stop=should_stop is not None,
                )
                best_params = study.best_params
                history = _study_history(study)
            else:
                best_params, history = self._optimize_processes(
                    optuna=optuna,
                    config=config,
                    trial_progress=trial_progress,
                    progress=progress,
                    stop_requested=stop_requested,
                )
        finally:
            trial_progress.close()

        return {**dict(base_params or {}), **best_params}, history

    def _optimize_threads(
        self,
        *,
        optuna: Any,
        config: _ObjectiveConfig,
        trial_progress: Any,
        progress: Optional[ProgressCallback],
        stop_requested: Callable[[], bool],
        has_should_stop: bool,
    ) -> Any:
        """Run the existing in-process Optuna thread backend."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(),
        )
        callbacks = []
        if self.show_progress_bar or progress is not None or has_should_stop:
            callback_lock = Lock()

            def _on_trial(study: Any, trial: Any) -> None:
                # Optuna invokes callbacks from worker threads when n_jobs > 1.
                with callback_lock:
                    trial_progress.update()
                    if progress is not None and trial.value is not None:
                        finished = sum(
                            frozen.state.is_finished()
                            for frozen in study.trials
                        )
                        progress(ProgressEvent(
                            stage="tuning",
                            message=f"trial {trial.number} complete",
                            current=finished,
                            total=self.n_trials,
                            payload={
                                "trial_value": trial.value,
                                "best_value": study.best_value,
                            },
                        ))
                    if stop_requested():
                        study.stop()

            callbacks.append(_on_trial)

        study.optimize(
            lambda trial: _evaluate_trial(trial, config, stop_requested),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            callbacks=callbacks,
        )
        return study

    def _optimize_processes(
        self,
        *,
        optuna: Any,
        config: _ObjectiveConfig,
        trial_progress: Any,
        progress: Optional[ProgressCallback],
        stop_requested: Callable[[], bool],
    ) -> tuple[dict, pl.DataFrame]:
        """Run trials in subprocesses coordinated by JournalStorage."""
        import cloudpickle

        worker_count = (
            os.cpu_count() or 1
            if self.n_jobs == -1
            else self.n_jobs
        )
        worker_count = min(worker_count, self.n_trials)
        if worker_count < 1:
            raise ValueError("n_trials must be a positive integer")

        with tempfile.TemporaryDirectory(prefix="ins-gbm-optuna-") as temp_dir:
            temp_path = Path(temp_dir)
            if self.journal_path is None:
                journal_path = temp_path / "journal.log"
            else:
                journal_path = Path(self.journal_path).expanduser().resolve()
                journal_path.parent.mkdir(parents=True, exist_ok=True)

            cancellation_path = temp_path / "cancel"
            config.cancellation_path = str(cancellation_path)
            payload_path = temp_path / "payload.pkl"
            try:
                payload = cloudpickle.dumps(config)
            except Exception as exc:
                raise TypeError(
                    "Process tuning inputs must be cloudpickle-serializable"
                ) from exc
            payload_path.write_bytes(payload)

            storage = _create_journal_storage(str(journal_path))
            study_name = f"ins-gbm-{uuid.uuid4()}"
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                storage=storage,
                sampler=optuna.samplers.TPESampler(seed=self.seed),
                pruner=optuna.pruners.MedianPruner(),
            )

            base_trials, extra_trials = divmod(self.n_trials, worker_count)
            quotas = [
                base_trials + (worker_index < extra_trials)
                for worker_index in range(worker_count)
            ]
            processes: list[subprocess.Popen] = []
            log_handles: list[Any] = []
            log_paths: list[Path] = []

            try:
                for worker_index, quota in enumerate(quotas):
                    log_path = temp_path / f"worker-{worker_index}.log"
                    log_handle = log_path.open("w", encoding="utf-8")
                    command = [
                        sys.executable,
                        "-m",
                        "ins_gbm.tuning._process_worker",
                        str(payload_path),
                        str(journal_path),
                        study_name,
                        str(quota),
                        str(self.seed + worker_index),
                    ]
                    processes.append(subprocess.Popen(
                        command,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    ))
                    log_handles.append(log_handle)
                    log_paths.append(log_path)

                seen_trials: set[int] = set()
                cancelled = False
                while True:
                    if stop_requested() and not cancelled:
                        cancellation_path.touch()
                        cancelled = True

                    self._report_process_progress(
                        study=study,
                        seen_trials=seen_trials,
                        trial_progress=trial_progress,
                        progress=progress,
                    )

                    return_codes = [process.poll() for process in processes]
                    failed_index = next(
                        (
                            index
                            for index, code in enumerate(return_codes)
                            if code not in (None, 0)
                        ),
                        None,
                    )
                    if failed_index is not None and not cancelled:
                        cancellation_path.touch(exist_ok=True)
                        for process in processes:
                            if process.poll() is None:
                                process.terminate()
                        for process in processes:
                            process.wait()
                        for handle in log_handles:
                            handle.flush()
                        details = log_paths[failed_index].read_text(
                            encoding="utf-8"
                        )
                        raise RuntimeError(
                            f"Hyperparameter worker {failed_index} failed:\n"
                            f"{details.strip()}"
                        )
                    if all(code is not None for code in return_codes):
                        break
                    time.sleep(0.05)

                self._report_process_progress(
                    study=study,
                    seen_trials=seen_trials,
                    trial_progress=trial_progress,
                    progress=progress,
                )
                if cancelled:
                    raise PipelineCancelled("cancelled during hyperparameter tuning")
                # Materialize results while a temporary journal still exists.
                return study.best_params, _study_history(study)
            except BaseException:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    process.wait()
                raise
            finally:
                for handle in log_handles:
                    handle.close()

    def _report_process_progress(
        self,
        *,
        study: Any,
        seen_trials: set[int],
        trial_progress: Any,
        progress: Optional[ProgressCallback],
    ) -> None:
        """Replay newly finished JournalStorage trials in the parent process."""
        finished_trials = [
            trial
            for trial in study.get_trials(deepcopy=False)
            if trial.state.is_finished() and trial.number not in seen_trials
        ]
        finished_trials.sort(
            key=lambda trial: (
                trial.datetime_complete or trial.datetime_start,
                trial.number,
            )
        )
        for trial in finished_trials:
            seen_trials.add(trial.number)
            trial_progress.update()
            if progress is not None and trial.value is not None:
                try:
                    best_value = study.best_value
                except ValueError:
                    best_value = trial.value
                progress(ProgressEvent(
                    stage="tuning",
                    message=f"trial {trial.number} complete",
                    current=len(seen_trials),
                    total=self.n_trials,
                    payload={
                        "trial_value": trial.value,
                        "best_value": best_value,
                    },
                ))
