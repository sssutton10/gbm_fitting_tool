from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData, slice_model_data
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


@dataclass
class HyperparameterTuner:
    """Optuna-based hyperparameter tuner with CV fold evaluation.

    For each trial, encoder/selector/preprocessor are refit independently on
    each training fold to prevent target leakage. Trials run concurrently when
    ``n_jobs`` is greater than one.
    """
    n_trials: int = 20
    cv_folds: int = 5
    metric: str = "poisson_deviance"
    seed: int = 42
    use_data_folds: bool = False
    n_jobs: int = 1

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
        progress: Optional[ProgressCallback] = None,
        should_stop: Optional[Any] = None,
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
        from sklearn.model_selection import KFold

        if (
            not isinstance(self.n_jobs, int)
            or isinstance(self.n_jobs, bool)
            or self.n_jobs == 0
            or self.n_jobs < -1
        ):
            raise ValueError("n_jobs must be -1 or a positive integer")
        if self.metric not in _METRIC_FN:
            raise ValueError(
                f"Unknown metric: {self.metric!r}. Choose from {list(_METRIC_FN)}"
            )

        tuning_data = (
            data.select_features(feature_names)
            if feature_names is not None
            else data
        )
        encoder_schema = _select_schema(
            schema if schema is not None else tuning_data.schema,
            feature_names,
        )

        metric_fn = _METRIC_FN[self.metric]
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

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(),
        )

        if self.use_data_folds:
            if tuning_data.cv_fold is None:
                raise ValueError("use_data_folds=True but data.cv_fold is None")
            folds_arr = tuning_data.cv_fold.to_numpy()
            unique_folds = np.unique(folds_arr)
            fold_splits = [
                (np.where(folds_arr != f)[0], np.where(folds_arr == f)[0])
                for f in unique_folds
            ]
        else:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
            fold_splits = list(kf.split(range(tuning_data.n_rows)))

        stop_lock = Lock()

        def stop_requested() -> bool:
            if should_stop is None:
                return False
            with stop_lock:
                return bool(should_stop())

        def objective(trial: Any) -> float:
            params = {
                name: _suggest_from_distribution(trial, name, dist)
                for name, dist in search_space.items()
            }

            fold_scores: list[float] = []
            for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
                if stop_requested():
                    raise PipelineCancelled("cancelled during CV fold")

                train_data = slice_model_data(tuning_data, train_idx)
                val_data = slice_model_data(tuning_data, val_idx)

                transform_result = fit_transform_chain(
                    train_data,
                    encoder=encoder,
                    selector=selector,
                    preprocessing=preprocessing_chain,
                    schema=encoder_schema,
                )
                train_data = transform_result.data
                val_data = transform_result.chain.transform(val_data)

                fitted_model = model.fit(train_data, params=params)
                preds = fitted_model.predict(val_data, prediction_type="response")

                metric_actual = val_data.target
                metric_predicted = preds
                if (
                    val_data.objective == "poisson"
                    and self.metric == "poisson_deviance"
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
                    # Response predictions are expected counts for frequency
                    # models, so exposure is not also a count-error weight.
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

        optuna_callbacks = []
        if progress is not None or should_stop is not None:
            callback_lock = Lock()

            def _on_trial(study: Any, trial: Any) -> None:
                # Optuna invokes callbacks from worker threads when n_jobs > 1.
                # Serialize user callbacks and progress accounting.
                with callback_lock:
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
            optuna_callbacks.append(_on_trial)

        try:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                n_jobs=self.n_jobs,
                callbacks=optuna_callbacks,
            )
        except PipelineCancelled:
            raise

        rows = []
        for t in study.trials:
            if t.value is not None:
                row: dict = {"trial": t.number, "value": t.value}
                row.update(t.params)
                rows.append(row)

        if rows:
            history = pl.DataFrame(rows)
        else:
            history = pl.DataFrame({"trial": pl.Series([], dtype=pl.Int64),
                                    "value": pl.Series([], dtype=pl.Float64)})

        return study.best_params, history
