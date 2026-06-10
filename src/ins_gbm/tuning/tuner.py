from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.evaluation.metrics import poisson_deviance, gamma_deviance, rmse, mae
from ins_gbm.progress import ProgressCallback, ProgressEvent, PipelineCancelled


_METRIC_FN = {
    "poisson_deviance": poisson_deviance,
    "gamma_deviance": gamma_deviance,
    "rmse": rmse,
    "mae": mae,
}


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
    each training fold to prevent target leakage.
    """
    n_trials: int = 20
    cv_folds: int = 5
    metric: str = "poisson_deviance"
    seed: int = 42
    use_data_folds: bool = False

    def tune(
        self,
        data: ModelData,
        model: Any,
        encoder: Optional[Any] = None,
        selector: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        schema: Optional[Any] = None,
        *,
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
            Unfitted dimensionality reducer. Fit on each fold's train split.
        schema : optional
            FeatureSchema passed to encoder.fit() when encoder is provided.

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

        if self.metric not in _METRIC_FN:
            raise ValueError(
                f"Unknown metric: {self.metric!r}. Choose from {list(_METRIC_FN)}"
            )

        metric_fn = _METRIC_FN[self.metric]
        search_space = model.default_search_space()

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(),
        )

        if self.use_data_folds:
            if data.cv_fold is None:
                raise ValueError("use_data_folds=True but data.cv_fold is None")
            folds_arr = data.cv_fold.to_numpy()
            unique_folds = np.unique(folds_arr)
            fold_splits = [
                (np.where(folds_arr != f)[0], np.where(folds_arr == f)[0])
                for f in unique_folds
            ]
        else:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
            fold_splits = list(kf.split(range(data.n_rows)))

        def objective(trial: Any) -> float:
            params = {
                name: _suggest_from_distribution(trial, name, dist)
                for name, dist in search_space.items()
            }

            fold_scores: list[float] = []
            for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
                if should_stop is not None and should_stop():
                    raise PipelineCancelled("cancelled during CV fold")

                train_data = slice_model_data(data, train_idx)
                val_data = slice_model_data(data, val_idx)

                if encoder is not None:
                    fitted_enc = encoder.fit(train_data.features, schema)
                    train_data = train_data.with_features(
                        fitted_enc.transform(train_data.features)
                    )
                    val_data = val_data.with_features(
                        fitted_enc.transform(val_data.features)
                    )

                if selector is not None:
                    fitted_sel = selector.fit(train_data)
                    selected = fitted_sel.selected_features()
                    train_data = train_data.with_features(
                        train_data.features.select(selected)
                    )
                    val_data = val_data.with_features(
                        val_data.features.select(selected)
                    )

                if preprocessor is not None:
                    fitted_pre = preprocessor.fit(train_data.features)
                    train_data = train_data.with_features(
                        fitted_pre.transform(train_data.features)
                    )
                    val_data = val_data.with_features(
                        fitted_pre.transform(val_data.features)
                    )

                fitted_model = model.fit(train_data, params=params)
                preds = fitted_model.predict(val_data, prediction_type="response")

                weights = (
                    val_data.exposure
                    if val_data.exposure is not None
                    else val_data.weight
                )
                score = metric_fn(
                    val_data.target,
                    preds,
                    weights=weights,
                )
                fold_scores.append(score)

                trial.report(float(np.mean(fold_scores)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(fold_scores))

        optuna_callbacks = []
        if progress is not None or should_stop is not None:
            def _on_trial(study: Any, trial: Any) -> None:
                if progress is not None and trial.value is not None:
                    progress(ProgressEvent(
                        stage="tuning",
                        message=f"trial {trial.number} complete",
                        current=len(study.trials),
                        total=self.n_trials,
                        payload={
                            "trial_value": trial.value,
                            "best_value": study.best_value,
                        },
                    ))
                if should_stop is not None and should_stop():
                    study.stop()
            optuna_callbacks.append(_on_trial)

        try:
            study.optimize(objective, n_trials=self.n_trials, callbacks=optuna_callbacks)
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
