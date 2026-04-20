from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData
from ins_gbm.evaluation.metrics import poisson_deviance, gamma_deviance, rmse, mae


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


def _slice_model_data(data: ModelData, indices: np.ndarray) -> ModelData:
    return ModelData(
        features=data.features[indices],
        target=data.target[indices],
        exposure=data.exposure[indices] if data.exposure is not None else None,
        weight=data.weight[indices] if data.weight is not None else None,
        feature_names=data.feature_names,
        schema=data.schema,
        objective=data.objective,
    )


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

    def tune(
        self,
        data: ModelData,
        model: Any,
        encoder: Optional[Any] = None,
        selector: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        schema: Optional[Any] = None,
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

        indices = np.arange(data.n_rows)
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        fold_splits = list(kf.split(indices))

        def objective(trial: Any) -> float:
            params = {
                name: _suggest_from_distribution(trial, name, dist)
                for name, dist in search_space.items()
            }

            fold_scores: list[float] = []
            for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
                train_data = _slice_model_data(data, train_idx)
                val_data = _slice_model_data(data, val_idx)

                # Encoder: fit on train fold only
                if encoder is not None:
                    fitted_enc = encoder.fit(train_data.features, schema)
                    train_data = train_data.with_features(
                        fitted_enc.transform(train_data.features)
                    )
                    val_data = val_data.with_features(
                        fitted_enc.transform(val_data.features)
                    )

                # Selector: fit on train fold only
                if selector is not None:
                    fitted_sel = selector.fit(train_data)
                    selected = fitted_sel.selected_features()
                    train_data = train_data.with_features(
                        train_data.features.select(selected)
                    )
                    val_data = val_data.with_features(
                        val_data.features.select(selected)
                    )

                # Preprocessor: fit on train fold only
                if preprocessor is not None:
                    fitted_pre = preprocessor.fit(train_data.features)
                    train_data = train_data.with_features(
                        fitted_pre.transform(train_data.features)
                    )
                    val_data = val_data.with_features(
                        fitted_pre.transform(val_data.features)
                    )

                # Fit model and predict
                fitted_model = model.fit(train_data, params=params)
                preds = fitted_model.predict(val_data, prediction_type="response")

                # Compute metric (metrics.py expects pl.Series)
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

                # Report intermediate value for pruning
                trial.report(float(np.mean(fold_scores)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(fold_scores))

        study.optimize(objective, n_trials=self.n_trials)

        # Build trial history — one row per completed trial
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
