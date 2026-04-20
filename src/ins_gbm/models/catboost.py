from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData
from ins_gbm.models.base import FittedModel, ModelCapabilities


Objective = Literal["poisson", "gamma"]

_CB_OBJECTIVE = {
    "poisson": "Poisson",
    # CatBoost Tweedie requires 1 < power < 2 strictly.
    # power=1.99 approximates Gamma (power=2) within this constraint.
    "gamma": "Tweedie:variance_power=1.99",
}


def _catboost_supports_offset() -> bool:
    """Check if installed CatBoost version supports the baseline (offset) parameter."""
    try:
        from catboost import CatBoostRegressor
        import inspect
        sig = inspect.signature(CatBoostRegressor.fit)
        return "baseline" in sig.parameters
    except Exception:
        return False


@dataclass
class CatBoostModel:
    objective: Objective = "poisson"

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_poisson=True,
            supports_gamma=True,
            supports_offset=_catboost_supports_offset(),
            supports_sample_weight=True,
            supports_feature_importance=True,
        )

    def default_search_space(self) -> dict:
        import optuna
        return {
            "iterations": optuna.distributions.IntDistribution(50, 500),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
            "depth": optuna.distributions.IntDistribution(3, 10),
            "l2_leaf_reg": optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
            "subsample": optuna.distributions.FloatDistribution(0.5, 1.0),
            "colsample_bylevel": optuna.distributions.FloatDistribution(0.5, 1.0),
        }

    def fit(
        self,
        data: ModelData,
        params: Optional[dict] = None,
    ) -> FittedModel:
        from catboost import CatBoostRegressor, Pool

        p = dict(params or {})
        p.setdefault("loss_function", _CB_OBJECTIVE[self.objective])
        p.setdefault("verbose", 0)
        p.setdefault("allow_writing_files", False)

        X = data.features.select(data.feature_names).to_numpy().astype(np.float64)
        y = data.target.to_numpy().astype(np.float64)

        baseline: Optional[np.ndarray] = None
        if self.objective == "poisson" and data.exposure is not None:
            if _catboost_supports_offset():
                baseline = np.log(data.exposure.to_numpy().astype(np.float64))

        sample_weight: Optional[np.ndarray] = None
        if data.weight is not None:
            sample_weight = data.weight.to_numpy().astype(np.float64)

        pool = Pool(
            data=X,
            label=y,
            baseline=baseline,
            weight=sample_weight,
            feature_names=list(data.feature_names),
        )

        model = CatBoostRegressor(**p)
        model.fit(pool)

        feature_names = list(data.feature_names)
        objective = self.objective
        has_offset = _catboost_supports_offset()

        def _predict(pred_data: ModelData, prediction_type: str) -> pl.Series:
            X_pred = pred_data.features.select(pred_data.feature_names).to_numpy().astype(np.float64)

            pred_baseline: Optional[np.ndarray] = None
            if objective == "poisson" and pred_data.exposure is not None and has_offset:
                pred_baseline = np.log(pred_data.exposure.to_numpy().astype(np.float64))

            pred_pool = Pool(data=X_pred, baseline=pred_baseline, feature_names=feature_names)
            raw = model.predict(pred_pool)

            if objective == "poisson":
                if prediction_type == "response":
                    return pl.Series(raw)
                elif prediction_type == "rate":
                    if pred_data.exposure is not None:
                        return pl.Series(raw / pred_data.exposure.to_numpy())
                    return pl.Series(raw)
                else:
                    return pl.Series(np.log(np.maximum(raw, 1e-10)))
            else:
                return pl.Series(raw)

        def _importance() -> pl.DataFrame:
            scores = model.get_feature_importance()
            return pl.DataFrame({"feature": feature_names, "importance": scores.astype(float).tolist()})

        return FittedModel(
            model=model,
            params=p,
            framework="catboost",
            objective=self.objective,
            feature_names=feature_names,
            predict_fn=_predict,
            importance_fn=_importance,
        )
