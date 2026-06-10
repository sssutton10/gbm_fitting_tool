from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData
from ins_gbm.models.base import FittedModel, ModelCapabilities
from ins_gbm.preprocessing.encoder import _NUMERIC_FILL


Objective = Literal["poisson", "gamma"]

_LGB_OBJECTIVE = {
    "poisson": "poisson",
    "gamma": "gamma",
}


@dataclass
class LightGBMModel:
    """LightGBM wrapper for Poisson (frequency) and Gamma (severity) objectives.

    Missing values
    --------------
    Expects numeric features pre-filled with ``_NUMERIC_FILL`` (``-999_999_999.0``).
    Before constructing the ``Dataset``, the wrapper converts that sentinel back
    to ``NaN`` so LightGBM can apply its native missing-value branch logic
    (learns the optimal direction at each split).
    """
    objective: Objective = "poisson"

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_poisson=True,
            supports_gamma=True,
            supports_offset=True,
            supports_sample_weight=True,
            supports_feature_importance=True,
        )

    def default_search_space(self) -> dict:
        import optuna
        return {
            "n_estimators": optuna.distributions.IntDistribution(50, 500),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
            "num_leaves": optuna.distributions.IntDistribution(16, 128),
            "min_child_samples": optuna.distributions.IntDistribution(10, 100),
            "subsample": optuna.distributions.FloatDistribution(0.5, 1.0),
            "colsample_bytree": optuna.distributions.FloatDistribution(0.5, 1.0),
            "reg_alpha": optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
            "reg_lambda": optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
        }

    def fit(
        self,
        data: ModelData,
        params: Optional[dict] = None,
    ) -> FittedModel:
        import lightgbm as lgb

        p = dict(params or {})
        p.setdefault("objective", _LGB_OBJECTIVE[self.objective])
        p.setdefault("verbose", -1)

        X = data.features.select(data.feature_names).to_numpy().astype(np.float64)
        X[X == _NUMERIC_FILL] = np.nan
        y = data.target.to_numpy().astype(np.float64)

        init_score_parts: list[np.ndarray] = []
        if self.objective == "poisson" and data.exposure is not None:
            init_score_parts.append(np.log(data.exposure.to_numpy().astype(np.float64)))
        if data.offset is not None:
            init_score_parts.append(data.offset.to_numpy().astype(np.float64))
        init_score: Optional[np.ndarray] = np.sum(init_score_parts, axis=0) if init_score_parts else None

        sample_weight: Optional[np.ndarray] = None
        if data.weight is not None:
            sample_weight = data.weight.to_numpy().astype(np.float64)

        n_estimators = p.pop("n_estimators", 100)

        ds = lgb.Dataset(
            X,
            label=y,
            init_score=init_score,
            weight=sample_weight,
            feature_name=list(data.feature_names),
            free_raw_data=False,
        )

        booster = lgb.train(
            params=p,
            train_set=ds,
            num_boost_round=n_estimators,
        )

        feature_names = list(data.feature_names)
        objective = self.objective

        def _predict(pred_data: ModelData, prediction_type: str) -> pl.Series:
            X_pred = pred_data.features.select(pred_data.feature_names).to_numpy().astype(np.float64)
            X_pred[X_pred == _NUMERIC_FILL] = np.nan
            raw_scores = booster.predict(X_pred)

            offset = (
                pred_data.offset.to_numpy().astype(np.float64)
                if pred_data.offset is not None
                else None
            )

            if objective == "poisson":
                # raw_scores = log(rate) on link scale; exposure and offset add on link scale
                link = raw_scores if offset is None else raw_scores + offset
                if prediction_type == "response":
                    exposure = (
                        pred_data.exposure.to_numpy()
                        if pred_data.exposure is not None
                        else 1.0
                    )
                    return pl.Series(np.exp(link) * exposure)
                elif prediction_type == "rate":
                    return pl.Series(np.exp(link))
                else:  # link
                    return pl.Series(link)
            else:  # gamma — raw_scores are on response scale (log link used internally)
                if prediction_type == "response":
                    if offset is not None:
                        return pl.Series(raw_scores * np.exp(offset))
                    return pl.Series(raw_scores)
                else:  # link = log(response) + offset
                    link = np.log(raw_scores)
                    if offset is not None:
                        link = link + offset
                    return pl.Series(link)

        def _importance() -> pl.DataFrame:
            names = booster.feature_name()
            scores = booster.feature_importance(importance_type="gain").astype(float)
            return pl.DataFrame({"feature": names, "importance": scores})

        return FittedModel(
            model=booster,
            params={**p, "n_estimators": n_estimators},
            framework="lightgbm",
            objective=self.objective,
            feature_names=feature_names,
            predict_fn=_predict,
            importance_fn=_importance,
        )
