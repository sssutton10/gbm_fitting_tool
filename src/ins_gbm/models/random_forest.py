from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData
from ins_gbm.models.base import FittedModel, ModelCapabilities


Objective = Literal["poisson", "gamma"]


@dataclass
class RandomForestModel:
    """Random Forest benchmark model.

    Does not support native exposure offsets. For Poisson frequency, exposure
    is incorporated via sample weights (exposure-weighted MSE), which is an
    approximation. For Gamma severity, log-transformed target with MSE is used.
    Both are documented limitations — this model is a benchmark, not a GLM-style
    objective wrapper.
    """
    objective: Objective = "poisson"

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_poisson=True,
            supports_gamma=True,
            supports_offset=False,  # no native log-offset support
            supports_sample_weight=True,
            supports_feature_importance=True,
        )

    def default_search_space(self) -> dict:
        import optuna
        return {
            "n_estimators": optuna.distributions.IntDistribution(50, 300),
            "max_depth": optuna.distributions.IntDistribution(3, 15),
            "min_samples_leaf": optuna.distributions.IntDistribution(5, 50),
            "max_features": optuna.distributions.FloatDistribution(0.3, 1.0),
        }

    def fit(
        self,
        data: ModelData,
        params: Optional[dict] = None,
    ) -> FittedModel:
        from sklearn.ensemble import RandomForestRegressor

        p = dict(params or {})
        p.setdefault("random_state", 42)

        X = data.features.select(data.feature_names).to_numpy().astype(np.float64)
        y = data.target.to_numpy().astype(np.float64)

        # Approximate Poisson objective: weight by exposure, fit on claim rate
        if self.objective == "poisson" and data.exposure is not None:
            exposure = data.exposure.to_numpy().astype(np.float64)
            y_fit = y / exposure  # fit on claim rate
            sample_weight = exposure
        elif data.weight is not None:
            y_fit = y
            sample_weight = data.weight.to_numpy().astype(np.float64)
        else:
            y_fit = y
            sample_weight = None

        rf = RandomForestRegressor(**p)
        rf.fit(X, y_fit, sample_weight=sample_weight)

        feature_names = list(data.feature_names)
        objective = self.objective

        def _predict(pred_data: ModelData, prediction_type: str) -> pl.Series:
            X_pred = pred_data.features.select(pred_data.feature_names).to_numpy().astype(np.float64)
            raw = rf.predict(X_pred)  # predicted claim rate or severity

            if objective == "poisson":
                if prediction_type == "response":
                    if pred_data.exposure is not None:
                        return pl.Series(raw * pred_data.exposure.to_numpy())
                    return pl.Series(raw)
                elif prediction_type == "rate":
                    return pl.Series(raw)
                else:  # link
                    return pl.Series(np.log(np.maximum(raw, 1e-10)))
            else:  # gamma
                return pl.Series(np.maximum(raw, 1e-10))

        def _importance(importance_type: Optional[str] = None) -> pl.DataFrame:
            importance_type = importance_type or "impurity"
            if importance_type != "impurity":
                raise ValueError(
                    "RandomForest importance_type must be 'impurity'"
                )
            return pl.DataFrame({
                "feature": feature_names,
                "importance": rf.feature_importances_.astype(float).tolist(),
            })

        return FittedModel(
            model=rf,
            params=p,
            framework="random_forest",
            objective=self.objective,
            feature_names=feature_names,
            predict_fn=_predict,
            importance_fn=_importance,
        )
