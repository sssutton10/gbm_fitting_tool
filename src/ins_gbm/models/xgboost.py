from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData
from ins_gbm.models.base import FittedModel, ModelCapabilities
from ins_gbm.preprocessing.chain import fit_transform_chain
from ins_gbm.preprocessing.encoder import _NUMERIC_FILL


Objective = Literal["poisson", "gamma"]

_XGB_OBJECTIVE = {
    "poisson": "count:poisson",
    "gamma": "reg:gamma",
}


@dataclass
class XGBoostModel:
    """XGBoost wrapper for Poisson (frequency) and Gamma (severity) objectives.

    Missing values
    --------------
    With no encoder, expects numeric features ready for model fitting. When an
    encoder is supplied to :meth:`fit`, raw features are encoded at fit time.
    Encoded numeric values use ``_NUMERIC_FILL`` (``-999_999_999.0``).
    Both ``DMatrix`` calls (train and predict) declare ``missing=_NUMERIC_FILL``
    so XGBoost treats that sentinel as missing and applies its sparse-aware
    split-finding rather than treating it as a real value.
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
            "max_depth": optuna.distributions.IntDistribution(3, 10),
            "min_child_weight": optuna.distributions.IntDistribution(1, 20),
            "subsample": optuna.distributions.FloatDistribution(0.5, 1.0),
            "colsample_bytree": optuna.distributions.FloatDistribution(0.5, 1.0),
            "reg_alpha": optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
            "reg_lambda": optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
        }

    def fit(
        self,
        data: ModelData,
        params: Optional[dict] = None,
        *,
        feature_names: Optional[list[str]] = None,
        encoder: Optional[object] = None,
        preprocessing: Optional[list[object]] = None,
    ) -> FittedModel:
        import xgboost as xgb

        transform_result = fit_transform_chain(
            data,
            feature_names=feature_names,
            encoder=encoder,
            preprocessing=preprocessing,
        )
        data = transform_result.data

        p = dict(params or {})
        p.setdefault("objective", _XGB_OBJECTIVE[self.objective])
        p.setdefault("verbosity", 0)

        X = data.features.select(data.feature_names).to_numpy().astype(np.float64)
        y = data.target.to_numpy().astype(np.float64)

        base_margin: Optional[np.ndarray] = None
        if self.objective == "poisson" and data.exposure is not None:
            base_margin = np.log(data.exposure.to_numpy().astype(np.float64))

        sample_weight: Optional[np.ndarray] = None
        if data.weight is not None:
            sample_weight = data.weight.to_numpy().astype(np.float64)

        n_estimators = p.pop("n_estimators", 100)

        dtrain = xgb.DMatrix(
            X,
            label=y,
            base_margin=base_margin,
            weight=sample_weight,
            feature_names=list(data.feature_names),
            missing=_NUMERIC_FILL,
        )

        booster = xgb.train(
            params=p,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            verbose_eval=False,
        )

        feature_names = list(data.feature_names)
        objective = self.objective

        def _predict(pred_data: ModelData, prediction_type: str) -> pl.Series:
            X_pred = pred_data.features.select(pred_data.feature_names).to_numpy().astype(np.float64)

            pred_margin: Optional[np.ndarray] = None
            if objective == "poisson" and pred_data.exposure is not None:
                pred_margin = np.log(pred_data.exposure.to_numpy().astype(np.float64))

            dtest = xgb.DMatrix(X_pred, base_margin=pred_margin, feature_names=feature_names, missing=_NUMERIC_FILL)
            raw = booster.predict(dtest)

            if objective == "poisson":
                if prediction_type == "response":
                    return pl.Series(raw)
                elif prediction_type == "rate":
                    if pred_data.exposure is not None:
                        return pl.Series(raw / pred_data.exposure.to_numpy())
                    return pl.Series(raw)
                else:  # link
                    return pl.Series(np.log(raw))
            else:  # gamma
                return pl.Series(raw)

        def _importance(importance_type: Optional[str] = None) -> pl.DataFrame:
            importance_type = importance_type or "gain"
            allowed = {"weight", "gain", "cover", "total_gain", "total_cover"}
            if importance_type not in allowed:
                raise ValueError(
                    "XGBoost importance_type must be one of: "
                    "'weight', 'gain', 'cover', 'total_gain', 'total_cover'"
                )
            scores_dict = booster.get_score(importance_type=importance_type)
            names = feature_names
            scores = [float(scores_dict.get(n, 0.0)) for n in names]
            return pl.DataFrame({"feature": names, "importance": scores})

        return FittedModel(
            model=booster,
            params={**p, "n_estimators": n_estimators},
            framework="xgboost",
            objective=self.objective,
            feature_names=feature_names,
            predict_fn=_predict,
            importance_fn=_importance,
            transform_chain=transform_result.chain,
        )
