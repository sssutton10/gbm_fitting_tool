from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import polars as pl

from ins_gbm.data.dtypes import (
    frame_to_fit_array,
    replace_value_with_nan,
    series_to_fit_array,
)
from ins_gbm.data.model_data import ModelData
from ins_gbm.models.base import FittedModel, ModelCapabilities, resolve_objective
from ins_gbm.preprocessing.chain import fit_transform_chain
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
    With no encoder, expects numeric features ready for model fitting. When an
    encoder is supplied to :meth:`fit`, raw features are encoded at fit time.
    Encoded numeric values use ``_NUMERIC_FILL`` (``-999_999_999.0``).
    Before constructing the ``Dataset``, the wrapper converts that sentinel back
    to ``NaN`` so LightGBM can apply its native missing-value branch logic
    (learns the optimal direction at each split).
    """
    objective: Optional[Objective] = None

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
        *,
        feature_names: Optional[list[str]] = None,
        encoder: Optional[object] = None,
        preprocessing: Optional[list[object]] = None,
    ) -> FittedModel:
        import lightgbm as lgb

        transform_result = fit_transform_chain(
            data,
            feature_names=feature_names,
            encoder=encoder,
            preprocessing=preprocessing,
        )
        data = transform_result.data
        objective = resolve_objective(self.objective, data)

        p = dict(params or {})
        p.setdefault("objective", _LGB_OBJECTIVE[objective])
        p.setdefault("verbose", -1)

        X = frame_to_fit_array(data.features, data.feature_names)
        X = replace_value_with_nan(X, _NUMERIC_FILL)
        y = series_to_fit_array(data.target)

        init_score_parts: list[np.ndarray] = []
        if objective == "poisson" and data.exposure is not None:
            init_score_parts.append(np.log(series_to_fit_array(data.exposure)))
        if data.offset is not None:
            init_score_parts.append(series_to_fit_array(data.offset))
        init_score: Optional[np.ndarray] = np.sum(init_score_parts, axis=0) if init_score_parts else None

        sample_weight: Optional[np.ndarray] = None
        if data.weight is not None:
            sample_weight = series_to_fit_array(data.weight)

        n_estimators = p.pop("n_estimators", 100)

        dataset_kwargs = {
            "label": y,
            "weight": sample_weight,
            "feature_name": list(data.feature_names),
            "free_raw_data": True,
        }
        if init_score is not None:
            dataset_kwargs["init_score"] = init_score
        ds = lgb.Dataset(X, **dataset_kwargs)

        booster = lgb.train(
            params=p,
            train_set=ds,
            num_boost_round=n_estimators,
        )

        feature_names = list(data.feature_names)
        def _predict(pred_data: ModelData, prediction_type: str) -> pl.Series:
            X_pred = frame_to_fit_array(
                pred_data.features, pred_data.feature_names
            )
            X_pred = replace_value_with_nan(X_pred, _NUMERIC_FILL)
            raw_scores = booster.predict(X_pred)

            offset = (
                series_to_fit_array(pred_data.offset)
                if pred_data.offset is not None
                else None
            )

            if objective == "poisson":
                # raw_scores = log(rate) on link scale; exposure and offset add on link scale
                link = raw_scores if offset is None else raw_scores + offset
                if prediction_type == "response":
                    response = np.exp(link)
                    if pred_data.exposure is not None:
                        response = response * pred_data.exposure.to_numpy()
                    return pl.Series(response)
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

        def _importance(importance_type: Optional[str] = None) -> pl.DataFrame:
            importance_type = importance_type or "gain"
            if importance_type not in {"gain", "split"}:
                raise ValueError(
                    "LightGBM importance_type must be one of: 'gain', 'split'"
                )
            names = booster.feature_name()
            scores = booster.feature_importance(importance_type=importance_type).astype(float)
            return pl.DataFrame({"feature": names, "importance": scores})

        return FittedModel(
            model=booster,
            params={**p, "n_estimators": n_estimators},
            framework="lightgbm",
            objective=objective,
            feature_names=feature_names,
            predict_fn=_predict,
            importance_fn=_importance,
            transform_chain=transform_result.chain,
        )
