from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from ins_gbm.data.dtypes import frame_to_fit_array, series_to_fit_array
from ins_gbm.data.model_data import ModelData
from ins_gbm.models.base import FittedModel, ModelCapabilities, resolve_objective
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

    Args:
        objective (Optional[Objective]): Model objective: "poisson" or "gamma". Optional.
    """

    objective: Objective | None = None

    def capabilities(self) -> ModelCapabilities:
        """Describe supported objectives and model features."""
        return ModelCapabilities(
            supports_poisson=True,
            supports_gamma=True,
            supports_offset=True,
            supports_sample_weight=True,
            supports_feature_importance=True,
        )

    def default_search_space(self) -> dict:
        """Return Optuna distributions for tunable model parameters."""
        import optuna

        return {
            "n_estimators": optuna.distributions.IntDistribution(50, 500),
            "learning_rate": optuna.distributions.FloatDistribution(
                0.01, 0.3, log=True
            ),
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
        params: dict | None = None,
        *,
        feature_names: list[str] | None = None,
        encoder: object | None = None,
        preprocessing: list[object] | None = None,
    ) -> FittedModel:
        """Fit the model on training data and return a fitted wrapper.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
            params (Optional[dict]): Optional model or estimator parameter mapping.
            feature_names (Optional[list[str]]): Ordered names of input features to use. Optional.
            encoder (Optional[object]): Optional encoder applied before model fitting.
            preprocessing (Optional[list[object]]): Ordered preprocessing steps applied before
                fitting. Optional.
        """
        import xgboost as xgb

        transform_result = fit_transform_chain(
            data,
            feature_names=feature_names,
            encoder=encoder,
            preprocessing=preprocessing,
        )
        data = transform_result.data
        objective = resolve_objective(self.objective, data)

        p = dict(params or {})
        p.setdefault("objective", _XGB_OBJECTIVE[objective])
        p.setdefault("verbosity", 0)

        X = frame_to_fit_array(data.features, data.feature_names)
        y = series_to_fit_array(data.target)

        margin_parts: list[np.ndarray] = []
        if objective == "poisson" and data.exposure is not None:
            margin_parts.append(np.log(series_to_fit_array(data.exposure)))
        if data.offset is not None:
            margin_parts.append(series_to_fit_array(data.offset))
        base_margin = np.sum(margin_parts, axis=0) if margin_parts else None

        sample_weight: np.ndarray | None = None
        if data.weight is not None:
            sample_weight = series_to_fit_array(data.weight)

        n_estimators = p.pop("n_estimators", 100)

        dtrain_kwargs = {
            "label": y,
            "weight": sample_weight,
            "feature_names": list(data.feature_names),
            "missing": _NUMERIC_FILL,
        }
        if base_margin is not None:
            dtrain_kwargs["base_margin"] = base_margin
        dtrain = xgb.DMatrix(X, **dtrain_kwargs)

        booster = xgb.train(
            params=p,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            verbose_eval=False,
        )

        feature_names = list(data.feature_names)

        def _predict(pred_data: ModelData, prediction_type: str) -> pl.Series:
            """Predict from the fitted estimator on the requested scale.

            Args:
                pred_data (ModelData): Prepared model data to score.
                prediction_type (str): Prediction scale: "response", "rate", or "link"; "rate" is
                    unavailable for Gamma.
            """
            X_pred = frame_to_fit_array(pred_data.features, pred_data.feature_names)

            pred_margin_parts: list[np.ndarray] = []
            if objective == "poisson" and pred_data.exposure is not None:
                pred_margin_parts.append(
                    np.log(series_to_fit_array(pred_data.exposure))
                )
            if pred_data.offset is not None:
                pred_margin_parts.append(series_to_fit_array(pred_data.offset))
            pred_margin = (
                np.sum(pred_margin_parts, axis=0) if pred_margin_parts else None
            )

            dtest_kwargs = {
                "feature_names": feature_names,
                "missing": _NUMERIC_FILL,
            }
            if pred_margin is not None:
                dtest_kwargs["base_margin"] = pred_margin
            dtest = xgb.DMatrix(X_pred, **dtest_kwargs)
            link = booster.predict(dtest, output_margin=True)
            response = np.exp(link)

            if objective == "poisson":
                if prediction_type == "response":
                    return pl.Series(response)
                elif prediction_type == "rate":
                    if pred_data.exposure is not None:
                        return pl.Series(response / pred_data.exposure.to_numpy())
                    return pl.Series(response)
                else:  # link
                    return pl.Series(link)
            return pl.Series(link if prediction_type == "link" else response)

        def _importance(importance_type: str | None = None) -> pl.DataFrame:
            """Return feature importance from the fitted estimator.

            Args:
                importance_type (Optional[str]): Optional framework-specific importance measure.
            """
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
            objective=objective,
            feature_names=feature_names,
            predict_fn=_predict,
            importance_fn=_importance,
            transform_chain=transform_result.chain,
        )
