from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from ins_gbm.data.dtypes import frame_to_fit_array, series_to_fit_array
from ins_gbm.data.model_data import ModelData
from ins_gbm.models.base import FittedModel, ModelCapabilities, resolve_objective
from ins_gbm.preprocessing.chain import fit_transform_chain

Objective = Literal["poisson", "gamma"]


@dataclass
class RandomForestModel:
    """Random Forest benchmark model.

    Does not support native exposure offsets. For Poisson frequency, exposure
    is incorporated via sample weights (exposure-weighted MSE), which is an
    approximation. For Gamma severity, the untransformed target is fit with MSE.
    Both are documented limitations — this model is a benchmark, not a GLM-style
    objective wrapper.

    Args:
        objective (Optional[Objective]): Model objective: "poisson" or "gamma". Optional.
    """

    objective: Objective | None = None

    def capabilities(self) -> ModelCapabilities:
        """Describe supported objectives and model features."""
        return ModelCapabilities(
            supports_poisson=True,
            supports_gamma=True,
            supports_offset=False,  # no native log-offset support
            supports_sample_weight=True,
            supports_feature_importance=True,
        )

    def default_search_space(self) -> dict:
        """Return Optuna distributions for tunable model parameters."""
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
        from sklearn.ensemble import RandomForestRegressor

        transform_result = fit_transform_chain(
            data,
            feature_names=feature_names,
            encoder=encoder,
            preprocessing=preprocessing,
        )
        data = transform_result.data
        objective = resolve_objective(self.objective, data)
        if data.offset is not None:
            raise ValueError("RandomForestModel does not support offsets")

        p = dict(params or {})
        p.setdefault("random_state", 42)

        X = frame_to_fit_array(data.features, data.feature_names)
        y = series_to_fit_array(data.target)

        # Approximate Poisson objective: weight by exposure, fit on claim rate
        if objective == "poisson" and data.exposure is not None:
            exposure = series_to_fit_array(data.exposure)
            y_fit = y / exposure  # fit on claim rate
            sample_weight = exposure
            if data.weight is not None:
                sample_weight = sample_weight * series_to_fit_array(data.weight)
        elif data.weight is not None:
            y_fit = y
            sample_weight = series_to_fit_array(data.weight)
        else:
            y_fit = y
            sample_weight = None

        rf = RandomForestRegressor(**p)
        if sample_weight is None:
            rf.fit(X, y_fit)
        else:
            rf.fit(X, y_fit, sample_weight=sample_weight)

        feature_names = list(data.feature_names)

        def _predict(pred_data: ModelData, prediction_type: str) -> pl.Series:
            """Predict from the fitted estimator on the requested scale.

            Args:
                pred_data (ModelData): Prepared model data to score.
                prediction_type (str): Prediction scale: "response", "rate", or "link"; "rate" is
                    unavailable for Gamma.
            """
            if pred_data.offset is not None:
                raise ValueError("RandomForestModel does not support offsets")
            X_pred = frame_to_fit_array(pred_data.features, pred_data.feature_names)
            raw = rf.predict(X_pred)  # predicted claim rate or severity

            if objective == "poisson":
                if prediction_type == "response":
                    if pred_data.exposure is not None:
                        response = raw * pred_data.exposure.to_numpy()
                    else:
                        response = raw
                    # A zero-rate leaf is a valid random-forest estimate, but
                    # Poisson deviance requires strictly positive means.
                    return pl.Series(np.maximum(response, 1e-10))
                elif prediction_type == "rate":
                    return pl.Series(np.maximum(raw, 1e-10))
                else:  # link is log expected response, including exposure
                    response = raw
                    if pred_data.exposure is not None:
                        response = response * pred_data.exposure.to_numpy()
                    return pl.Series(np.log(np.maximum(response, 1e-10)))
            else:  # gamma
                response = np.maximum(raw, 1e-10)
                return pl.Series(
                    np.log(response) if prediction_type == "link" else response
                )

        def _importance(importance_type: str | None = None) -> pl.DataFrame:
            """Return feature importance from the fitted estimator.

            Args:
                importance_type (Optional[str]): Optional framework-specific importance measure.
            """
            importance_type = importance_type or "impurity"
            if importance_type != "impurity":
                raise ValueError("RandomForest importance_type must be 'impurity'")
            return pl.DataFrame(
                {
                    "feature": feature_names,
                    "importance": rf.feature_importances_.astype(float).tolist(),
                }
            )

        return FittedModel(
            model=rf,
            params=p,
            framework="random_forest",
            objective=objective,
            feature_names=feature_names,
            predict_fn=_predict,
            importance_fn=_importance,
            transform_chain=transform_result.chain,
        )
