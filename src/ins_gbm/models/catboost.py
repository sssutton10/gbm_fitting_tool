from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
from ins_gbm.preprocessing.encoder import _MISSING_LEVEL, _NUMERIC_FILL

Objective = Literal["poisson", "gamma"]

_CB_OBJECTIVE = {
    "poisson": "Poisson",
    # CatBoost Tweedie requires 1 < power < 2 strictly.
    # power=1.99 approximates Gamma (power=2) within this constraint.
    "gamma": "Tweedie:variance_power=1.99",
}


def _pool_matrix(
    features: pl.DataFrame,
    feature_names: list[str],
    categorical_names: list[str],
) -> np.ndarray:
    """Keep native categories as strings and numeric features as floats."""
    if not categorical_names:
        return replace_value_with_nan(
            frame_to_fit_array(features, feature_names), _NUMERIC_FILL
        )

    categorical = set(categorical_names)
    matrix = np.empty((features.height, len(feature_names)), dtype=object)
    for index, name in enumerate(feature_names):
        if name in categorical:
            values = pl.col(name)
            if features.schema[name].is_float():
                values = values.fill_nan(None)
            matrix[:, index] = (
                features.select(values.cast(pl.Utf8).fill_null(_MISSING_LEVEL))
                .to_series()
                .to_list()
            )
        else:
            numeric = frame_to_fit_array(features, [name])[:, 0]
            matrix[:, index] = replace_value_with_nan(numeric, _NUMERIC_FILL)
    return matrix


def _catboost_supports_offset() -> bool:
    """Check if installed CatBoost version supports the baseline (offset) parameter."""
    try:
        import inspect

        from catboost import CatBoostRegressor

        sig = inspect.signature(CatBoostRegressor.fit)
        return "baseline" in sig.parameters
    except Exception:  # noqa: BLE001 - optional dependency introspection may fail in several ways
        return False


@dataclass
class CatBoostModel:
    """CatBoost wrapper for Poisson (frequency) and Gamma (severity) objectives.

    Missing values
    --------------
    With no encoder, categorical features are passed to CatBoost natively and
    other features must be numeric. When an encoder is supplied to :meth:`fit`,
    raw features are encoded at fit time.
    Encoded numeric values use ``_NUMERIC_FILL`` (``-999_999_999.0``).
    Before constructing the ``Pool``, the wrapper converts that sentinel back to
    ``NaN`` so CatBoost can apply its native missing-value handling.

    Args:
        objective (Optional[Objective]): Model objective: "poisson" or "gamma". Optional.
        categorical_features (list[str] | "auto"): Categorical feature names, or
            "auto" to use the surviving names in ``ModelData.schema.categorical``.
            Explicit names can mark numeric columns as categorical.
    """

    objective: Objective | None = None
    categorical_features: list[str] | Literal["auto"] = "auto"

    def capabilities(self) -> ModelCapabilities:
        """Describe supported objectives and model features."""
        return ModelCapabilities(
            supports_poisson=True,
            supports_gamma=True,
            supports_offset=_catboost_supports_offset(),
            supports_sample_weight=True,
            supports_feature_importance=True,
        )

    def default_search_space(self) -> dict:
        """Return Optuna distributions for tunable model parameters."""
        import optuna

        return {
            "iterations": optuna.distributions.IntDistribution(50, 500),
            "learning_rate": optuna.distributions.FloatDistribution(
                0.01, 0.3, log=True
            ),
            "depth": optuna.distributions.IntDistribution(3, 10),
            "l2_leaf_reg": optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
            "subsample": optuna.distributions.FloatDistribution(0.5, 1.0),
            "colsample_bylevel": optuna.distributions.FloatDistribution(0.5, 1.0),
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
        from catboost import CatBoostRegressor, Pool

        transform_result = fit_transform_chain(
            data,
            feature_names=feature_names,
            encoder=encoder,
            preprocessing=preprocessing,
        )
        data = transform_result.data
        objective = resolve_objective(self.objective, data)

        p = dict(params or {})
        if "cat_features" in p:
            raise ValueError(
                "Set cat_features with CatBoostModel(categorical_features=...) "
                "instead of params"
            )
        p.setdefault("loss_function", _CB_OBJECTIVE[objective])
        p.setdefault("verbose", 0)
        p.setdefault("allow_writing_files", False)

        if isinstance(self.categorical_features, str):
            if self.categorical_features != "auto":
                raise ValueError(
                    "categorical_features must be 'auto' or a list of feature names"
                )
            categorical_names = [
                name
                for name in (data.schema.categorical if data.schema is not None else [])
                if name in data.feature_names
            ]
        else:
            categorical_names = list(self.categorical_features)
            missing = [
                name for name in categorical_names if name not in data.feature_names
            ]
            if missing:
                raise ValueError(
                    f"Categorical features missing after preprocessing: {missing}"
                )
        if len(set(categorical_names)) != len(categorical_names):
            raise ValueError("categorical_features must be unique")

        X = _pool_matrix(data.features, data.feature_names, categorical_names)
        y = series_to_fit_array(data.target)

        baseline_parts: list[np.ndarray] = []
        if objective == "poisson" and data.exposure is not None:
            baseline_parts.append(np.log(series_to_fit_array(data.exposure)))
        if data.offset is not None:
            baseline_parts.append(series_to_fit_array(data.offset))
        baseline = np.sum(baseline_parts, axis=0) if baseline_parts else None
        if baseline is not None and not _catboost_supports_offset():
            raise ValueError(
                "This CatBoost version does not support exposure or offsets"
            )

        sample_weight: np.ndarray | None = None
        if data.weight is not None:
            sample_weight = series_to_fit_array(data.weight)

        pool_kwargs = {
            "data": X,
            "label": y,
            "weight": sample_weight,
            "feature_names": list(data.feature_names),
        }
        if categorical_names:
            pool_kwargs["cat_features"] = categorical_names
        if baseline is not None:
            pool_kwargs["baseline"] = baseline
        pool = Pool(**pool_kwargs)

        model = CatBoostRegressor(**p)
        model.fit(pool)

        # LossFunctionChange normally requires the training Pool. Cache its
        # compact result now so the fitted wrapper does not retain that matrix.
        loss_function_importance = model.get_feature_importance(
            data=pool,
            type="LossFunctionChange",
        )

        feature_names = list(data.feature_names)
        has_offset = _catboost_supports_offset()

        def _predict(pred_data: ModelData, prediction_type: str) -> pl.Series:
            """Predict from the fitted estimator on the requested scale.

            Args:
                pred_data (ModelData): Prepared model data to score.
                prediction_type (str): Prediction scale: "response", "rate", or "link"; "rate" is
                    unavailable for Gamma.
            """
            X_pred = _pool_matrix(pred_data.features, feature_names, categorical_names)

            baseline_parts: list[np.ndarray] = []
            if objective == "poisson" and pred_data.exposure is not None:
                baseline_parts.append(np.log(series_to_fit_array(pred_data.exposure)))
            if pred_data.offset is not None:
                baseline_parts.append(series_to_fit_array(pred_data.offset))
            pred_baseline = np.sum(baseline_parts, axis=0) if baseline_parts else None
            if pred_baseline is not None and not has_offset:
                raise ValueError(
                    "This CatBoost version does not support exposure or offsets"
                )

            pred_pool_kwargs = {
                "data": X_pred,
                "feature_names": feature_names,
            }
            if categorical_names:
                pred_pool_kwargs["cat_features"] = categorical_names
            if pred_baseline is not None:
                pred_pool_kwargs["baseline"] = pred_baseline
            pred_pool = Pool(**pred_pool_kwargs)
            link = model.predict(pred_pool, prediction_type="RawFormulaVal")
            response = np.exp(link)

            if objective == "poisson":
                if prediction_type == "response":
                    return pl.Series(response)
                elif prediction_type == "rate":
                    if pred_data.exposure is not None:
                        return pl.Series(response / pred_data.exposure.to_numpy())
                    return pl.Series(response)
                else:
                    return pl.Series(link)
            return pl.Series(link if prediction_type == "link" else response)

        def _importance(importance_type: str | None = None) -> pl.DataFrame:
            # These types produce one scalar per input feature.  Interaction
            # and SHAP outputs are intentionally excluded because they are not
            # rankable 1:1 here.
            """Return feature importance from the fitted estimator.

            Args:
                importance_type (Optional[str]): Optional framework-specific importance measure.
            """
            importance_type = importance_type or "PredictionValuesChange"
            allowed = {
                "FeatureImportance",
                "PredictionValuesChange",
                "LossFunctionChange",
            }
            if importance_type not in allowed:
                raise ValueError(
                    "CatBoost importance_type must be one of: "
                    "'FeatureImportance', 'PredictionValuesChange', "
                    "'LossFunctionChange'"
                )
            scores = (
                loss_function_importance
                if importance_type == "LossFunctionChange"
                else model.get_feature_importance(type=importance_type)
            )
            return pl.DataFrame(
                {"feature": feature_names, "importance": scores.astype(float).tolist()}
            )

        return FittedModel(
            model=model,
            params=p,
            framework="catboost",
            objective=objective,
            feature_names=feature_names,
            predict_fn=_predict,
            importance_fn=_importance,
            transform_chain=transform_result.chain,
        )
