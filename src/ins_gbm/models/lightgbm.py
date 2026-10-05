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

_LGB_OBJECTIVE = {
    "poisson": "poisson",
    "gamma": "gamma",
}
_CATEGORICAL_PARAM_ALIASES = {
    "categorical_feature",
    "categorical_column",
    "cat_feature",
    "cat_column",
}


def _category_strings(features: pl.DataFrame, name: str) -> pl.Expr:
    """Normalize a categorical column before fitting or applying its code map."""
    values = pl.col(name)
    if features.schema[name].is_float():
        values = values.fill_nan(None)
    return values.cast(pl.Utf8).replace(_MISSING_LEVEL, None)


def _categorical_matrix(
    features: pl.DataFrame,
    feature_names: list[str],
    category_codes: dict[str, dict[str, int]],
) -> np.ndarray:
    """Encode fitted categorical levels while keeping other features numeric."""
    if category_codes:
        expressions = [
            _category_strings(features, name)
            .replace_strict(codes, default=None)
            .cast(pl.Int32)
            .alias(name)
            for name, codes in category_codes.items()
        ]
        features = features.with_columns(expressions)
    return replace_value_with_nan(
        frame_to_fit_array(features, feature_names), _NUMERIC_FILL
    )


@dataclass
class LightGBMModel:
    """LightGBM wrapper for Poisson (frequency) and Gamma (severity) objectives.

    Missing values
    --------------
    With no encoder, categorical features are handled natively and other
    features must be numeric. When an encoder is supplied to :meth:`fit`, raw
    features are encoded at fit time. Encoded numeric values use
    ``_NUMERIC_FILL`` (``-999_999_999.0``).
    Before constructing the ``Dataset``, the wrapper converts that sentinel back
    to ``NaN`` so LightGBM can apply its native missing-value branch logic
    (learns the optimal direction at each split).

    Args:
        objective (Optional[Objective]): Model objective: "poisson" or "gamma". Optional.
        categorical_features (list[str] | "auto"): Categorical feature names, or
            "auto" to use the surviving names in ``ModelData.schema.categorical``.
            Categories are learned from training rows and missing or unseen levels
            use LightGBM's native missing-value handling.
    """

    objective: Objective | None = None
    categorical_features: list[str] | Literal["auto"] = "auto"

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
        categorical_params = sorted(_CATEGORICAL_PARAM_ALIASES.intersection(p))
        if categorical_params:
            raise ValueError(
                f"Set {categorical_params[0]!r} with "
                "LightGBMModel(categorical_features=...) instead of params"
            )
        p.setdefault("objective", _LGB_OBJECTIVE[objective])
        p.setdefault("verbose", -1)

        if (
            isinstance(self.categorical_features, str)
            and self.categorical_features != "auto"
        ):
            raise ValueError(
                "categorical_features must be 'auto' or a list of feature names"
            )
        if self.categorical_features == "auto":
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

        category_codes: dict[str, dict[str, int]] = {}
        for name in categorical_names:
            levels = (
                data.features.select(_category_strings(data.features, name).alias(name))
                .to_series()
                .drop_nulls()
                .unique()
                .sort()
                .to_list()
            )
            category_codes[name] = {
                level: index for index, level in enumerate(levels)
            }

        X = _categorical_matrix(data.features, data.feature_names, category_codes)
        y = series_to_fit_array(data.target)

        init_score_parts: list[np.ndarray] = []
        if objective == "poisson" and data.exposure is not None:
            init_score_parts.append(np.log(series_to_fit_array(data.exposure)))
        if data.offset is not None:
            init_score_parts.append(series_to_fit_array(data.offset))
        init_score: np.ndarray | None = (
            np.sum(init_score_parts, axis=0) if init_score_parts else None
        )

        sample_weight: np.ndarray | None = None
        if data.weight is not None:
            sample_weight = series_to_fit_array(data.weight)

        n_estimators = p.pop("n_estimators", 100)

        dataset_kwargs = {
            "label": y,
            "weight": sample_weight,
            "feature_name": list(data.feature_names),
            "free_raw_data": True,
        }
        if categorical_names:
            dataset_kwargs["categorical_feature"] = categorical_names
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
            """Predict from the fitted estimator on the requested scale.

            Args:
                pred_data (ModelData): Prepared model data to score.
                prediction_type (str): Prediction scale: "response", "rate", or "link"; "rate" is
                    unavailable for Gamma.
            """
            X_pred = _categorical_matrix(
                pred_data.features, feature_names, category_codes
            )
            # LightGBM's default prediction is already on the response scale.
            # Request the tree contribution on the link scale so exposure and
            # user offsets can be applied exactly once.
            tree_link = booster.predict(X_pred, raw_score=True)

            offset = (
                series_to_fit_array(pred_data.offset)
                if pred_data.offset is not None
                else None
            )

            if objective == "poisson":
                link = tree_link.copy()
                if pred_data.exposure is not None:
                    link = link + np.log(series_to_fit_array(pred_data.exposure))
                if offset is not None:
                    link = link + offset
                response = np.exp(link)
                if prediction_type == "response":
                    return pl.Series(response)
                elif prediction_type == "rate":
                    if pred_data.exposure is None:
                        return pl.Series(response)
                    return pl.Series(response / series_to_fit_array(pred_data.exposure))
                else:  # link
                    return pl.Series(link)
            else:
                link = tree_link if offset is None else tree_link + offset
                return pl.Series(link if prediction_type == "link" else np.exp(link))

        def _importance(importance_type: str | None = None) -> pl.DataFrame:
            """Return feature importance from the fitted estimator.

            Args:
                importance_type (Optional[str]): Optional framework-specific importance measure.
            """
            importance_type = importance_type or "gain"
            if importance_type not in {"gain", "split"}:
                raise ValueError(
                    "LightGBM importance_type must be one of: 'gain', 'split'"
                )
            names = booster.feature_name()
            scores = booster.feature_importance(importance_type=importance_type).astype(
                float
            )
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
