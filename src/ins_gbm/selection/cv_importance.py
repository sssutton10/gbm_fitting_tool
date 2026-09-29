"""Cross-validated feature importance for a shallow selection screen."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from ins_gbm.data.folds import CVConfig, resolve_folds
from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.models.catboost import CatBoostModel
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.models.random_forest import RandomForestModel
from ins_gbm.models.xgboost import XGBoostModel


def _shallow_params(model: Any) -> dict[str, Any]:
    """Build shallow model parameters for CV importance fitting.

    Args:
        model (Any): Model wrapper or fitted model to use.
    """
    if isinstance(model, XGBoostModel):
        return {"n_estimators": 50, "max_depth": 2}
    if isinstance(model, LightGBMModel):
        return {"n_estimators": 50, "max_depth": 2, "num_leaves": 4}
    if isinstance(model, CatBoostModel):
        return {"iterations": 50, "depth": 2}
    if isinstance(model, RandomForestModel):
        return {"n_estimators": 50, "max_depth": 2}
    return {}


def cv_feature_importance(
    data: ModelData,
    *,
    model: Any = None,
    cv: CVConfig | None = None,
    feature_names: Sequence[str] | None = None,
    encoder: Any = None,
    importance_types: Sequence[str] | str = ("weight", "gain", "cover"),
    params: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Summarize feature importance from one shallow fit per CV training split.

    The default model is XGBoost, whose ``weight``, ``gain``, and ``cover``
    measures correspond to split count, average gain, and average cover.
    ``cv`` uses stored ``ModelData.cv_fold`` IDs when present; pass
    ``CVConfig(folds="random", n_splits=..., seed=...)`` for random splits.

    ``n_folds_selected`` counts folds where *any requested measure* is > 0.
    ``mean_<type>`` averages over every fold, including zero for a feature
    absent from a fold's importance output. Without ``encoder``, rows retain
    the input feature order. With an encoder, it is fitted separately on each
    fold's training rows and rows report the union of encoded columns.
    For other model wrappers, supply importance types supported by that model.
    Explicit ``params`` override the shallow defaults for built-in wrappers.

    Args:
        data (ModelData): Model data to fit, transform, predict, or evaluate.
        model (Any): Model wrapper or fitted model to use. Optional.
        cv (CVConfig | None): Cross-validation configuration or explicit fold assignments.
            Optional.
        feature_names (Sequence[str] | None): Ordered names of input features to use. Optional.
        encoder (Any): Optional encoder applied before model fitting.
        importance_types (Sequence[str] | str): Importance measures to compute.
        params (dict[str, Any] | None): Optional model or estimator parameter mapping.
    """
    data.validate()
    selected = (
        data.select_features(list(feature_names)) if feature_names is not None else data
    )
    if not selected.feature_names:
        raise ValueError("feature_names must contain at least one feature")

    if isinstance(importance_types, str):
        types = [importance_types]
    else:
        types = list(importance_types)
    if not types or any(
        not isinstance(t, str) or not t or not t.isidentifier() for t in types
    ):
        raise ValueError("importance_types must contain valid, non-empty names")
    if len(types) != len(set(types)):
        raise ValueError("importance_types must be unique")

    model = model if model is not None else XGBoostModel()
    if not model.capabilities().supports_feature_importance:
        raise ValueError("model does not support feature importance")
    fit_params = {**_shallow_params(model), **(params or {})}
    _, splits = resolve_folds(selected, cv or CVConfig())
    names: list[str] = []
    name_set: set[str] = set()
    scores: dict[str, list[dict[str, float]]] = {kind: [] for kind in types}

    for train_indices, _ in splits:
        train_data = slice_model_data(selected, train_indices)
        if encoder is not None:
            fitted_encoder = encoder.fit(train_data.features, train_data.schema)
            train_data = train_data.with_features(
                fitted_encoder.transform(train_data.features)
            )
        fold_names = list(train_data.feature_names)
        if len(fold_names) != len(set(fold_names)):
            raise ValueError("encoded feature names must be unique")
        for name in fold_names:
            if name not in name_set:
                name_set.add(name)
                names.append(name)
        fitted = model.fit(train_data, params=dict(fit_params))
        for kind in types:
            importance = fitted.feature_importance(kind)
            if not {"feature", "importance"}.issubset(importance.columns):
                raise ValueError(
                    "feature importance must contain 'feature' and 'importance' columns"
                )
            reported = importance["feature"].to_list()
            if len(reported) != len(set(reported)) or not set(reported).issubset(
                set(fold_names)
            ):
                raise ValueError(
                    "feature importance contains duplicate or unknown features"
                )
            try:
                by_name = {
                    name: float(score)
                    for name, score in zip(reported, importance["importance"].to_list())
                }
            except (TypeError, ValueError) as exc:
                raise ValueError("feature importance scores must be numeric") from exc
            if not np.isfinite(list(by_name.values())).all():
                raise ValueError("feature importance scores must be finite")
            scores[kind].append(by_name)

    arrays = {
        kind: np.array(
            [[fold.get(name, 0.0) for name in names] for fold in scores[kind]],
            dtype=float,
        )
        for kind in types
    }
    selected_in_fold = np.logical_or.reduce([arrays[kind] > 0 for kind in types])
    columns: dict[str, Any] = {
        "feature": names,
        "n_folds_selected": selected_in_fold.sum(axis=0).tolist(),
    }
    for kind in types:
        columns[f"mean_{kind}"] = arrays[kind].mean(axis=0).tolist()
    return pl.DataFrame(columns)
