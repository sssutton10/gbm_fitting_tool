"""Cross-validated feature importance for a shallow selection screen."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ins_gbm.data.folds import CVConfig, resolve_folds
from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.models.catboost import CatBoostModel
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.models.random_forest import RandomForestModel
from ins_gbm.models.xgboost import XGBoostModel


def _shallow_params(model: Any) -> dict[str, Any]:
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
    absent from a fold's importance output. Rows retain the input feature order.
    Features must be directly fit-ready; encode them before calling if needed.
    For other model wrappers, supply importance types supported by that model.
    Explicit ``params`` override the shallow defaults for built-in wrappers.
    """
    data.validate()
    selected = data.select_features(list(feature_names)) if feature_names is not None else data
    names = list(selected.feature_names)
    if not names:
        raise ValueError("feature_names must contain at least one feature")

    if isinstance(importance_types, str):
        types = [importance_types]
    else:
        types = list(importance_types)
    if not types or any(not isinstance(t, str) or not t or not t.isidentifier() for t in types):
        raise ValueError("importance_types must contain valid, non-empty names")
    if len(types) != len(set(types)):
        raise ValueError("importance_types must be unique")

    model = model if model is not None else XGBoostModel()
    if not model.capabilities().supports_feature_importance:
        raise ValueError("model does not support feature importance")
    fit_params = {**_shallow_params(model), **(params or {})}
    _, splits = resolve_folds(selected, cv or CVConfig())
    scores = {kind: np.zeros((len(splits), len(names)), dtype=float) for kind in types}
    name_set = set(names)

    for fold_index, (train_indices, _) in enumerate(splits):
        fitted = model.fit(slice_model_data(selected, train_indices), params=dict(fit_params))
        for kind in types:
            importance = fitted.feature_importance(kind)
            if not {"feature", "importance"}.issubset(importance.columns):
                raise ValueError("feature importance must contain 'feature' and 'importance' columns")
            reported = importance["feature"].to_list()
            if len(reported) != len(set(reported)) or not set(reported).issubset(name_set):
                raise ValueError("feature importance contains duplicate or unknown features")
            try:
                by_name = dict(zip(reported, importance["importance"].to_list()))
                fold_scores = np.array([float(by_name.get(name, 0.0)) for name in names])
            except (TypeError, ValueError) as exc:
                raise ValueError("feature importance scores must be numeric") from exc
            if not np.isfinite(fold_scores).all():
                raise ValueError("feature importance scores must be finite")
            scores[kind][fold_index] = fold_scores

    selected_in_fold = np.logical_or.reduce([scores[kind] > 0 for kind in types])
    columns: dict[str, Any] = {
        "feature": names,
        "n_folds_selected": selected_in_fold.sum(axis=0).tolist(),
    }
    for kind in types:
        columns[f"mean_{kind}"] = scores[kind].mean(axis=0).tolist()
    return pl.DataFrame(columns)
