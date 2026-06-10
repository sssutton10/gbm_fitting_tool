from typing import Optional

import polars as pl

from .model_data import ModelData, Objective
from .schema import FeatureSchema, infer_schema


def load_model_data(
    path: str,
    target: str,
    exposure: Optional[str] = None,
    weight: Optional[str] = None,
    feature_cols: Optional[list[str]] = None,
    schema: Optional[FeatureSchema] = None,
    objective: Optional[Objective] = None,
    cv_fold: Optional[str] = None,
    comparison_cols: Optional[list[str]] = None,
) -> ModelData:
    df = pl.read_parquet(path)

    if feature_cols is None:
        reserved = {target}
        if exposure is not None:
            reserved.add(exposure)
        if weight is not None:
            reserved.add(weight)
        if cv_fold is not None:
            reserved.add(cv_fold)
        if comparison_cols is not None:
            reserved.update(comparison_cols)
        feature_cols = [c for c in df.columns if c not in reserved]

    features = df.select(feature_cols)
    target_series = df[target]
    exposure_series = df[exposure] if exposure else None
    weight_series = df[weight] if weight else None
    cv_fold_series = df[cv_fold] if cv_fold else None
    comparisons_df = df.select(comparison_cols) if comparison_cols else None

    if schema is None:
        schema = infer_schema(df, feature_cols)

    data = ModelData(
        features=features,
        target=target_series,
        exposure=exposure_series,
        weight=weight_series,
        feature_names=list(feature_cols),
        schema=schema,
        objective=objective,
        cv_fold=cv_fold_series,
        comparisons=comparisons_df,
    )
    return data.validate()
