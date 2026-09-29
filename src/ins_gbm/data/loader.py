import polars as pl

from .model_data import ModelData, Objective
from .schema import FeatureSchema, infer_schema


def load_model_data(
    path: str,
    target: str,
    exposure: str | None = None,
    weight: str | None = None,
    feature_cols: list[str] | None = None,
    schema: FeatureSchema | None = None,
    objective: Objective | None = None,
    cv_fold: str | None = None,
    comparison_cols: list[str] | None = None,
    # Note: `offset` is intentionally absent here. It is not a load-time parameter
    # because it is computed after loading (e.g. from a base model's predictions)
    # and set via ModelData.with_offset().
) -> ModelData:
    """Read a data file and assemble a validated model data object.

    Args:
        path (str): Path to the input data file.
        target (str): Observed outcome series aligned with feature rows.
        exposure (Optional[str]): Positive exposure series aligned with rows, when used.
            Optional.
        weight (Optional[str]): Nonnegative observation weight series aligned with rows.
            Optional.
        feature_cols (Optional[list[str]]): Names of feature columns to load. Optional.
        schema (Optional[FeatureSchema]): Optional feature schema; inferred when omitted.
        objective (Optional[Objective]): Model objective: "poisson" or "gamma". Optional.
        cv_fold (Optional[str]): Optional integer fold assignment for each row.
        comparison_cols (Optional[list[str]]): Column names to load as benchmark predictions.
            Optional.
    """
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
