from dataclasses import dataclass, replace
from typing import Literal, Optional

import polars as pl

from .schema import FeatureSchema


Objective = Literal["poisson", "gamma"]

_INTEGER_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
}

_NUMERIC_DTYPES = _INTEGER_DTYPES | {pl.Float32, pl.Float64}


@dataclass
class ModelData:
    features: pl.DataFrame
    target: pl.Series
    exposure: Optional[pl.Series]
    weight: Optional[pl.Series]
    feature_names: list[str]
    schema: Optional[FeatureSchema] = None
    objective: Optional[Objective] = None
    offset: Optional[pl.Series] = None
    cv_fold: Optional[pl.Series] = None
    comparisons: Optional[pl.DataFrame] = None

    @property
    def n_rows(self) -> int:
        return self.features.height

    def validate(self) -> "ModelData":
        n = self.n_rows
        if self.target.len() != n:
            raise ValueError(
                f"target row count {self.target.len()} != features row count {n}"
            )
        if self.exposure is not None and self.exposure.len() != n:
            raise ValueError(
                f"exposure row count {self.exposure.len()} != features row count {n}"
            )
        if self.weight is not None and self.weight.len() != n:
            raise ValueError(
                f"weight row count {self.weight.len()} != features row count {n}"
            )
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        missing = [f for f in self.feature_names if f not in self.features.columns]
        if missing:
            raise ValueError(f"features DataFrame missing columns: {missing}")
        if self.exposure is not None:
            if self.exposure.null_count() > 0:
                raise ValueError("exposure must be non-null")
            if (self.exposure <= 0).any():
                raise ValueError("exposure must be positive and non-zero")
        if self.objective == "poisson":
            if self.exposure is None:
                raise ValueError("exposure is required for Poisson objective")
            if (self.target < 0).any():
                raise ValueError("Poisson target must be non-negative")
        if self.objective == "gamma":
            if (self.target <= 0).any():
                raise ValueError("Gamma target must be strictly positive")

        # --- offset validation ---
        if self.offset is not None:
            if self.offset.len() != n:
                raise ValueError(
                    f"offset row count {self.offset.len()} != features row count {n}"
                )
            if self.offset.dtype not in _NUMERIC_DTYPES:
                raise ValueError(
                    f"offset must have a numeric dtype, got {self.offset.dtype!r}"
                )
            if self.offset.null_count() > 0:
                raise ValueError("offset must be non-null (no missing values)")
            if self.offset.is_infinite().any():
                raise ValueError("offset must be finite (no inf values)")

        # --- cv_fold validation ---
        if self.cv_fold is not None:
            if self.cv_fold.len() != n:
                raise ValueError(
                    f"cv_fold row count {self.cv_fold.len()} != features row count {n}"
                )
            if self.cv_fold.dtype not in _INTEGER_DTYPES:
                raise ValueError(
                    f"cv_fold must have an integer dtype, got {self.cv_fold.dtype!r}"
                )
            if self.cv_fold.null_count() > 0:
                raise ValueError("cv_fold must be non-null (no missing values)")
            if self.cv_fold.n_unique() < 2:
                raise ValueError(
                    f"cv_fold must have at least 2 unique values, got {self.cv_fold.n_unique()}"
                )

        # --- comparisons validation ---
        if self.comparisons is not None:
            if self.comparisons.shape[0] != n:
                raise ValueError(
                    f"comparisons row count {self.comparisons.shape[0]} != features row count {n}"
                )
            for col in self.comparisons.columns:
                col_series = self.comparisons[col]
                if col_series.dtype not in _NUMERIC_DTYPES:
                    raise ValueError(
                        f"comparisons column '{col}' must be numeric, got {col_series.dtype!r}"
                    )
                if (col_series <= 0).any():
                    raise ValueError(
                        f"comparisons column '{col}' must be strictly positive (> 0)"
                    )

        return self

    def with_features(self, features: pl.DataFrame) -> "ModelData":
        """Return a copy with replaced features and updated feature_names."""
        return replace(self, features=features, feature_names=list(features.columns))

    def select_features(self, feature_names: list[str]) -> "ModelData":
        """Return a copy restricted to an ordered subset of feature columns.

        Row-level fields are deliberately retained so one loaded ``ModelData``
        can be reused for several fits with different predictor sets.
        """
        selected = list(feature_names)
        if not selected:
            raise ValueError("feature_names must contain at least one feature")
        if len(set(selected)) != len(selected):
            raise ValueError("feature_names must be unique")
        missing = [name for name in selected if name not in self.features.columns]
        if missing:
            raise ValueError(f"features DataFrame missing columns: {missing}")

        schema = self.schema
        if schema is not None:
            selected_set = set(selected)
            schema = FeatureSchema(
                numeric=[name for name in schema.numeric if name in selected_set],
                categorical=[name for name in schema.categorical if name in selected_set],
                ordinal=[name for name in schema.ordinal if name in selected_set],
                passthrough=[name for name in schema.passthrough if name in selected_set],
            )
        return replace(
            self,
            features=self.features.select(selected),
            feature_names=selected,
            schema=schema,
        )

    def with_offset(self, offset: pl.Series) -> "ModelData":
        """Return a copy with the given offset Series set."""
        return replace(self, offset=offset)


def slice_model_data(data: "ModelData", indices) -> "ModelData":
    """Return a new ModelData containing only the rows at *indices*."""
    return ModelData(
        features=data.features[indices],
        target=data.target[indices],
        exposure=data.exposure[indices] if data.exposure is not None else None,
        weight=data.weight[indices] if data.weight is not None else None,
        feature_names=data.feature_names,
        schema=data.schema,
        objective=data.objective,
        offset=data.offset[indices] if data.offset is not None else None,
        cv_fold=data.cv_fold[indices] if data.cv_fold is not None else None,
        comparisons=data.comparisons[indices] if data.comparisons is not None else None,
    )
