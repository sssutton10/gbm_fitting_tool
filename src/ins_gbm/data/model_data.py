from dataclasses import dataclass, replace
from typing import Literal, Optional

import polars as pl

from .schema import FeatureSchema


Objective = Literal["poisson", "gamma"]


@dataclass
class ModelData:
    features: pl.DataFrame
    target: pl.Series
    exposure: Optional[pl.Series]
    weight: Optional[pl.Series]
    feature_names: list[str]
    schema: Optional[FeatureSchema] = None
    objective: Optional[Objective] = None

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
        return self

    def with_features(self, features: pl.DataFrame) -> "ModelData":
        """Return a copy with replaced features and updated feature_names."""
        return replace(self, features=features, feature_names=list(features.columns))
