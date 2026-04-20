from dataclasses import dataclass, field
from typing import Iterable

import polars as pl


@dataclass
class FeatureSchema:
    numeric: list[str]
    categorical: list[str]
    ordinal: list[str] = field(default_factory=list)
    passthrough: list[str] = field(default_factory=list)

    def all_features(self) -> list[str]:
        return [*self.numeric, *self.categorical, *self.ordinal, *self.passthrough]


_NUMERIC_DTYPES = {
    pl.Float32, pl.Float64,
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
}
_CATEGORICAL_DTYPES = {pl.Utf8, pl.String, pl.Categorical, pl.Boolean}


def infer_schema(df: pl.DataFrame, feature_cols: Iterable[str]) -> FeatureSchema:
    numeric: list[str] = []
    categorical: list[str] = []
    for col in feature_cols:
        dtype = df.schema[col]
        base = type(dtype)
        if dtype in _NUMERIC_DTYPES or base in _NUMERIC_DTYPES:
            numeric.append(col)
        elif dtype in _CATEGORICAL_DTYPES or base in _CATEGORICAL_DTYPES or isinstance(dtype, (pl.Enum,)):
            categorical.append(col)
        else:
            raise ValueError(f"Unsupported dtype {dtype!r} for column '{col}'. "
                             f"Supply a FeatureSchema explicitly.")
    return FeatureSchema(numeric=numeric, categorical=categorical)
