from dataclasses import dataclass, field

import polars as pl

from ins_gbm.data.schema import FeatureSchema


_NUMERIC_FILL: float = -999_999_999.0  # sentinel for missing numeric/ordinal values
_MISSING_LEVEL: str = "-999999999"     # sentinel for missing categorical values


@dataclass
class OneHotEncoder:
    """Fit-then-transform one-hot encoder for categorical columns.

    Missing value convention
    -----------------------
    The encoder expects missing values to be pre-filled by the caller before
    ``fit()`` is invoked:

    - Numeric/ordinal columns: ``_NUMERIC_FILL`` (``-999_999_999.0``).
      ``transform()`` passes these through unchanged.  Model wrappers that
      support native missing-value handling (LightGBM, CatBoost) convert this
      sentinel back to ``NaN``; XGBoost declares it via ``missing=_NUMERIC_FILL``
      on ``DMatrix``.
    - Categorical columns: ``_MISSING_LEVEL`` (``"-999999999"``).
      Treated as an explicit level during ``fit()`` so it gets its own indicator
      column, just like any observed category.
    """

    def fit(self, features: pl.DataFrame, schema: FeatureSchema) -> "FittedOneHotEncoder":
        levels: dict[str, list[str]] = {}
        for col in schema.categorical:
            unique = (
                features[col]
                .cast(pl.Utf8)
                .fill_null(_MISSING_LEVEL)
                .unique()
                .sort()
                .to_list()
            )
            levels[col] = unique

        numeric_cols = list(schema.numeric)
        ordinal_cols = list(schema.ordinal)
        passthrough_cols = list(schema.passthrough)

        # Build stable output column order
        output_names: list[str] = []
        output_names.extend(numeric_cols)
        output_names.extend(ordinal_cols)
        output_names.extend(passthrough_cols)
        for col, lvls in levels.items():
            for lvl in lvls:
                output_names.append(f"{col}__{lvl}")

        return FittedOneHotEncoder(
            levels=levels,
            numeric_cols=numeric_cols,
            ordinal_cols=ordinal_cols,
            passthrough_cols=passthrough_cols,
            _output_names=output_names,
        )


@dataclass
class FittedOneHotEncoder:
    levels: dict[str, list[str]]
    numeric_cols: list[str]
    ordinal_cols: list[str]
    passthrough_cols: list[str]
    _output_names: list[str]

    def output_feature_names(self) -> list[str]:
        return list(self._output_names)

    def transform(self, features: pl.DataFrame) -> pl.DataFrame:
        parts: list[pl.Series] = []

        for col in self.numeric_cols:
            parts.append(features[col].fill_null(_NUMERIC_FILL))
        for col in self.ordinal_cols:
            parts.append(features[col].fill_null(_NUMERIC_FILL))
        for col in self.passthrough_cols:
            parts.append(features[col])

        for col, lvls in self.levels.items():
            col_str = features[col].cast(pl.Utf8).fill_null(_MISSING_LEVEL)
            for lvl in lvls:
                indicator = (col_str == lvl).cast(pl.Float64).alias(f"{col}__{lvl}")
                parts.append(indicator)

        return pl.DataFrame(parts)[self._output_names]
