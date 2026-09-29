from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import polars as pl

from .dtypes import cast_float64_frame, cast_float64_series
from .schema import FeatureSchema, infer_schema

Objective = Literal["poisson", "gamma"]

_INTEGER_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
}

_NUMERIC_DTYPES = _INTEGER_DTYPES | {pl.Float32, pl.Float64}


@dataclass
class ModelData:
    """Hold model features, targets, and optional fitting and evaluation data.

    Args:
        features (pl.DataFrame): Input feature frame; rows align with the target and optional
            series.
        target (pl.Series): Observed outcome series aligned with feature rows.
        feature_names (list[str]): Ordered names of input features to use.
        exposure (Optional[pl.Series]): Positive exposure per row for frequency models;
            optional.
        weight (Optional[pl.Series]): Nonnegative sample weight per row; optional.
        schema (Optional[FeatureSchema]): Optional feature schema; inferred when omitted.
        objective (Optional[Objective]): Insurance objective: "poisson" for counts or "gamma"
            for positive severity; optional for prediction data.
        offset (Optional[pl.Series]): Optional offset on the model link scale, aligned with
            feature rows.
        cv_fold (Optional[pl.Series]): Optional integer fold assignment for each row.
        comparisons (Optional[pl.DataFrame]): Optional positive benchmark predictions in named
            columns, aligned with rows.
    """

    features: pl.DataFrame
    target: pl.Series
    feature_names: list[str]
    exposure: pl.Series | None = None
    weight: pl.Series | None = None
    schema: FeatureSchema | None = None
    objective: Objective | None = None
    offset: pl.Series | None = None
    cv_fold: pl.Series | None = None
    comparisons: pl.DataFrame | None = None

    def __post_init__(self) -> None:
        """Apply the fitting dtype policy and infer a schema when needed."""
        self.features = cast_float64_frame(self.features)
        self.target = cast_float64_series(self.target)
        self.exposure = cast_float64_series(self.exposure)
        self.weight = cast_float64_series(self.weight)
        self.offset = cast_float64_series(self.offset)
        if self.comparisons is not None:
            self.comparisons = cast_float64_frame(self.comparisons)

        if self.schema is None and all(
            name in self.features.columns for name in self.feature_names
        ):
            self.schema = infer_schema(self.features, self.feature_names)

    @property
    def n_rows(self) -> int:
        """Return the number of feature rows."""
        return self.features.height

    def validate(self, *, require_multiple_folds: bool = True) -> "ModelData":
        """Validate training fields and optional evaluation inputs.

        Args:
            require_multiple_folds (bool): Whether supplied fold IDs must contain at least two
                values. Defaults to True.
        """
        self._validate_features_and_target()
        self._validate_exposure_and_weight()
        self._validate_objective()
        self._validate_offset()
        self._validate_cv_fold(require_multiple_folds)
        self._validate_comparisons()
        return self

    def _validate_features_and_target(self) -> None:
        """Validate training row counts, feature names, and target values."""
        n = self.n_rows
        if n == 0:
            raise ValueError("features must contain at least one row")
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
        self._validate_numeric_series("target", self.target, finite=True)

    def _validate_exposure_and_weight(self) -> None:
        """Validate exposure and observation weights when supplied."""
        if self.exposure is not None:
            self._validate_numeric_series("exposure", self.exposure, finite=True)
            if self.exposure.null_count() > 0:
                raise ValueError("exposure must be non-null")
            if (self.exposure <= 0).any():
                raise ValueError("exposure must be positive and non-zero")
        if self.weight is not None:
            self._validate_numeric_series("weight", self.weight, finite=True)
            if (self.weight < 0).any():
                raise ValueError("weight must be non-negative")
            if float(self.weight.sum()) <= 0:
                raise ValueError("weight must have a positive total")

    def _validate_objective(self) -> None:
        """Check target values against the selected objective."""
        if self.objective == "poisson" and (self.target < 0).any():
            raise ValueError("Poisson target must be non-negative")
        if self.objective == "gamma" and (self.target <= 0).any():
            raise ValueError("Gamma target must be strictly positive")

    def _validate_offset(self) -> None:
        """Validate an optional model offset."""
        if self.offset is None:
            return
        if self.offset.len() != self.n_rows:
            raise ValueError(
                f"offset row count {self.offset.len()} "
                f"!= features row count {self.n_rows}"
            )
        if self.offset.dtype not in _NUMERIC_DTYPES:
            raise ValueError(
                f"offset must have a numeric dtype, got {self.offset.dtype!r}"
            )
        if self.offset.null_count() > 0:
            raise ValueError("offset must be non-null (no missing values)")
        if self.offset.is_infinite().any():
            raise ValueError("offset must be finite (no inf values)")
        if self.offset.is_nan().any():
            raise ValueError("offset must be finite (no NaN values)")

    def _validate_cv_fold(self, require_multiple_folds: bool) -> None:
        """Validate optional cross-validation fold assignments.

        Args:
            require_multiple_folds (bool): Whether supplied fold IDs must contain at least two
                values.
        """
        if self.cv_fold is None:
            return
        if self.cv_fold.len() != self.n_rows:
            raise ValueError(
                f"cv_fold row count {self.cv_fold.len()} "
                f"!= features row count {self.n_rows}"
            )
        if self.cv_fold.dtype not in _INTEGER_DTYPES:
            raise ValueError(
                f"cv_fold must have an integer dtype, got {self.cv_fold.dtype!r}"
            )
        if self.cv_fold.null_count() > 0:
            raise ValueError("cv_fold must be non-null (no missing values)")
        if require_multiple_folds and self.cv_fold.n_unique() < 2:
            raise ValueError("cv_fold must have at least 2 unique values")

    def _validate_comparisons(self) -> None:
        """Validate optional comparison predictions."""
        if self.comparisons is None:
            return
        if self.comparisons.shape[0] != self.n_rows:
            raise ValueError(
                f"comparisons row count {self.comparisons.shape[0]} "
                f"!= features row count {self.n_rows}"
            )
        for col in self.comparisons.columns:
            series = self.comparisons[col]
            if series.dtype not in _NUMERIC_DTYPES:
                raise ValueError(
                    f"comparisons column '{col}' must be numeric, got {series.dtype!r}"
                )
            if (series <= 0).any():
                raise ValueError(
                    f"comparisons column '{col}' must be strictly positive (> 0)"
                )
            if (
                series.null_count()
                or series.is_nan().any()
                or series.is_infinite().any()
            ):
                raise ValueError(
                    f"comparisons column '{col}' must be finite and non-null"
                )

    def validate_for_prediction(self) -> "ModelData":
        """Validate fields used for scoring without requiring a meaningful target."""
        n = self.n_rows
        if n == 0:
            raise ValueError("features must contain at least one row")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        missing = [
            name for name in self.feature_names if name not in self.features.columns
        ]
        if missing:
            raise ValueError(f"features DataFrame missing columns: {missing}")
        for name, values in (
            ("exposure", self.exposure),
            ("weight", self.weight),
            ("offset", self.offset),
        ):
            if values is None:
                continue
            if values.len() != n:
                raise ValueError(
                    f"{name} row count {values.len()} != features row count {n}"
                )
            self._validate_numeric_series(name, values, finite=True)
        if self.exposure is not None and (self.exposure <= 0).any():
            raise ValueError("exposure must be positive and non-zero")
        if self.weight is not None and (self.weight < 0).any():
            raise ValueError("weight must be non-negative")
        return self

    @staticmethod
    def _validate_numeric_series(name: str, values: pl.Series, *, finite: bool) -> None:
        """Check a series has a supported numeric dtype and finite values.

        Args:
            name (str): Name of the requested feature, model, metric, or stage.
            values (pl.Series): Metric values keyed by report name.
            finite (bool): Whether all values must be finite.
        """
        if values.dtype not in _NUMERIC_DTYPES:
            raise ValueError(f"{name} must have a numeric dtype, got {values.dtype!r}")
        if values.null_count() > 0:
            raise ValueError(f"{name} must be non-null")
        if finite:
            array = values.to_numpy()
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")

    def with_features(self, features: pl.DataFrame) -> "ModelData":
        """Return a copy with replaced features and updated feature_names.

        Args:
            features (pl.DataFrame): Input feature frame; rows align with the target and optional
                series.
        """
        return replace(self, features=features, feature_names=list(features.columns))

    def select_features(self, feature_names: list[str]) -> "ModelData":
        """Return a copy restricted to an ordered subset of feature columns.

        Row-level fields are deliberately retained so one loaded ``ModelData``
        can be reused for several fits with different predictor sets.

        Args:
            feature_names (list[str]): Ordered names of input features to use.
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
                categorical=[
                    name for name in schema.categorical if name in selected_set
                ],
                ordinal=[name for name in schema.ordinal if name in selected_set],
                passthrough=[
                    name for name in schema.passthrough if name in selected_set
                ],
            )
        return replace(
            self,
            features=self.features.select(selected),
            feature_names=selected,
            schema=schema,
        )

    def with_offset(self, offset: pl.Series) -> "ModelData":
        """Return a copy with the given offset Series set.

        Args:
            offset (pl.Series): Optional offset on the model link scale, aligned with feature rows.
        """
        return replace(self, offset=offset)


def slice_model_data(data: "ModelData", indices) -> "ModelData":
    """Return a new ModelData containing only the rows at *indices*.

    Args:
        data ('ModelData'): Model data to fit, transform, predict, or evaluate.
        indices (object): Row indices to retain.
    """
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
