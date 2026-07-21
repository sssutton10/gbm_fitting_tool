"""Column-targeted preprocessing wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import polars as pl


def validate_preprocessing_steps(preprocessors: list[Any]) -> None:
    """Reject duplicate names among targeted preprocessing steps."""
    names = [step.name for step in preprocessors if isinstance(step, PreprocessingStep)]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"PreprocessingStep names must be unique: {duplicates}")


@dataclass
class PreprocessingStep:
    """Apply a preprocessor to selected columns and pass all others through.

    The wrapped preprocessor is fit only on ``feature_names``.  Its outputs are
    prefixed with ``name`` before replacing those input columns in the current
    feature frame.
    """

    name: str
    preprocessor: Any
    feature_names: list[str]

    def fit(
        self,
        features: pl.DataFrame,
        target: Optional[pl.Series] = None,
    ) -> "FittedPreprocessingStep":
        if not self.name:
            raise ValueError("PreprocessingStep.name must be non-empty")
        input_names = list(self.feature_names)
        if not input_names:
            raise ValueError("PreprocessingStep.feature_names must be non-empty")
        if len(set(input_names)) != len(input_names):
            raise ValueError("PreprocessingStep.feature_names must be unique")
        missing = [name for name in input_names if name not in features.columns]
        if missing:
            raise ValueError(
                f"PreprocessingStep {self.name!r} references missing features: {missing}"
            )

        fitted = self.preprocessor.fit(features.select(input_names), target)
        transformed = fitted.transform(features.select(input_names))
        output_names = [f"{self.name}__{column}" for column in transformed.columns]
        if len(set(output_names)) != len(output_names):
            raise ValueError(
                f"PreprocessingStep {self.name!r} produced duplicate output names"
            )
        collision = [
            name
            for name in output_names
            if name in features.columns and name not in input_names
        ]
        if collision:
            raise ValueError(
                f"PreprocessingStep {self.name!r} output names collide with existing features: {collision}"
            )
        return FittedPreprocessingStep(
            name=self.name,
            fitted_preprocessor=fitted,
            input_names=input_names,
            output_names=output_names,
        )


@dataclass
class FittedPreprocessingStep:
    """Fitted counterpart of :class:`PreprocessingStep`."""

    name: str
    fitted_preprocessor: Any
    input_names: list[str]
    output_names: list[str]

    def transform(self, features: pl.DataFrame) -> pl.DataFrame:
        missing = [name for name in self.input_names if name not in features.columns]
        if missing:
            raise ValueError(
                f"PreprocessingStep {self.name!r} requires missing features: {missing}"
            )
        transformed = self.fitted_preprocessor.transform(
            features.select(self.input_names)
        )
        expected_output_names = getattr(
            self.fitted_preprocessor,
            "output_feature_names",
            lambda: list(transformed.columns),
        )()
        if transformed.columns != expected_output_names:
            # The wrapped preprocessor must keep its fitted output schema stable.
            raise ValueError(
                f"PreprocessingStep {self.name!r} produced unexpected output columns"
            )
        transformed = transformed.rename(dict(zip(transformed.columns, self.output_names)))

        selected = set(self.input_names)
        first_input = next(column for column in features.columns if column in selected)
        columns: list[pl.Series] = []
        for column in features.columns:
            if column == first_input:
                columns.extend(transformed.get_columns())
            if column not in selected:
                columns.append(features[column])
        return pl.DataFrame(columns)

    def output_feature_names(self) -> list[str]:
        return list(self.output_names)

    def component_mapping(self) -> dict[str, list[str]]:
        return {name: list(self.input_names) for name in self.output_names}
