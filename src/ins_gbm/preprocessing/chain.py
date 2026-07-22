"""Shared fitting and application of model feature transforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ins_gbm.data.model_data import ModelData
from ins_gbm.preprocessing.steps import validate_preprocessing_steps


@dataclass
class FittedTransformChain:
    """Fitted, replayable transforms between raw data and a model matrix."""

    input_feature_names: list[str]
    encoder: Optional[Any] = None
    selected_features: Optional[list[str]] = None
    preprocessors: list[Any] = field(default_factory=list)

    def transform(self, data: ModelData) -> ModelData:
        current = data.select_features(self.input_feature_names)
        if self.encoder is not None:
            current = current.with_features(
                self.encoder.transform(current.features)
            )
        if self.selected_features is not None:
            missing = [
                name
                for name in self.selected_features
                if name not in current.features.columns
            ]
            if missing:
                raise ValueError(
                    f"Selected features missing after encoding: {missing}"
                )
            current = current.with_features(
                current.features.select(self.selected_features)
            )
        for preprocessor in self.preprocessors:
            current = current.with_features(
                preprocessor.transform(current.features)
            )
        return current


@dataclass
class TransformFitResult:
    """Fit-local transformed data plus the compact state needed to replay it."""

    data: ModelData
    raw_data: ModelData
    chain: FittedTransformChain
    fitted_selector: Optional[Any] = None


def fit_transform_chain(
    data: ModelData,
    *,
    feature_names: Optional[list[str]] = None,
    encoder: Optional[Any] = None,
    selector: Optional[Any] = None,
    preprocessing: Optional[list[Any]] = None,
    schema: Optional[Any] = None,
) -> TransformFitResult:
    """Fit an ordered transform chain without modifying or retaining its matrix."""

    preprocessing_chain = list(preprocessing or [])
    validate_preprocessing_steps(preprocessing_chain)

    raw_data = (
        data.select_features(feature_names)
        if feature_names is not None
        else data
    )
    current = raw_data
    fitted_encoder: Optional[Any] = None
    fitted_selector: Optional[Any] = None

    if encoder is not None:
        encoder_schema = schema if schema is not None else current.schema
        if encoder_schema is None:
            raise ValueError(
                "An encoder requires ModelData.schema or an explicit schema"
            )
        fitted_encoder = encoder.fit(current.features, encoder_schema)
        current = current.with_features(
            fitted_encoder.transform(current.features)
        )

    selected_features: Optional[list[str]] = None
    if selector is not None:
        fitted_selector = selector.fit(current)
        selected_features = fitted_selector.selected_features()
        current = current.with_features(
            current.features.select(selected_features)
        )

    fitted_preprocessors: list[Any] = []
    for preprocessor in preprocessing_chain:
        fitted = preprocessor.fit(current.features, current.target)
        current = current.with_features(fitted.transform(current.features))
        fitted_preprocessors.append(fitted)

    return TransformFitResult(
        data=current,
        raw_data=raw_data,
        chain=FittedTransformChain(
            input_feature_names=list(raw_data.feature_names),
            encoder=fitted_encoder,
            selected_features=selected_features,
            preprocessors=fitted_preprocessors,
        ),
        fitted_selector=fitted_selector,
    )
