"""Shared fitting and application of model feature transforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ins_gbm.data.model_data import ModelData
from ins_gbm.preprocessing.steps import validate_preprocessing_steps


def select_model_features(data: ModelData, feature_names: list[str]) -> ModelData:
    """Apply a fixed selection to the post-preprocessing model matrix.

    Args:
        data (ModelData): Model data to fit, transform, predict, or evaluate.
        feature_names (list[str]): Ordered names of input features to use.
    """
    missing = [name for name in feature_names if name not in data.features.columns]
    if missing:
        raise ValueError(f"Model features missing after preprocessing: {missing}")
    return data.with_features(data.features.select(feature_names))


@dataclass
class FittedTransformChain:
    """Fitted, replayable transforms between raw data and a model matrix.

    Args:
        input_feature_names (list[str]): Names of features expected before fitted transforms.
        encoder (Optional[Any]): Optional encoder applied before model fitting.
        selected_features (Optional[list[str]]): Feature names retained by selection. Optional.
        preprocessors (list[Any]): Ordered preprocessing steps or their fitted counterparts.
        model_selected_features (Optional[list[str]]): Optional feature subset applied after
            preprocessing.
    """

    input_feature_names: list[str]
    encoder: Any | None = None
    selected_features: list[str] | None = None
    preprocessors: list[Any] = field(default_factory=list)
    model_selected_features: list[str] | None = None

    def transform(self, data: ModelData) -> ModelData:
        """Apply the fitted transformation to input features.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
        """
        current = data.select_features(self.input_feature_names)
        if self.encoder is not None:
            current = current.with_features(self.encoder.transform(current.features))
        if self.selected_features is not None:
            missing = [
                name
                for name in self.selected_features
                if name not in current.features.columns
            ]
            if missing:
                raise ValueError(f"Selected features missing after encoding: {missing}")
            current = current.with_features(
                current.features.select(self.selected_features)
            )
        for preprocessor in self.preprocessors:
            current = current.with_features(preprocessor.transform(current.features))
        if self.model_selected_features is not None:
            current = select_model_features(current, self.model_selected_features)
        return current


@dataclass
class TransformFitResult:
    """Fit-local transformed data plus the compact state needed to replay it.

    Args:
        data (ModelData): Model data to fit, transform, predict, or evaluate.
        raw_data (ModelData): Original training data before fitted transforms.
        chain (FittedTransformChain): Fitted sequence of feature transformations.
        fitted_selector (Optional[Any]): Selector fitted on training data, when configured. Optional.
    """

    data: ModelData
    raw_data: ModelData
    chain: FittedTransformChain
    fitted_selector: Any | None = None


def fit_transform_chain(
    data: ModelData,
    *,
    feature_names: list[str] | None = None,
    encoder: Any | None = None,
    selector: Any | None = None,
    preprocessing: list[Any] | None = None,
    schema: Any | None = None,
    model_selected_features: list[str] | None = None,
) -> TransformFitResult:
    """Fit an ordered transform chain without modifying or retaining its matrix.

    Args:
        data (ModelData): Model data to fit, transform, predict, or evaluate.
        feature_names (Optional[list[str]]): Ordered names of input features to use. Optional.
        encoder (Optional[Any]): Optional encoder applied before model fitting.
        selector (Optional[Any]): Optional feature selector fitted on training rows.
        preprocessing (Optional[list[Any]]): Ordered preprocessing steps applied before fitting.
            Optional.
        schema (Optional[Any]): Optional feature schema; inferred when omitted.
        model_selected_features (Optional[list[str]]): Optional feature subset applied after
            preprocessing.
    """

    data.validate(require_multiple_folds=False)

    preprocessing_chain = list(preprocessing or [])
    validate_preprocessing_steps(preprocessing_chain)

    raw_data = (
        data.select_features(feature_names) if feature_names is not None else data
    )
    current = raw_data
    fitted_encoder: Any | None = None
    fitted_selector: Any | None = None

    if encoder is not None:
        encoder_schema = schema if schema is not None else current.schema
        if encoder_schema is None:
            raise ValueError(
                "An encoder requires ModelData.schema or an explicit schema"
            )
        fitted_encoder = encoder.fit(current.features, encoder_schema)
        current = current.with_features(fitted_encoder.transform(current.features))

    selected_features: list[str] | None = None
    if selector is not None:
        fitted_selector = selector.fit(current)
        selected_features = fitted_selector.selected_features()
        current = current.with_features(current.features.select(selected_features))

    fitted_preprocessors: list[Any] = []
    for preprocessor in preprocessing_chain:
        fitted = preprocessor.fit(current.features, current.target)
        current = current.with_features(fitted.transform(current.features))
        fitted_preprocessors.append(fitted)

    if model_selected_features is not None:
        current = select_model_features(current, model_selected_features)

    return TransformFitResult(
        data=current,
        raw_data=raw_data,
        chain=FittedTransformChain(
            input_feature_names=list(raw_data.feature_names),
            encoder=fitted_encoder,
            selected_features=selected_features,
            preprocessors=fitted_preprocessors,
            model_selected_features=model_selected_features,
        ),
        fitted_selector=fitted_selector,
    )
