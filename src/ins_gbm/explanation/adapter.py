"""Translate shared diagnostic coordinates to each fitted model's design."""

from dataclasses import dataclass, replace
import hashlib

import cloudpickle

import numpy as np
import polars as pl

from ins_gbm.data.dtypes import FIT_DTYPE
from ins_gbm.models.base import validate_prediction_type
from ins_gbm.preprocessing.encoder import FittedOneHotEncoder, _MISSING_LEVEL, _NUMERIC_FILL
from ins_gbm.preprocessing.steps import FittedPreprocessingStep


@dataclass
class Member:
    pipeline: object
    # model column -> (diagnostic feature, one-hot level or None)
    columns: dict
    schema: dict


class PredictionAdapter:
    """Keep components processed and categories whole across heterogeneous models."""

    def __init__(self, fitted, data, prediction_type):
        self.fitted = fitted
        self.data = data
        self.ensemble = getattr(fitted, "ensemble", None)
        pipelines = fitted.base_pipelines if self.ensemble is not None else [fitted]
        self.objective = pipelines[0].fitted_model.objective
        validate_prediction_type(prediction_type, self.objective)
        self.prediction_type = prediction_type
        self.members = []
        self.metadata = {}
        self._definitions = {}
        self._raw_uses = set()
        for index, pipeline in enumerate(pipelines):
            prepared = pipeline._prepare_data(data)
            encoder = pipeline.encoder
            if encoder is not None and not isinstance(encoder, FittedOneHotEncoder):
                raise ValueError("Explanation requires the built-in encoder or no encoder")
            lineage = {name: {name} for name in pipeline.input_feature_names}
            levels = {}
            if encoder is not None:
                for raw, values in encoder.levels.items():
                    for level in values:
                        name = f"{raw}__{level}"
                        lineage[name] = {raw}
                        levels[name] = (raw, level)
            definitions = {}
            for step in pipeline.preprocessors:
                targeted = isinstance(step, FittedPreprocessingStep)
                reducer = step.fitted_preprocessor if targeted else step
                inputs = step.input_names if targeted else getattr(reducer, "input_names", None)
                outputs = step.output_names if targeted else reducer.output_feature_names()
                if inputs is None or not hasattr(reducer, "component_mapping"):
                    raise ValueError("Custom preprocessing requires input_names and component_mapping metadata")
                mapping = reducer.component_mapping()
                original_outputs = reducer.output_feature_names()
                if any(name not in mapping for name in original_outputs):
                    raise ValueError("component_mapping must describe every preprocessing output")
                output_origins = {}
                for output, original in zip(outputs, original_outputs):
                    contributing = mapping[original]
                    if not contributing or set(contributing) - set(inputs):
                        raise ValueError("component_mapping must reference the transform's fitted inputs")
                    output_origins[output] = (list(contributing), set().union(*(lineage[n] for n in contributing)))
                # Include upstream fitted state and encoder: equal names are insufficient.
                state = (inputs, outputs, reducer, [definitions.get(n) for n in inputs], encoder)
                fingerprint = hashlib.sha256(cloudpickle.dumps(state)).hexdigest()
                description = {"transform": type(reducer).__name__, "basis": fingerprint}
                if hasattr(reducer, "pca"):
                    description.update(loadings=reducer.pca.components_.tolist(),
                                       explained_variance_ratio=reducer.pca.explained_variance_ratio_.tolist(),
                                       scaler_mean=reducer.scaler.mean_.tolist(),
                                       scaler_scale=reducer.scaler.scale_.tolist())
                for name in outputs:
                    contributing, origins = output_origins[name]
                    lineage[name] = origins
                    definitions[name] = {"inputs": contributing, "source_variables": sorted(origins), **description}
                    levels.pop(name, None)
            columns = {}
            for name in pipeline.fitted_model.feature_names:
                if name in definitions:
                    coordinate = name
                    definition = definitions[name]
                    if self.metadata.get(name, {}).get("stage") == "raw":
                        raise ValueError(f"Incompatible raw/component coordinate name {name!r}; member {index}")
                    existing = self._definitions.get(name)
                    if existing is not None and existing != definition:
                        raise ValueError(f"Incompatible fitted component basis for {name!r} in member {index}")
                    # Different component names for overlapping source families also conflict.
                    for other_name, other in self._definitions.items():
                        if set(other["source_variables"]) & set(definition["source_variables"]):
                            if other["basis"] != definition["basis"]:
                                raise ValueError(f"Incompatible component transforms: {other_name!r}, {name!r}; member {index}")
                    self._definitions[name] = definition
                    self.metadata[name] = {"kind": "numeric", "stage": "processed", **definition}
                    columns[name] = (coordinate, None)
                else:
                    coordinate, level = levels.get(name, (name, None))
                    if coordinate in self._definitions:
                        raise ValueError(f"Incompatible raw/component coordinate name {coordinate!r}; member {index}")
                    self._raw_uses.add(coordinate)
                    columns[name] = (coordinate, level)
                    schema = pipeline._input_schema()
                    explicit = getattr(pipeline.recipe.model, "categorical_features", "auto")
                    categorical = (level is not None or
                                   (schema is not None and coordinate in schema.categorical) or
                                   (explicit != "auto" and coordinate in explicit))
                    categorical = categorical or self.metadata.get(coordinate, {}).get("kind") == "categorical"
                    self.metadata[coordinate] = {
                        "kind": "categorical" if categorical else "numeric", "stage": "raw"}
            self.members.append(Member(pipeline, columns, prepared.features.schema))
        for name, definition in self._definitions.items():
            overlap = self._raw_uses & set(definition["source_variables"])
            if overlap:
                raise ValueError(f"Incompatible raw/component coordinates for {name!r}: {sorted(overlap)}")
        self.names = list(self.metadata)
        if not self.names:
            raise ValueError("No diagnostic features available")
        self.frame = self.transform(data)

    def transform(self, data):
        columns = {}
        for member in self.members:
            prepared = member.pipeline._prepare_data(data).features
            for model_name, (name, _) in member.columns.items():
                if name in columns:
                    continue
                value = prepared[model_name] if name in self._definitions else data.features[name]
                columns[name] = value.rename(name)
        return pl.DataFrame(columns).select(self.names)

    def predict(self, frame, indices):
        """Score diagnostic rows with exposure/offset context from original row indices."""
        indices = np.asarray(indices, dtype=int).tolist()
        data = self.data
        context = replace(data, features=frame, feature_names=frame.columns,
                          target=data.target[indices],
                          exposure=None if data.exposure is None else data.exposure[indices],
                          offset=None if data.offset is None else data.offset[indices],
                          weight=None if data.weight is None else data.weight[indices],
                          cv_fold=None, comparisons=None, schema=None)
        predictions = []
        for member in self.members:
            columns = []
            for model_name, (name, level) in member.columns.items():
                values = frame[name]
                if level is not None:
                    values = (values.cast(pl.String).fill_null(_MISSING_LEVEL) == level).cast(pl.Float32)
                elif self.metadata[name]["stage"] == "raw" and member.pipeline.encoder is not None:
                    values = values.fill_null(_NUMERIC_FILL)
                dtype = member.schema[model_name]
                if self.metadata[name]["kind"] == "numeric":
                    dtype = pl.Float32
                columns.append(values.cast(dtype).rename(model_name))
            prepared = context.with_features(pl.DataFrame(columns))
            scale = "response" if self.ensemble is not None else self.prediction_type
            predictions.append(member.pipeline.fitted_model.predict(prepared, scale).to_numpy())
        matrix = np.column_stack(predictions)
        if self.ensemble is None:
            # Preserve wrapper-specific clipping/conversion exactly for single models.
            result = predictions[0]
        else:
            # Ensemble inference explicitly casts each member to the fit dtype.
            matrix = matrix.astype(FIT_DTYPE, copy=False)
            if hasattr(self.ensemble, "predict_from_predictions"):
                result = self.ensemble.predict_from_predictions(matrix, context).to_numpy()
            elif hasattr(self.ensemble, "weights"):
                result = matrix @ np.asarray(self.ensemble.weights, dtype=FIT_DTYPE)
            else:
                raise ValueError("Unsupported ensemble prediction contract")
            if self.prediction_type == "rate" and context.exposure is not None:
                result = result / context.exposure.to_numpy()
            elif self.prediction_type == "link":
                if np.any(result <= 0):
                    raise ValueError("Log explanations require strictly positive final predictions")
                result = np.log(result)
        if not np.isfinite(result).all():
            raise ValueError("Diagnostic predictions must be finite")
        return np.asarray(result, dtype=float)
