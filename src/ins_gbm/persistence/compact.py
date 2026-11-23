"""Build prediction artifacts without mutating the live training pipeline."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import TYPE_CHECKING

from ins_gbm.preprocessing.encoder import FittedOneHotEncoder
from ins_gbm.preprocessing.pca import FittedPCAReducer
from ins_gbm.preprocessing.pls import FittedPLSReducer
from ins_gbm.preprocessing.steps import FittedPreprocessingStep
from ins_gbm.preprocessing.umap import FittedUMAPReducer

if TYPE_CHECKING:
    from ins_gbm.pipeline import FittedPipeline


def pipeline_for_save(
    pipeline: FittedPipeline, *, include_training_data: bool = False,
) -> FittedPipeline:
    """Keep the full fit on request; otherwise trace model inputs back to raw inputs.

    Opaque custom transforms retain their full input contract: guessing their
    dependencies could silently change predictions.
    """
    if include_training_data:
        return replace(pipeline, input_schema=pipeline._input_schema())

    compact = replace(
        pipeline, raw_train_data=None, input_schema=pipeline._input_schema(),
    )
    encoder = pipeline.encoder
    if encoder is not None and type(encoder) is not FittedOneHotEncoder:
        return compact

    names = (
        encoder.output_feature_names() if encoder is not None
        else list(pipeline.input_feature_names)
    )
    if pipeline.selected_features is not None:
        names = list(pipeline.selected_features)
    stages = []
    reducers = (FittedPCAReducer, FittedPLSReducer, FittedUMAPReducer)
    for step in pipeline.preprocessors:
        if type(step) is FittedPreprocessingStep:
            inputs, outputs = step.input_names, step.output_names
            # A targeted wrapper passes other columns through.
            after = []
            first_input = next(n for n in names if n in inputs)
            for name in names:
                if name == first_input:
                    after.extend(outputs)
                if name not in inputs:
                    after.append(name)
        elif type(step) in reducers:
            inputs, outputs = step.input_names, step.output_names
            after = list(outputs)
        else:
            return compact
        stages.append((list(names), list(inputs), list(outputs)))
        names = after

    required = set(pipeline.fitted_model.feature_names)
    retained = []
    for index in reversed(range(len(stages))):
        before, inputs, outputs = stages[index]
        if required.intersection(outputs):
            retained.append(index)
            required.difference_update(outputs)
            required.update(inputs)
        required.intersection_update(before)
    retained.reverse()

    encoded_names = (
        encoder.output_feature_names() if encoder is not None
        else list(pipeline.input_feature_names)
    )
    selected = [name for name in encoded_names if name in required]
    if encoder is not None:
        levels = {
            col: [level for level in values if f"{col}__{level}" in required]
            for col, values in encoder.levels.items()
            if any(f"{col}__{level}" in required for level in values)
        }
        encoder = replace(
            encoder,
            levels=levels,
            numeric_cols=[n for n in encoder.numeric_cols if n in required],
            ordinal_cols=[n for n in encoder.ordinal_cols if n in required],
            passthrough_cols=[n for n in encoder.passthrough_cols if n in required],
            _output_names=selected,
        )
        raw_required = set(
            encoder.numeric_cols + encoder.ordinal_cols
            + encoder.passthrough_cols + list(levels)
        )
    else:
        raw_required = required
    raw_names = [n for n in pipeline.input_feature_names if n in raw_required]
    schema = pipeline._input_schema()
    if schema is not None:
        schema = replace(schema, **{
            field.name: [n for n in getattr(schema, field.name) if n in raw_required]
            for field in fields(schema)
        })
    final_names = list(pipeline.fitted_model.feature_names)
    # Refits from a compact artifact freeze the retained feature selection.
    # The full original recipe and selection diagnostics belong to full saves.
    recipe = replace(
        pipeline.recipe, selection=None,
        preprocessing=[pipeline.recipe.preprocessing[i] for i in retained],
    )
    metadata = replace(
        pipeline.metadata, input_feature_names=raw_names,
        selected_features=selected, model_selected_features=final_names,
        selection_stages=None,
    )
    return replace(
        compact, input_feature_names=raw_names, input_schema=schema,
        encoder=encoder, selected_features=selected,
        preprocessors=[pipeline.preprocessors[i] for i in retained],
        model_selected_features=final_names, recipe=recipe, metadata=metadata,
        selection_results=None,
    )
