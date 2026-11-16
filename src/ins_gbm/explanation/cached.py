"""Bounded, row-local caches for raw-feature permutation SHAP.

Only known row-independent transforms are optimized. Unknown/custom and UMAP
transforms retain ordinary pipeline inference, including its batch boundaries.
No component contribution is ever redistributed to raw variables.
"""

import math

import numpy as np
import polars as pl

from ins_gbm.evaluation.permutation import _model_raw_features
from ins_gbm.preprocessing.pca import FittedPCAReducer
from ins_gbm.preprocessing.pls import FittedPLSReducer
from ins_gbm.preprocessing.steps import FittedPreprocessingStep
from ins_gbm.preprocessing.encoder import FittedOneHotEncoder, _NUMERIC_FILL


def _key(row):
    # NaN is unequal to itself; normalize it without conflating it with null.
    return tuple(("nan",) if isinstance(v, float) and math.isnan(v) else v for v in row)


def _keys(frame):
    # Numeric coalition matrices dominate wide-model workloads. Copy bytes per
    # row rather than allocating a Python object for every numeric cell. Mixed
    # dtypes/nulls use typed Python values to avoid numeric coercion collisions.
    numeric = [name for name, dtype in frame.schema.items() if dtype.is_numeric()]
    if numeric:
        numbers = frame.select(numeric)
        if len(set(numbers.dtypes)) == 1 and not any(numbers.null_count().row(0)):
            values = np.ascontiguousarray(numbers.to_numpy())
            keys = [row.tobytes() for row in values]
            other = [name for name in frame.columns if name not in numeric]
            if other:
                return [(key, _key(row)) for key, row in zip(keys, frame.select(other).iter_rows())]
            return keys
    return [_key(row) for row in frame.iter_rows()]


class RowCache:
    """Memoize row-independent frame functions without hash-collision shortcuts."""

    def __init__(self, limit, stats, counter):
        self.limit, self.stats, self.counter = limit, stats, counter
        self.values = {}
        self.schema = None

    def apply(self, frame, function):
        keys = _keys(frame)
        missing = {}
        for i, key in enumerate(keys):
            if key not in self.values:
                missing.setdefault(key, i)
        found = self.values
        if missing:
            output = function(frame[list(missing.values())])
            self.schema = output.schema
            fresh = dict(zip(missing, output.iter_rows()))
            found.update(fresh)
            self.stats[self.counter] += len(missing)
        self.stats["cache_hits"] += len(keys) - len(missing)
        result = pl.DataFrame([found[key] for key in keys], schema=self.schema, orient="row")
        # Caches live for one explained row and never grow with the population.
        if len(self.values) > self.limit:
            self.values.clear()
        return result


class CachedRawPredictor:
    """Score one explained row's coalitions, retaining its exposure and offset."""

    def __init__(self, report, source_row, bank, limit, stats):
        self.report, self.adapter, self.source = report, report.adapter, source_row
        self.stats = stats
        self.plans = []
        raw_bank = self.raw(bank)
        for member in self.adapter.members:
            pipeline = member.pipeline
            reducers = [s.fitted_preprocessor if isinstance(s, FittedPreprocessingStep) else s
                        for s in pipeline.preprocessors]
            supported = (type(pipeline.encoder) in {type(None), FittedOneHotEncoder}
                         and all(type(r) in {FittedPCAReducer, FittedPLSReducer} for r in reducers)
                         and all(not isinstance(s, FittedPreprocessingStep) or type(s) is FittedPreprocessingStep
                                 for s in pipeline.preprocessors))
            if not supported:
                self.plans.append(None)
                continue
            encoded_columns = None
            if pipeline.encoder is not None:
                encoded = pipeline.encoder.transform(raw_bank.select(pipeline.input_feature_names))
                origins = {name: name for name in encoded.columns}
                for name, levels in pipeline.encoder.levels.items():
                    origins.update({f"{name}__{level}": name for level in levels})
                encoded_columns = []
                for name in encoded.columns:
                    origin = origins[name]
                    lookup = ({_key((value,)): result for value, result in
                               zip(raw_bank[origin], encoded[name])}
                              if origin in pipeline.encoder.levels else None)
                    encoded_columns.append((name, origin, lookup, encoded.schema[name]))
            self.plans.append((
                _model_raw_features(pipeline), encoded_columns,
                RowCache(limit, stats, "member_rows_evaluated"),
                [RowCache(limit, stats, "transform_rows_evaluated") for _ in reducers],
            ))

    def raw(self, frame):
        indices = np.full(frame.height, self.source, dtype=int)
        return self.adapter.data.features[indices.tolist()].with_columns(frame.get_columns())

    def context(self, frame):
        return self.adapter.context(frame, np.full(frame.height, self.source, dtype=int))

    def prepare(self, pipeline, raw, encoded_columns, stage_caches):
        if encoded_columns is None:
            frame = raw.select(pipeline.input_feature_names)
        else:
            frame = pl.DataFrame([
                (pl.Series(name, [lookup[_key((v,))] for v in raw[origin]], dtype=dtype)
                 if lookup is not None else
                 raw[origin].fill_null(_NUMERIC_FILL).cast(dtype).rename(name)
                 if origin not in pipeline.encoder.passthrough_cols else
                 raw[origin].cast(dtype).rename(name))
                for name, origin, lookup, dtype in encoded_columns
            ])
        if pipeline.selected_features is not None:
            frame = frame.select(pipeline.selected_features)
        for step, cache in zip(pipeline.preprocessors, stage_caches):
            if isinstance(step, FittedPreprocessingStep):
                def transform(inputs):
                    output = step.fitted_preprocessor.transform(inputs)
                    return output.rename(dict(zip(output.columns, step.output_names)))

                output = cache.apply(frame.select(step.input_names), transform)
                first = next(n for n in frame.columns if n in step.input_names)
                columns = []
                for name in frame.columns:
                    if name == first:
                        columns.extend(output.get_columns())
                    if name not in step.input_names:
                        columns.append(frame[name])
                frame = pl.DataFrame(columns)
            else:
                frame = cache.apply(frame, step.transform)
        if pipeline.model_selected_features is not None:
            frame = frame.select(pipeline.model_selected_features)
        return frame

    def __call__(self, frame):
        output = []
        for start in range(0, frame.height, self.report.batch_size):
            self.report._event("prediction", start, frame.height)
            batch = frame.slice(start, self.report.batch_size)
            raw = self.raw(batch)
            predictions = []
            scale = "response" if self.adapter.ensemble is not None else self.adapter.prediction_type
            for member, plan in zip(self.adapter.members, self.plans):
                pipeline = member.pipeline
                self.stats["member_rows_requested"] += batch.height
                if plan is None:
                    self.stats["uncached_member_rows"] += batch.height
                    values = pipeline.predict(self.context(raw), scale).to_numpy()
                else:
                    dependencies, encoded, prediction_cache, stages = plan

                    def predict(inputs):
                        # Nondependent required inputs stay at this row's original
                        # values; lineage proves that they cannot affect output.
                        current = self.raw(inputs)
                        prepared = self.prepare(pipeline, current, encoded, stages)
                        return pl.DataFrame({"prediction": pipeline.fitted_model.predict(
                            self.context(prepared), scale)})

                    values = prediction_cache.apply(raw.select(dependencies), predict)["prediction"].to_numpy()
                predictions.append(values)
            output.append(self.adapter.combine(predictions, self.context(raw)))
        return np.concatenate(output)
