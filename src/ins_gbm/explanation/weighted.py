"""Explicit aggregation of native member raw-score explanations.

This is a diagnostic with per-member tree-path references, not a replacement
for SHAP of the final response blend on a common raw-feature/background game.
"""

from dataclasses import replace
from time import perf_counter

import numpy as np
import polars as pl

from ins_gbm.data.dtypes import FIT_DTYPE, frame_to_fit_array, replace_value_with_nan
from ins_gbm.data.model_data import slice_model_data
from ins_gbm.ensemble.blending import FittedBlendingEnsemble
from ins_gbm.preprocessing.encoder import _NUMERIC_FILL

from .adapter import PredictionAdapter
from .results import ShapResult


def _native_input(fitted, prepared):
    names = fitted.feature_names
    if fitted.framework == "lightgbm":
        converter = getattr(fitted, "native_input_fn", None)
        if converter is not None:
            return converter(prepared)
        # Numeric legacy artifacts remain usable. Never relearn categorical
        # codes from the explanation rows: they must come from the fitted map.
        info = fitted.model.dump_model().get("feature_infos", {})
        if (any(not prepared.features.schema[n].is_numeric() for n in names)
                or any(info.get(n, {}).get("values") for n in names)):
            raise ValueError("This LightGBM artifact lacks its fitted native category input adapter; "
                             "refit and save with the updated wrapper")
        return replace_value_with_nan(frame_to_fit_array(prepared.features, names), _NUMERIC_FILL)
    if fitted.framework == "xgboost":
        import xgboost as xgb
        return xgb.DMatrix(frame_to_fit_array(prepared.features, names), feature_names=names,
                           missing=_NUMERIC_FILL,
                           nthread=fitted.params.get("nthread", fitted.params.get("n_jobs", -1)))
    from catboost import Pool
    from ins_gbm.models.catboost import _pool_matrix
    categorical = [names[i] for i in fitted.model.get_cat_feature_indices()]
    return Pool(_pool_matrix(prepared.features, names, categorical),
                feature_names=names, cat_features=categorical)


def _native_raw(fitted, inputs):
    if fitted.framework == "lightgbm":
        return fitted.model.predict(inputs, raw_score=True)
    if fitted.framework == "xgboost":
        return fitted.model.predict(inputs, output_margin=True)
    return fitted.model.predict(inputs, prediction_type="RawFormulaVal")


def weighted_member_shap(report, *, max_samples, weights, group_one_hot):
    if type(report.adapter.ensemble) is not FittedBlendingEnsemble:
        raise ValueError("weighted_member_shap requires a fixed-weight blend, not stacking or a single model")
    if not isinstance(group_one_hot, bool):
        raise ValueError("group_one_hot must be a bool")
    members = report.adapter.members
    if any(m.pipeline.fitted_model.framework not in {"lightgbm", "xgboost", "catboost"} for m in members):
        raise ValueError("weighted_member_shap requires boosted LightGBM/XGBoost/CatBoost members on raw log scales; "
                         "Random Forest native response scores cannot be mixed with log scores")
    blend_weights = np.asarray(report.adapter.ensemble.weights, dtype=FIT_DTYPE).astype(float)
    if (blend_weights.shape != (len(members),) or not np.isfinite(blend_weights).all()
            or np.any(blend_weights < 0) or not np.any(blend_weights > 0)):
        raise ValueError("Blend weights must be finite, nonnegative, nonzero and align with members")
    indices = report._rows(max_samples)
    summary_weights = report._weights(weights, indices)
    selected = slice_model_data(report.data, indices.tolist())
    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP explanations require ins_gbm[explain] (pip install 'ins_gbm[explain]')") from exc

    started = perf_counter()
    report._event("weighted_member_shap", 0, len(members))
    columns, metadata, contributions = {}, {}, {}
    baselines, predictions = np.zeros(len(indices)), np.zeros(len(indices))
    member_settings = []
    reserved = set(report.data.features.columns)

    def qualified(index, name):
        label = f"member[{index}]::{name}"
        while label in reserved or label in columns:
            label += "#"
        return label

    for index, (member, weight) in enumerate(zip(members, blend_weights)):
        pipeline = member.pipeline
        fitted = pipeline.fitted_model
        # Single-member processed coordinates allow PCA/raw overlap across
        # members without pretending that their component games are identical.
        prepared = pipeline._prepare_data(selected)
        adapter = PredictionAdapter(pipeline, selected, "link", "processed", _prepared=prepared)
        mapping = {}
        for model_column, (coordinate, level) in adapter.members[0].columns.items():
            meta = dict(adapter.metadata[coordinate])
            component = meta["stage"] == "processed"
            if component:
                label = qualified(index, coordinate)
                values = prepared.features[model_column]
            elif level is not None and not group_one_hot:
                label = qualified(index, model_column)
                values = prepared.features[model_column]
                meta = dict(kind="numeric", stage="encoded", source_variable=coordinate, level=level)
            else:
                label = coordinate
                values = selected.features[coordinate]
            mapping[model_column] = label
            if label not in columns:
                columns[label] = values.rename(label)
                metadata[label] = {**meta, "members": [], "encoded_columns": {}}
                contributions[label] = np.zeros(len(indices))
            if index not in metadata[label]["members"]:
                metadata[label]["members"].append(index)
            metadata[label]["encoded_columns"].setdefault(str(index), []).append(model_column)

        explainer = shap.TreeExplainer(fitted.model, model_output="raw",
                                       feature_perturbation="tree_path_dependent",
                                       feature_names=fitted.feature_names)
        adjustment = None
        member_bases = []
        for start in range(0, len(indices), report.batch_size):
            stop = min(start + report.batch_size, len(indices))
            report._event("weighted_member_shap_batch", start, len(indices), member_index=index)
            batch = slice_model_data(prepared, list(range(start, stop)))
            inputs = _native_input(fitted, batch)
            values = np.asarray(explainer.shap_values(inputs, check_additivity=True), dtype=float)
            base = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
            if values.shape != (stop - start, len(fitted.feature_names)) or base.size != 1:
                raise ValueError(f"Unsupported native raw SHAP output for member {index}")
            native = np.asarray(_native_raw(fitted, inputs), dtype=float).reshape(-1)
            if not np.allclose(native, base[0] + values.sum(axis=1), atol=1e-6, rtol=2e-5):
                raise ValueError(f"Native raw SHAP reconstruction failed for member {index}")
            # Match the wrapper's context-free intercept convention (XGBoost
            # training base margins can replace the native learned intercept).
            actual = fitted.predict_fn(replace(batch, exposure=None, offset=None), "link").to_numpy().astype(float)
            delta = actual - native
            if adjustment is None:
                adjustment = float(delta[0])
            if not np.allclose(delta, adjustment, atol=1e-6, rtol=2e-5):
                raise ValueError(f"Member {index} wrapper raw scores are not affine-equivalent to native tree output")
            baseline = float(base[0]) + adjustment
            if not np.isfinite(values).all() or not np.isfinite(actual).all() or not np.isfinite(baseline):
                raise ValueError(f"Nonfinite native raw explanation for member {index}")
            for j, name in enumerate(fitted.feature_names):
                contributions[mapping[name]][start:stop] += weight * values[:, j]
            baselines[start:stop] += weight * baseline
            predictions[start:stop] += weight * actual
            member_bases.append(baseline)
        member_settings.append(dict(member_index=index, framework=fitted.framework,
                                    weight=float(weight), coordinate_mapping=mapping,
                                    raw_baselines_by_batch=member_bases,
                                    wrapper_intercept_adjustment=adjustment))
        report._event("weighted_member_shap", index + 1, len(members),
                      elapsed_seconds=perf_counter() - started)

    names = list(columns)
    matrix = np.column_stack([contributions[n] for n in names])
    residual = predictions - baselines - matrix.sum(axis=1)
    if not np.allclose(predictions, baselines + matrix.sum(axis=1), atol=1e-6, rtol=2e-5):
        raise ValueError("Weighted member raw-score reconstruction failed")
    limits = [
        "Explains sum(weight * member context-free log score), not the final response blend or log(final blend).",
        "Fixed blend weights do not reflect row-specific response-scale influence; opposing signed effects can cancel.",
        "Tree-path references are model-specific; this is not common-background raw-feature ensemble SHAP.",
        "Summed one-hot contributions are display groups, not SHAP with the original category masked as one player.",
        "Components remain member-qualified; raw inputs through PCA/other transforms are not attributed to source features.",
        "Scoring exposure and offset are excluded; fitted model parameters still reflect training context.",
    ]
    settings = report._settings(
        "weighted_member_shap", indices, summary_weights,
        prediction_type="weighted_member_raw", units="weighted member log score (not final blend)",
        explanation_target="weighted_member_raw", display_label="Weighted member raw-score diagnostic",
        score_label="Diagnostic score", context="Scoring exposure and offset excluded",
        feature_space="member_processed", feature_metadata=metadata, features=names,
        interpretation="Weighted signed native member contributions; see limitations. Not causal effects.",
        engine="tree", perturbation="tree_path_dependent", background_indices=None,
        background_distribution="Model-specific training tree paths; supplied background is not used",
        shap_version=shap.__version__, members=member_settings, blend_weights=blend_weights.tolist(),
        group_one_hot=group_one_hot, n_repeats=1, limitations=limits,
        elapsed_seconds=perf_counter() - started,
    )
    table = pl.DataFrame(dict(repeat=np.zeros(len(indices) * len(names), dtype=int),
                              row=np.repeat(np.arange(len(indices)), len(names)),
                              feature=pl.Series("feature", names).gather(
                                  np.tile(np.arange(len(names), dtype=np.int32), len(indices))),
                              contribution=matrix.reshape(-1)))
    prediction_table = pl.DataFrame(dict(repeat=np.zeros(len(indices), dtype=int),
                                         row=np.arange(len(indices)), source_row=indices,
                                         base_value=baselines, prediction=predictions, residual=residual))
    return ShapResult(table, prediction_table, pl.DataFrame(list(columns.values())), settings, summary_weights)
