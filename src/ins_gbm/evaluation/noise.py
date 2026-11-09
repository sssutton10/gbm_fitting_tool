"""Native feature importance relative to matched, shuffled shadow features."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from ins_gbm.data.folds import CVConfig, resolve_folds
from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.data.schema import FeatureSchema, infer_schema
from ins_gbm.evaluation.permutation import _transform_fold
from ins_gbm.preprocessing.encoder import FittedOneHotEncoder

if TYPE_CHECKING:
    from ins_gbm.pipeline import FittedPipeline


_DEFAULTS = {
    "lightgbm": "gain",
    "xgboost": "total_gain",
    "catboost": "PredictionValuesChange",
    "random_forest": "impurity",
}
_SUPPORTED = {
    "lightgbm": {"gain", "split"},
    "xgboost": {"weight", "gain", "cover", "total_gain", "total_cover"},
    "catboost": {"FeatureImportance", "PredictionValuesChange", "LossFunctionChange"},
    "random_forest": {"impurity"},
}


@dataclass
class NoiseScreeningResult:
    """Training-importance diagnostic; win fractions are not p-values.

    ``fit_importances`` contains raw and normalized real/shadow scores for each
    fold and shadow realization. Shares divide by the sum of absolute grouped
    real and shadow scores in that fit (zero totals yield zero shares).
    ``fold_importances`` averages realizations; ``summary`` equally averages
    those fold summaries. Standard deviations use ddof=0. ``fit_details`` stores
    sizes and normalization totals, and ``settings`` describes the assessment.
    No fitted models or training rows are retained.
    """

    summary: pl.DataFrame
    fold_importances: pl.DataFrame
    fit_importances: pl.DataFrame
    fit_details: pl.DataFrame
    settings: dict[str, Any]


def _feature_groups(fitted: FittedPipeline) -> tuple[str, dict[str, list[str]]]:
    names = list(fitted.fitted_model.feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Fitted model must have non-empty, unique feature names")
    # General transforms can mix inputs. Report their final columns explicitly.
    if fitted.preprocessors:
        return "model", {name: [name] for name in names}
    if fitted.encoder is None:
        return "raw", {name: [name] for name in names}
    if not isinstance(fitted.encoder, FittedOneHotEncoder):
        return "model", {name: [name] for name in names}
    remaining = set(names)
    groups = {}
    for raw in fitted.input_feature_names:
        candidates = (
            [f"{raw}__{level}" for level in fitted.encoder.levels[raw]]
            if raw in fitted.encoder.levels else [raw]
        )
        columns = [name for name in candidates if name in remaining]
        if columns:
            groups[raw] = columns
            remaining.difference_update(columns)
    if remaining:
        raise ValueError(f"Cannot map fitted encoded columns to inputs: {sorted(remaining)}")
    return "raw", groups


def _augment(data: ModelData, groups: dict[str, list[str]], rng):
    """Shuffle each group with one row permutation, preserving its joint design."""
    reserved = {f"__noise_shadow_{i}" for i in range(len(data.feature_names))}
    if reserved.intersection(data.feature_names):
        raise ValueError("Model feature names collide with reserved __noise_shadow_ names")
    shadow_by_column = {
        name: f"__noise_shadow_{i}" for i, name in enumerate(data.feature_names)
    }
    shadows = []
    for columns in groups.values():
        order = rng.permutation(data.n_rows)
        shadows.extend(
            data.features[name].gather(order).alias(shadow_by_column[name])
            for name in columns
        )
    features = data.features.with_columns(shadows)
    # Infer transformed columns missing from the original raw schema, but retain
    # explicit categorical/ordinal roles (including numeric-coded categories).
    inferred = infer_schema(data.features, data.feature_names)
    roles = {}
    for role in ("numeric", "categorical", "ordinal", "passthrough"):
        for name in getattr(inferred, role):
            roles[name] = role
    if data.schema is not None:
        for role in ("numeric", "categorical", "ordinal", "passthrough"):
            for name in getattr(data.schema, role):
                if name in shadow_by_column:
                    roles[name] = role
    schema = FeatureSchema(numeric=[], categorical=[])
    for name in data.feature_names:
        getattr(schema, roles[name]).extend([name, shadow_by_column[name]])
    return replace(data, features=features, feature_names=features.columns, schema=schema), shadow_by_column


def _importance_scores(model, kind: str, names: list[str]) -> dict[str, float]:
    table = model.feature_importance(kind)
    if not {"feature", "importance"}.issubset(table.columns):
        raise ValueError("Feature importance requires 'feature' and 'importance' columns")
    reported = table["feature"].to_list()
    if len(set(reported)) != len(reported) or not set(reported).issubset(names):
        raise ValueError("Feature importance contains duplicate or unknown features")
    try:
        scores = dict(zip(reported, map(float, table["importance"].to_list())))
    except (TypeError, ValueError) as exc:
        raise ValueError("Feature importance scores must be numeric") from exc
    if not np.isfinite(list(scores.values())).all():
        raise ValueError("Feature importance scores must be finite")
    return {name: scores.get(name, 0.0) for name in names}


def noise_screening_cv(
    fitted: FittedPipeline,
    data: ModelData,
    *,
    cv: CVConfig | None = None,
    importance_type: str | None = None,
    n_shadow_repeats: int = 3,
    seed: int = 42,
    show_progress_bar: bool = True,
) -> NoiseScreeningResult:
    """Refit the frozen model with matched noise on each CV training split.

    Defaults: LightGBM gain, XGBoost total_gain, CatBoost PredictionValuesChange,
    Random Forest impurity. An explicit ``importance_type`` overrides these.
    Selection and tuning never run; transforms are refitted on training rows.
    Shadows are appended to the final model design; encoded levels of one raw
    variable share a permutation and their importance is summed. After general
    preprocessing or custom encoding, report individual model columns instead.

    Results measure training usage relative to noise, not held-out predictive
    value or statistical significance. Ties do not count as beating noise.
    Each repeat generates fresh shadows but retains the fitted estimator seed.
    Data is explicit, supporting saved pipelines without attached training data.
    """
    if isinstance(n_shadow_repeats, bool) or not isinstance(n_shadow_repeats, int) or n_shadow_repeats < 1:
        raise ValueError("n_shadow_repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    framework = fitted.fitted_model.framework
    kind = _DEFAULTS.get(framework) if importance_type is None else importance_type
    if not isinstance(kind, str) or not kind:
        raise ValueError("Specify importance_type for an unrecognized model framework")
    if framework in _SUPPORTED and kind not in _SUPPORTED[framework]:
        raise ValueError(f"{framework} importance_type must be one of {sorted(_SUPPORTED[framework])}")
    if not fitted.recipe.model.capabilities().supports_feature_importance:
        raise ValueError("Model does not support native feature importance")
    objective = fitted.fitted_model.objective
    if data.objective is not None and data.objective != objective:
        raise ValueError("data objective conflicts with fitted objective")
    clean = replace(data, objective=objective).select_features(fitted.input_feature_names)
    clean.validate()
    stage, groups = _feature_groups(fitted)
    names = list(fitted.fitted_model.feature_names)
    config = cv or CVConfig()
    fold_ids, splits = resolve_folds(clean, config)
    rng = np.random.default_rng(seed)
    rows, details = [], []
    progress = tqdm(total=len(splits) * n_shadow_repeats, desc="Noise screening", unit="fit", disable=not show_progress_bar)
    try:
        for fold, (train_idx, _) in zip(fold_ids, splits):
            fold = fold.item() if isinstance(fold, np.generic) else fold
            train = slice_model_data(clean, train_idx).validate(require_multiple_folds=False)
            current, chain = _transform_fold(fitted, train)
            missing = set(names) - set(current.feature_names)
            if missing:
                raise ValueError(f"Frozen model features missing from fold design: {sorted(missing)}")
            current = current.select_features(names)
            for repeat in range(n_shadow_repeats):
                augmented, shadows = _augment(current, groups, rng)
                wrapper = deepcopy(fitted.recipe.model)
                # Explicit native categorical lists must include matched shadows.
                categorical = getattr(wrapper, "categorical_features", None)
                if isinstance(categorical, list):
                    wrapper.categorical_features = [
                        *categorical, *(shadows[name] for name in categorical),
                    ]
                model = wrapper.fit(augmented, params=deepcopy(fitted.fitted_model.params))
                scores = _importance_scores(model, kind, augmented.feature_names)
                real = {group: sum(scores[name] for name in cols) for group, cols in groups.items()}
                noise = {group: sum(scores[shadows[name]] for name in cols) for group, cols in groups.items()}
                total = sum(abs(value) for value in [*real.values(), *noise.values()])
                denominator = total or 1.0
                strongest = max(noise.values())
                details.append({"fold": fold, "shadow_repeat": repeat,
                                "n_train": current.n_rows, "normalization_total": total})
                for feature in groups:
                    rows.append({
                        "fold": fold, "shadow_repeat": repeat, "feature": feature,
                        "importance": real[feature], "shadow_importance": noise[feature],
                        "max_shadow_importance": strongest,
                        "importance_share": real[feature] / denominator,
                        "shadow_share": noise[feature] / denominator,
                        "excess_over_shadow": (real[feature] - noise[feature]) / denominator,
                        "excess_over_max_shadow": (real[feature] - strongest) / denominator,
                        "beats_shadow": real[feature] > noise[feature],
                        "beats_max_shadow": real[feature] > strongest,
                    })
                del model, wrapper, augmented
                progress.update()
            del chain, current, train
    finally:
        progress.close()
    fits = pl.DataFrame(rows)
    averages = {
        "importance": "mean_importance", "shadow_importance": "mean_shadow_importance",
        "max_shadow_importance": "mean_max_shadow_importance",
        "importance_share": "mean_importance_share", "shadow_share": "mean_shadow_share",
        "excess_over_shadow": "mean_excess_over_shadow",
        "excess_over_max_shadow": "mean_excess_over_max_shadow",
        "beats_shadow": "fraction_beating_shadow", "beats_max_shadow": "fraction_beating_max_shadow",
    }
    folds = fits.group_by(["fold", "feature"], maintain_order=True).agg(
        *[pl.col(source).mean().alias(target) for source, target in averages.items()]
    )
    summary = folds.group_by("feature", maintain_order=True).agg(
        *[pl.col(name).mean() for name in averages.values()],
        pl.col("mean_importance_share").std(ddof=0).alias("std_importance_share"),
    ).sort("mean_excess_over_max_shadow", descending=True, maintain_order=True)
    predefined = config.folds == "predefined" or (config.folds == "auto" and clean.cv_fold is not None)
    return NoiseScreeningResult(
        summary=summary, fold_importances=folds, fit_importances=fits,
        fit_details=pl.DataFrame(details),
        settings={
            "framework": framework, "importance_type": kind, "objective": objective,
            "n_shadow_repeats": n_shadow_repeats, "seed": seed, "feature_stage": stage,
            "feature_groups": deepcopy(groups), "model_params": deepcopy(fitted.fitted_model.params),
            "selected_features": deepcopy(fitted.selected_features),
            "model_selected_features": deepcopy(fitted.model_selected_features),
            "cv": {"folds": "predefined" if predefined else "random", "n_splits": len(splits),
                   "seed": None if predefined else config.seed},
        },
    )
