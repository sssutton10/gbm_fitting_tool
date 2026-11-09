"""Held-out permutation importance for a frozen fitted specification."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from ins_gbm.data.folds import CVConfig, resolve_folds
from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.evaluation.metrics import (
    _poisson_rate_metric_inputs,
    gamma_deviance,
    poisson_deviance,
)
from ins_gbm.preprocessing.chain import FittedTransformChain, select_model_features

if TYPE_CHECKING:
    from ins_gbm.pipeline import FittedPipeline


@dataclass
class PermutationImportanceResult:
    """Data-light CV importance tables; positive deltas mean worse predictions.

    ``summary`` equally averages fold means. ``std_importance`` describes
    variation across folds, while ``mean_repeat_std`` averages within-fold
    shuffle variability. Neither is a confidence interval. Standard deviations
    use ddof=0, including when there is just one repeat.

    ``baseline_scores`` records fold sizes and unshuffled deviance;
    ``repeat_importances`` records each shuffled score and signed delta;
    ``fold_importances`` aggregates repeats. ``settings`` records the frozen
    specification and resolved fold policy, without retaining models or rows.
    """

    summary: pl.DataFrame
    fold_importances: pl.DataFrame
    repeat_importances: pl.DataFrame
    baseline_scores: pl.DataFrame
    settings: dict[str, Any]


def _transform_fold(template: FittedPipeline, train: ModelData):
    """Refit transforms locally, applying fixed subsets without a selector."""
    chain = FittedTransformChain(
        input_feature_names=list(template.input_feature_names),
        selected_features=deepcopy(template.selected_features),
        model_selected_features=deepcopy(template.model_selected_features),
    )
    current = train
    if template.recipe.encoder is not None:
        chain.encoder = deepcopy(template.recipe.encoder).fit(
            current.features, current.schema
        )
        current = current.with_features(chain.encoder.transform(current.features))
    if chain.selected_features is not None:
        missing = set(chain.selected_features) - set(current.feature_names)
        if missing:
            raise ValueError(
                f"Frozen selected features missing after fold encoding: {sorted(missing)}"
            )
        current = current.with_features(current.features.select(chain.selected_features))
    for preprocessor in template.recipe.preprocessing:
        fitted = deepcopy(preprocessor).fit(current.features, current.target)
        chain.preprocessors.append(fitted)
        current = current.with_features(fitted.transform(current.features))
    if chain.model_selected_features is not None:
        current = select_model_features(current, chain.model_selected_features)
    return current, chain


def _fit_fold(template: FittedPipeline, train: ModelData):
    """Fit the frozen estimator after fold-local transformations."""
    current, chain = _transform_fold(template, train)
    model = deepcopy(template.recipe.model).fit(
        current, params=deepcopy(template.fitted_model.params)
    )
    return model, chain


def _deviance(data: ModelData, predicted: pl.Series) -> float:
    if len(predicted) != data.n_rows or not np.isfinite(predicted.to_numpy()).all():
        raise ValueError("Permutation importance requires finite, row-aligned predictions")
    if data.objective == "poisson":
        actual, predicted, weight = _poisson_rate_metric_inputs(
            data.target, predicted, data.exposure, data.weight
        )
        score = poisson_deviance(actual, predicted, weight)
    else:
        score = gamma_deviance(data.target, predicted, data.weight)
    if not np.isfinite(score):
        raise ValueError("Permutation importance requires finite deviance scores")
    return score


def permutation_importance_cv(
    fitted: FittedPipeline,
    data: ModelData,
    *,
    cv: CVConfig | None = None,
    n_repeats: int = 5,
    max_samples: int | None = None,
    seed: int = 42,
    show_progress_bar: bool = True,
) -> PermutationImportanceResult:
    """Refit a frozen specification per fold and shuffle held-out raw inputs.

    No selection or tuning is run. Encoding and preprocessing are refitted
    using only training rows. Raw inputs are reported even when an earlier
    selection makes them unused (their importance will be zero). A frozen
    encoded column absent from a training fold raises rather than silently
    changing the specification.

    ``cv`` defaults to auto folds. ``max_samples`` is a positive integer cap
    on validation rows per fold; the same sample is used for the baseline and
    every shuffle. ``seed`` controls sampling/shuffling, independently of the
    fold seed and frozen estimator parameters. Data must be supplied explicitly,
    so this also works with loaded pipelines without attached training rows.

    This audits a fixed specification, not the independence of any prior
    feature selection or tuning performed on these evaluation rows.
    """
    for name, value in (("n_repeats", n_repeats), ("max_samples", max_samples)):
        if value is None and name == "max_samples":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    objective = fitted.fitted_model.objective
    if data.objective is not None and data.objective != objective:
        raise ValueError("data objective conflicts with fitted objective")
    clean = replace(data, objective=objective).select_features(fitted.input_feature_names)
    clean.validate()
    config = cv or CVConfig()
    fold_ids, splits = resolve_folds(clean, config)
    # Keep sample draws independent from shuffle counts and feature counts.
    sample_seed, shuffle_seed = np.random.SeedSequence(seed).spawn(2)
    sample_rng = np.random.default_rng(sample_seed)
    shuffle_rng = np.random.default_rng(shuffle_seed)
    baseline_rows, repeat_rows = [], []
    iterator = tqdm(
        zip(fold_ids, splits), total=len(splits), desc="Permutation importance",
        unit="fold", disable=not show_progress_bar,
    )
    for fold, (train_idx, held_idx) in iterator:
        fold = fold.item() if isinstance(fold, np.generic) else fold
        n_held = len(held_idx)
        if max_samples is not None and n_held > max_samples:
            held_idx = np.sort(sample_rng.choice(held_idx, max_samples, replace=False))
        train = slice_model_data(clean, train_idx).validate(require_multiple_folds=False)
        held = slice_model_data(clean, held_idx).validate(require_multiple_folds=False)
        model, chain = _fit_fold(fitted, train)
        baseline = _deviance(held, model.predict(chain.transform(held)))
        baseline_rows.append({
            "fold": fold, "n_train": train.n_rows, "n_held": n_held,
            "n_evaluated": held.n_rows, "baseline_score": baseline,
        })
        for feature in clean.feature_names:
            for repeat in range(n_repeats):
                shuffled = held.features[feature].gather(
                    shuffle_rng.permutation(held.n_rows)
                )
                permuted = held.with_features(held.features.with_columns(shuffled))
                score = _deviance(permuted, model.predict(chain.transform(permuted)))
                repeat_rows.append({
                    "fold": fold, "feature": feature, "repeat": repeat,
                    "baseline_score": baseline, "permuted_score": score,
                    "importance": score - baseline,
                })
        # Do not retain fold models or training matrices between fits.
        del model, chain, train, held

    repeats = pl.DataFrame(repeat_rows)
    folds = repeats.group_by(["fold", "feature"], maintain_order=True).agg(
        pl.col("importance").mean().alias("mean_importance"),
        pl.col("importance").std(ddof=0).alias("repeat_std"),
    )
    summary = folds.group_by("feature", maintain_order=True).agg(
        pl.col("mean_importance").mean().alias("mean_importance"),
        pl.col("mean_importance").std(ddof=0).alias("std_importance"),
        pl.col("repeat_std").mean().alias("mean_repeat_std"),
        (pl.col("mean_importance") > 0).mean().alias("fraction_positive_folds"),
    ).sort("mean_importance", descending=True, maintain_order=True)
    predefined = config.folds == "predefined" or (
        config.folds == "auto" and clean.cv_fold is not None
    )
    return PermutationImportanceResult(
        summary=summary, fold_importances=folds, repeat_importances=repeats,
        baseline_scores=pl.DataFrame(baseline_rows),
        settings={
            "metric": f"{objective}_deviance", "objective": objective,
            "n_repeats": n_repeats, "max_samples": max_samples, "seed": seed,
            "cv": {"folds": "predefined" if predefined else "random",
                   "n_splits": len(splits), "seed": None if predefined else config.seed},
            "feature_names": list(clean.feature_names),
            "selected_features": deepcopy(fitted.selected_features),
            "model_selected_features": deepcopy(fitted.model_selected_features),
            "model_params": deepcopy(fitted.fitted_model.params),
            "framework": fitted.fitted_model.framework,
        },
    )
