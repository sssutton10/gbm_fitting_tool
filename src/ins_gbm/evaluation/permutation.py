"""Raw-feature permutation importance for fixed models and ensembles."""

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
from ins_gbm.preprocessing.encoder import FittedOneHotEncoder
from ins_gbm.preprocessing.steps import FittedPreprocessingStep

if TYPE_CHECKING:
    from ins_gbm.ensemble.pipeline import EnsemblePipeline, EnsembleResult
    from ins_gbm.pipeline import FittedPipeline


@dataclass
class HoldoutPermutationImportanceResult:
    """Final-ensemble deviance increases on one held-out sample.

    ``std_importance`` is shuffle variability (ddof=0), not a confidence
    interval. Positive importance means shuffling worsened predictions.
    No models or evaluation rows are retained.
    """

    summary: pl.DataFrame
    repeat_importances: pl.DataFrame
    baseline_score: float
    settings: dict[str, Any]


def _model_raw_features(pipeline: FittedPipeline) -> list[str]:
    """Trace final design columns to raw inputs, including selected components."""
    lineage = {name: {name} for name in pipeline.input_feature_names}
    encoder = pipeline.encoder
    if isinstance(encoder, FittedOneHotEncoder):
        encoded = {name: lineage[name] for name in (
            encoder.numeric_cols + encoder.ordinal_cols + encoder.passthrough_cols
        )}
        for raw, levels in encoder.levels.items():
            for level in levels:
                encoded[f"{raw}__{level}"] = {raw}
        lineage = encoded
    elif encoder is not None:
        if not hasattr(encoder, "component_mapping"):
            raise ValueError("Permutation importance requires component_mapping for custom encoders")
        lineage = _map_origins(encoder.component_mapping(), lineage)
    if pipeline.selected_features is not None:
        lineage = {name: lineage[name] for name in pipeline.selected_features}
    for step in pipeline.preprocessors:
        if isinstance(step, FittedPreprocessingStep):
            reducer = step.fitted_preprocessor
            if not hasattr(reducer, "component_mapping"):
                raise ValueError("Permutation importance requires component_mapping for custom preprocessing")
            mapping = reducer.component_mapping()
            mapped = _map_origins({
                output: mapping[original]
                for output, original in zip(step.output_names, reducer.output_feature_names())
            }, lineage)
            lineage = {name: origins for name, origins in lineage.items()
                       if name not in step.input_names}
            lineage.update(mapped)
        else:
            if not hasattr(step, "component_mapping"):
                raise ValueError("Permutation importance requires component_mapping for custom preprocessing")
            lineage = _map_origins(step.component_mapping(), lineage)
    try:
        used = set().union(*(lineage[name] for name in pipeline.fitted_model.feature_names))
    except KeyError as exc:
        raise ValueError(f"Cannot trace final model feature {exc.args[0]!r} to raw inputs") from exc
    return [name for name in pipeline.input_feature_names if name in used]


def _map_origins(mapping, lineage):
    try:
        return {output: set().union(*(lineage[name] for name in inputs))
                for output, inputs in mapping.items()}
    except KeyError as exc:
        raise ValueError(f"Invalid feature lineage for input {exc.args[0]!r}") from exc


def _ensemble_raw_features(pipelines):
    features = list(dict.fromkeys(
        name for pipeline in pipelines for name in _model_raw_features(pipeline)
    ))
    if not features:
        raise ValueError("Permutation importance requires at least one raw model feature")
    return features


def ensemble_permutation_importance(
    fitted: EnsembleResult,
    data: ModelData,
    *,
    n_repeats: int = 5,
    max_samples: int | None = None,
    seed: int = 42,
    show_progress_bar: bool = True,
) -> HoldoutPermutationImportanceResult:
    """Shuffle raw inputs once and score the complete, fixed ensemble.

    Supply separate held-out data. No member, encoder, preprocessor, blend
    weight, or stacker is refitted. Every member sees the same raw shuffle
    through its own fitted transform chain, including one-hot encoding.
    ``max_samples`` caps evaluation rows; baseline and shuffles use the same
    sample. Targets, exposure, weights, and offsets stay aligned.
    """
    from ins_gbm.ensemble._utils import _validate_ensemble_pipelines

    _validate_options(n_repeats, max_samples, seed)
    _validate_ensemble_pipelines(fitted.base_pipelines, require_training=False)
    objective = fitted.base_pipelines[0].fitted_model.objective
    if data.objective is not None and data.objective != objective:
        raise ValueError("data objective conflicts with fitted objective")
    # Preserve first-seen order across members with different raw input subsets.
    input_features = list(dict.fromkeys(
        name for member in fitted.base_pipelines for name in member.input_feature_names
    ))
    features = _ensemble_raw_features(fitted.base_pipelines)
    clean = replace(data, objective=objective).select_features(input_features)
    clean.validate(require_multiple_folds=False)
    n_held = clean.n_rows
    sample_seed, shuffle_seed = np.random.SeedSequence(seed).spawn(2)
    sample_rng = np.random.default_rng(sample_seed)
    shuffle_rng = np.random.default_rng(shuffle_seed)
    if max_samples is not None and n_held > max_samples:
        indices = np.sort(sample_rng.choice(n_held, max_samples, replace=False))
        clean = slice_model_data(clean, indices).validate(require_multiple_folds=False)
    baseline = _deviance(clean, fitted.predict(clean))
    rows = []
    for feature in tqdm(features, desc="Ensemble permutation importance",
                        unit="feature", disable=not show_progress_bar):
        for repeat in range(n_repeats):
            shuffled = clean.features[feature].gather(shuffle_rng.permutation(clean.n_rows))
            permuted = clean.with_features(clean.features.with_columns(shuffled))
            score = _deviance(permuted, fitted.predict(permuted))
            rows.append({
                "feature": feature, "repeat": repeat, "baseline_score": baseline,
                "permuted_score": score, "importance": score - baseline,
            })
    repeats = pl.DataFrame(rows)
    summary = repeats.group_by("feature", maintain_order=True).agg(
        pl.col("importance").mean().alias("mean_importance"),
        pl.col("importance").std(ddof=0).alias("std_importance"),
    ).sort("mean_importance", descending=True, maintain_order=True)
    return HoldoutPermutationImportanceResult(
        summary=summary, repeat_importances=repeats, baseline_score=baseline,
        settings={
            "evaluation": "holdout", "metric": f"{objective}_deviance",
            "objective": objective, "n_repeats": n_repeats,
            "max_samples": max_samples, "seed": seed, "n_held": n_held,
            "n_evaluated": clean.n_rows, "feature_names": features,
        },
    )


def _validate_options(n_repeats, max_samples, seed):
    for name, value in (("n_repeats", n_repeats), ("max_samples", max_samples)):
        if value is None and name == "max_samples":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


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


def _fit_fold(template: FittedPipeline, train: ModelData, *, params=None):
    """Fit the frozen estimator after fold-local transformations."""
    current, chain = _transform_fold(template, train)
    model = deepcopy(template.recipe.model).fit(
        current, params=deepcopy(template.fitted_model.params if params is None else params)
    )
    return model, chain


def ensemble_permutation_importance_cv(
    candidate: EnsemblePipeline,
    data: ModelData,
    *,
    cv: CVConfig | None = None,
    n_repeats: int = 5,
    max_samples: int | None = None,
    seed: int = 42,
    show_progress_bar: bool = True,
) -> PermutationImportanceResult:
    """Audit a frozen ensemble specification on outer held-out folds.

    Only fixed-weight blending, seed averaging, and bagging are supported.
    Frozen selections and fitted parameters are reused, without selection,
    tuning, or weight optimization. This audits a chosen specification; prior
    selection or tuning on these rows is not independently validated.
    """
    from ins_gbm.ensemble._utils import _validate_ensemble_pipelines
    from ins_gbm.ensemble.blending import BlendingEnsemble, FittedBlendingEnsemble
    from ins_gbm.ensemble.pipeline import EnsembleResult
    from ins_gbm.ensemble.resampling import ResamplingEnsemble, _member_params

    _validate_options(n_repeats, max_samples, seed)
    resampling = candidate.method in {"seed_average", "bagging"}
    if not resampling and not (
        candidate.method == "blending" and candidate.blend_mode == "fixed"
    ):
        raise ValueError(
            "CV permutation importance supports only fixed-weight blending, "
            "seed_average, and bagging; optimized blending and stacking are unsupported"
        )
    templates = candidate.fitted_pipelines
    _validate_ensemble_pipelines(templates, require_training=False)
    if resampling and len(templates) != 1:
        raise ValueError(f"{candidate.method} requires exactly one fitted template pipeline")
    objective = templates[0].fitted_model.objective
    if data.objective is not None and data.objective != objective:
        raise ValueError("data objective conflicts with fitted objective")
    input_features = list(dict.fromkeys(
        name for member in templates for name in member.input_feature_names
    ))
    features = _ensemble_raw_features(templates)
    clean = replace(data, objective=objective).select_features(input_features).validate()
    config = cv or CVConfig()
    fold_ids, splits = resolve_folds(clean, config)
    if resampling:
        sampler = ResamplingEnsemble(
            candidate.method, candidate.n_members, candidate.seed,
            candidate.sample_fraction, candidate.bootstrap,
        )
        _, settings = sampler.prepare_samples(clean)
        _member_params(templates[0].fitted_model.params,
                       templates[0].fitted_model.framework, settings["member_seeds"][0])
        weights = [1.0 / candidate.n_members] * candidate.n_members
    else:
        weights = BlendingEnsemble(weights=candidate.blend_weights)._fit_fixed(templates).weights
        if not np.isfinite(weights).all() or np.any(np.asarray(weights) < 0):
            raise ValueError("Fixed weights must be finite and nonnegative")

    # Separate per-fold shuffle streams from folds and ensemble sampling/seeds.
    fold_seeds = np.random.SeedSequence(seed).spawn(len(splits))
    baseline_rows, repeat_frames, fold_settings = [], [], []
    iterator = tqdm(zip(fold_ids, splits, fold_seeds), total=len(splits),
                    desc="Ensemble permutation importance", unit="fold",
                    disable=not show_progress_bar)
    for fold, (train_idx, held_idx), shuffle_seed in iterator:
        train = slice_model_data(clean, train_idx).validate(require_multiple_folds=False)
        held = slice_model_data(clean, held_idx).validate(require_multiple_folds=False)
        members = []
        if resampling:
            samples, settings = sampler.prepare_samples(train)
            template = templates[0]
            fits = (
                (template, slice_model_data(train, indices), _member_params(
                    template.fitted_model.params, template.fitted_model.framework, member_seed,
                )) for member_seed, indices in samples
            )
        else:
            settings = {"method": "blending", "blend_mode": "fixed"}
            fits = ((template, train, template.fitted_model.params) for template in templates)
        for number, (template, sample, params) in enumerate(fits, start=1):
            try:
                raw = sample.select_features(template.input_feature_names)
                model, chain = _fit_fold(template, raw, params=params)
                members.append(replace(
                    template, fitted_model=model, encoder=chain.encoder,
                    preprocessors=chain.preprocessors, raw_train_data=None,
                    selected_features=chain.selected_features,
                    model_selected_features=chain.model_selected_features,
                    selection_results=None, tuning_history=None, input_schema=raw.schema,
                ))
            except Exception as exc:
                raise RuntimeError(f"fold {fold}, member {number} failed: {exc}") from exc
        fitted = EnsembleResult(FittedBlendingEnsemble(weights, members), members)
        report = ensemble_permutation_importance(
            fitted, held, n_repeats=n_repeats, max_samples=max_samples,
            seed=int(shuffle_seed.generate_state(1)[0]), show_progress_bar=False,
        )
        baseline_rows.append({
            "fold": fold, "n_train": train.n_rows, "n_held": held.n_rows,
            "n_evaluated": report.settings["n_evaluated"],
            "baseline_score": report.baseline_score,
        })
        repeat_frames.append(report.repeat_importances.with_columns(pl.lit(fold).alias("fold")))
        fold_settings.append({
            "fold": fold, **settings, "weights": list(weights),
            "members": [deepcopy(member.fitted_model.params) for member in members],
        })
        # Release fitted members and fold-local matrices before the next fold.
        del fitted, members, model, chain, raw, sample, fits, train, held, report

    repeats = pl.concat(repeat_frames).select(
        "fold", "feature", "repeat", "baseline_score", "permuted_score", "importance",
    )
    folds, summary = _summarize_cv(repeats)
    predefined = config.folds == "predefined" or (
        config.folds == "auto" and clean.cv_fold is not None
    )
    return PermutationImportanceResult(
        summary=summary, fold_importances=folds, repeat_importances=repeats,
        baseline_scores=pl.DataFrame(baseline_rows), settings={
            "metric": f"{objective}_deviance", "objective": objective,
            "n_repeats": n_repeats, "max_samples": max_samples, "seed": seed,
            "cv": {"folds": "predefined" if predefined else "random",
                   "n_splits": len(splits), "seed": None if predefined else config.seed},
            "feature_names": features, "method": candidate.method,
            "ensemble_seed": candidate.seed if resampling else None,
            "fold_ensembles": fold_settings,
            "selected_features": [deepcopy(p.selected_features) for p in templates],
            "model_selected_features": [deepcopy(p.model_selected_features) for p in templates],
        },
    )


def _summarize_cv(repeats):
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
    return folds, summary


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
    using only training rows. Only raw inputs feeding final model columns
    are shuffled and reported. A frozen
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
    _validate_options(n_repeats, max_samples, seed)
    objective = fitted.fitted_model.objective
    if data.objective is not None and data.objective != objective:
        raise ValueError("data objective conflicts with fitted objective")
    clean = replace(data, objective=objective).select_features(fitted.input_feature_names)
    features = _model_raw_features(fitted)
    if not features:
        raise ValueError("Permutation importance requires at least one raw model feature")
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
        for feature in features:
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
    folds, summary = _summarize_cv(repeats)
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
            "feature_names": features,
            "selected_features": deepcopy(fitted.selected_features),
            "model_selected_features": deepcopy(fitted.model_selected_features),
            "model_params": deepcopy(fitted.fitted_model.params),
            "framework": fitted.fitted_model.framework,
        },
    )
