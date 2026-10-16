"""Outer CV for model and ensemble candidates, with fold-local fit reuse."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import pickle

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from ins_gbm.data.dtypes import FIT_DTYPE
from ins_gbm.data.folds import CVConfig, resolve_folds
from ins_gbm.data.model_data import ModelData, slice_model_data
from ins_gbm.data.selection_candidates import select_raw_candidates
from ins_gbm.ensemble._utils import _refit_pipeline, _validate_ensemble_pipelines
from ins_gbm.ensemble.blending import BlendingEnsemble
from ins_gbm.ensemble.pipeline import EnsemblePipeline
from ins_gbm.ensemble.resampling import ResamplingEnsemble, _member_params
from ins_gbm.ensemble.stacking import StackingEnsemble
from ins_gbm.evaluation.cv_report import CrossValidationReport, CVResult, _assemble_cv_result
from ins_gbm.models.base import resolve_objective
from ins_gbm.pipeline import FittedPipeline, ModelPipeline, ModelRecipe


def _freeze(value):
    """Fingerprint built-in configuration only; custom objects bypass sharing."""
    if value is None or type(value) in (bool, int, float, str, bytes):
        return (type(value).__name__, value)
    if isinstance(value, np.generic):
        return _freeze(value.item())
    if type(value) in (list, tuple):
        return (type(value).__name__, tuple(_freeze(item) for item in value))
    if type(value) is dict:
        return ("dict", tuple(sorted((_freeze(k), _freeze(v)) for k, v in value.items())))
    if isinstance(value, np.ndarray) and value.dtype.kind != "O":
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if is_dataclass(value) and type(value).__module__.startswith("ins_gbm."):
        return (
            type(value).__module__, type(value).__qualname__,
            tuple((f.name, _freeze(getattr(value, f.name))) for f in fields(value)),
        )
    raise TypeError("custom configuration cannot be safely shared")


@dataclass
class _FitSpec:
    recipe: ModelRecipe
    input_names: list[str]
    params: dict | None
    template: FittedPipeline | None = None
    retune: bool = False

    @property
    def encoded(self):
        if self.template is not None and self.recipe.selection is None:
            return self.template.selected_features
        return None

    @property
    def model_selected(self):
        return self.template.model_selected_features if self.template is not None else None

    def cache_key(self, rows):
        if self.retune and self.recipe.tuning is not None:
            return None
        try:
            configuration = _freeze((
                self.recipe.model, self.recipe.encoder, self.recipe.selection,
                self.recipe.preprocessing, self.recipe.selection_scope,
                self.input_names, self.encoded, self.model_selected, self.params or {},
            ))
        except TypeError:
            return None
        digest = hashlib.sha256(pickle.dumps(configuration)).digest()
        return digest, hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).digest()

    def fit(self, data):
        data = data.select_features(self.input_names)
        if self.template is not None and not (self.retune and self.recipe.tuning is not None):
            return _refit_pipeline(self.template, data, self.params or {})
        recipe = deepcopy(self.recipe)
        if not self.retune:
            recipe = replace(recipe, tuning=None, params=deepcopy(self.params))
        options = {}
        if self.model_selected is not None:
            # A configured selector plus a fixed model subset must both survive.
            if recipe.selection is not None or self.encoded is not None:
                return self._tune_with_fixed_stages(data, recipe)
            options = {"feature_names": self.model_selected, "feature_stage": "model"}
        elif self.encoded is not None:
            options = {"feature_names": self.encoded, "feature_stage": "encoded"}
        return ModelPipeline(data, recipe).run(**options)

    def _tune_with_fixed_stages(self, data, recipe):
        # Use the tuner's fold-local transform support for the combination that
        # ModelPipeline.run's single feature_stage argument cannot express.
        selector = recipe.selection
        if self.encoded is not None:
            selector = _FixedSelection(list(self.encoded))
        params, history = recipe.tuning.tune(
            data, recipe.model, encoder=recipe.encoder, selector=selector,
            preprocessors=recipe.preprocessing, base_params=recipe.params,
            model_selected_features=self.model_selected,
        )
        fitted = _refit_pipeline(self.template, data, params)
        return replace(fitted, recipe=recipe, tuning_history=history)


@dataclass
class _FixedSelection:
    names: list[str]

    def fit(self, data):
        data.features.select(self.names)
        return self

    def selected_features(self):
        return list(self.names)


def _template_spec(template, data, refit):
    names = list(template.input_feature_names)
    data.select_features(names)  # Fail missing raw columns before training.
    return _FitSpec(
        template.recipe, names, dict(template.fitted_model.params), template,
        retune=refit == "retune",
    )


def _no_importance(*args, **kwargs):
    return pl.DataFrame({"feature": pl.Series([], dtype=pl.String), "importance": pl.Series([], dtype=pl.Float64)})


class _FitContext:
    """Cache fitted pipelines and response predictions for one outer fold only."""

    def __init__(self, data, counters):
        self.data = data
        self.counters = counters
        self.fits = {}
        self.predictions = {}

    def fit(self, spec, rows):
        key = spec.cache_key(rows)
        if key is not None and key in self.fits:
            self.counters["fit_cache_hits"] += 1
            return self.fits[key]
        fitted = spec.fit(slice_model_data(self.data, rows))
        self.counters["pipeline_fits"] += 1
        if spec.retune and spec.recipe.tuning is not None:
            self.counters["tuned_pipeline_fits"] += 1
        # These models only serve predictions. Drop training rows and unused
        # diagnostics immediately, including CatBoost's deferred importance Pool.
        compact = replace(
            fitted, raw_train_data=None, selection_results=None, tuning_history=None,
            fitted_model=replace(fitted.fitted_model, importance_fn=_no_importance),
        )
        entry = (object(), compact)
        if key is not None:
            self.fits[key] = entry
            # Native wrappers add default params. This alias allows a freshly
            # fitted regular recipe to serve a template with those resolved params.
            resolved = replace(spec, params=fitted.fitted_model.params).cache_key(rows)
            if resolved is not None:
                self.fits.setdefault(resolved, entry)
        return entry

    def predict(self, entry, rows):
        token, fitted = entry
        key = token, hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).digest()
        if key in self.predictions:
            self.counters["prediction_cache_hits"] += 1
            return self.predictions[key]
        predictions = fitted.predict(slice_model_data(self.data, rows), "response").to_numpy()
        if not np.all(np.isfinite(predictions)):
            raise ValueError("CV response predictions must be finite")
        self.predictions[key] = predictions
        return predictions


def _inner_config(candidate, override):
    if override is not None:
        return override
    if candidate.cv is not None and candidate.use_data_folds is not None:
        raise ValueError("Pass either cv or use_data_folds, not both")
    return candidate.cv or CVConfig(
        candidate.cv_folds, candidate.seed,
        "auto" if candidate.use_data_folds is None
        else "predefined" if candidate.use_data_folds else "random",
    )


def _validation_split(data, config, fraction, seed):
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if config.folds not in {"auto", "random", "predefined"}:
        raise ValueError("folds must be 'auto', 'random', or 'predefined'")
    grouped = config.folds == "predefined" or (config.folds == "auto" and data.cv_fold is not None)
    if grouped:
        # Resolver validates that at least two nonempty groups remain.
        resolve_folds(data, config)
        from sklearn.model_selection import GroupShuffleSplit
        return next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(
            np.arange(data.n_rows), groups=data.cv_fold.to_numpy(),
        ))
    from sklearn.model_selection import train_test_split
    return train_test_split(np.arange(data.n_rows), test_size=fraction, random_state=seed)


def _prepare_candidate(candidate, data, outer_rows, base_refit, inner_cv, fraction, fold_number):
    """Resolve every required split/sample before any outer model fitting."""
    if isinstance(candidate, ModelRecipe):
        names = select_raw_candidates(data, candidate.selection).feature_names
        return _FitSpec(candidate, list(names), candidate.params, retune=candidate.tuning is not None)
    if not isinstance(candidate, EnsemblePipeline):
        raise TypeError("candidates must be ModelRecipe or EnsemblePipeline configurations")
    _validate_ensemble_pipelines(candidate.fitted_pipelines, require_training=False)
    if candidate.refit not in {"fixed", "retune"}:
        raise ValueError("refit must be 'fixed' or 'retune'")
    specs = [_template_spec(p, data, base_refit) for p in candidate.fitted_pipelines]
    train = slice_model_data(data, outer_rows)
    plan = {"specs": specs, "train_rows": outer_rows}
    if candidate.method in {"seed_average", "bagging"}:
        if len(specs) != 1:
            raise ValueError(f"{candidate.method} requires exactly one fitted template pipeline")
        samples, settings = ResamplingEnsemble(
            candidate.method, candidate.n_members, candidate.seed,
            candidate.sample_fraction, candidate.bootstrap,
        ).prepare_samples(train)
        _member_params(specs[0].params, candidate.fitted_pipelines[0].fitted_model.framework, settings["member_seeds"][0])
        plan.update(samples=samples, settings=settings)
    elif candidate.method == "stacking" or (
        candidate.method == "blending" and candidate.blend_mode == "oof"
    ):
        inner = _inner_config(candidate, inner_cv)
        labels, splits = resolve_folds(train, inner)
        plan["splits"] = splits
        plan["inner_cv"] = {
            "folds": inner.folds, "n_splits": len(splits),
            "seed": inner.seed, "fold_ids": labels,
        }
    elif candidate.method == "blending" and candidate.blend_mode == "validation":
        if candidate.validation_data is not None:
            raise ValueError("Ensemble CV rejects external validation_data; use its internal validation split")
        inner = _inner_config(candidate, inner_cv)
        fitting, validation = _validation_split(
            train, inner, fraction, (inner.seed + fold_number) % (2**32),
        )
        plan.update(
            train_rows=outer_rows[fitting], validation_rows=outer_rows[validation],
            validation_fraction=fraction,
        )
    elif candidate.method != "blending" or candidate.blend_mode != "fixed":
        raise ValueError("Unknown ensemble method or blending mode")
    return plan


def _matrix(context, entries, rows):
    return np.stack([context.predict(entry, rows) for entry in entries], axis=1).astype(FIT_DTYPE)


def _evaluate_ensemble(candidate, plan, context, held_rows):
    specs = plan["specs"]
    train_rows = plan["train_rows"]
    if candidate.method in {"seed_average", "bagging"}:
        spec = specs[0]
        params = spec.params
        if spec.retune and spec.recipe.tuning is not None:
            # Configured tuning selects parameters once per outer training fold;
            # member fits themselves never repeat tuning.
            params = context.fit(spec, train_rows)[1].fitted_model.params
        framework = candidate.fitted_pipelines[0].fitted_model.framework
        entries = []
        for number, (seed, sample) in enumerate(plan["samples"], start=1):
            try:
                member = replace(spec, params=_member_params(params, framework, seed), retune=False)
                entries.append(context.fit(member, train_rows[sample]))
            except Exception as exc:
                raise RuntimeError(f"member {number} (seed={seed}) failed: {exc}") from exc
        weights = np.full(len(entries), 1 / len(entries), dtype=FIT_DTYPE)
        return _matrix(context, entries, held_rows) @ weights, {
            **plan["settings"], "members": [entry[1].fitted_model.params for entry in entries],
        }

    entries = [context.fit(spec, train_rows) for spec in specs]
    fitted = [entry[1] for entry in entries]
    predictions = _matrix(context, entries, held_rows)
    record = {
        "method": candidate.method,
        "members": [p.fitted_model.params for p in fitted],
        "base_refit": "retune" if any(s.retune for s in specs) else "fixed",
        "refit": candidate.refit,
    }
    if "inner_cv" in plan:
        record["inner_cv"] = plan["inner_cv"]
    if "validation_fraction" in plan:
        record["validation_fraction"] = plan["validation_fraction"]
    blender = BlendingEnsemble(mode=candidate.blend_mode, weights=candidate.blend_weights, metric=candidate.metric)
    if candidate.method == "blending" and candidate.blend_mode == "fixed":
        weights = blender._fit_fixed(fitted).weights
    elif candidate.method == "blending" and candidate.blend_mode == "validation":
        validation_rows = plan["validation_rows"]
        weights = blender._learn_weights(
            _matrix(context, entries, validation_rows),
            slice_model_data(context.data, validation_rows), fitted,
        )
    else:
        oof = np.empty((len(train_rows), len(specs)), dtype=FIT_DTYPE)
        for column, (spec, entry) in enumerate(zip(specs, entries)):
            inner_spec = replace(
                spec, params=entry[1].fitted_model.params,
                retune=candidate.refit == "retune",
            )
            for inner_train, inner_held in plan["splits"]:
                inner_entry = context.fit(inner_spec, train_rows[inner_train])
                oof[inner_held, column] = context.predict(inner_entry, train_rows[inner_held])
        train = slice_model_data(context.data, train_rows)
        if candidate.method == "stacking":
            stacker = StackingEnsemble(meta_learner=candidate.meta_learner, meta_alpha=candidate.meta_alpha)
            stack = stacker.fit_from_predictions(fitted, oof, train)
            meta = stack.meta_learner
            record.update(
                meta_learner=type(meta).__name__, meta_alpha=candidate.meta_alpha,
                distribution_aware=stack.distribution_aware,
            )
            if type(meta).__module__.startswith("sklearn.") and hasattr(meta, "coef_"):
                record.update(
                    meta_coef=np.asarray(meta.coef_).tolist(),
                    meta_intercept=np.asarray(meta.intercept_).tolist(),
                )
            return stack.predict_from_predictions(
                predictions, slice_model_data(context.data, held_rows)
            ).to_numpy(), record
        weights = blender._learn_weights(oof, train, fitted)
    record.update(blend_mode=candidate.blend_mode, metric=candidate.metric, weights=weights)
    return predictions @ np.asarray(weights, dtype=FIT_DTYPE), record


def cross_validate_candidates(
    candidates: dict[str, ModelRecipe | EnsemblePipeline],
    data: ModelData,
    *,
    cv: CVConfig | None = None,
    base_refit: str = "fixed",
    inner_cv: CVConfig | None = None,
    validation_fraction: float = 0.2,
    show_progress_bar: bool = True,
) -> dict[str, CVResult]:
    """Compare models/ensembles on shared outer folds without using a holdout.

    Matching fixed-parameter base fits and predictions are cached within each
    outer fold. Each ensemble fits its own weights or meta-learner on inner
    training data. Custom configurations and hyperparameter searches bypass
    fit sharing. Regular ModelRecipes retain their own tuning configuration.

    Poisson targets remain counts and OOF predictions are expected counts;
    exposure-weighted rate deviance is calculated by the standard report.
    Gamma predictions are severity. ``base_refit`` controls outer ensemble
    templates; the candidate's ``refit`` controls its inner fits. ``inner_cv``
    overrides candidate inner-fold settings, independently of outer ``cv``.
    Validation blending uses an internal split and rejects external blend data.

    Result cv_config['execution'] records pipeline_fits (final pipeline fits,
    excluding individual tuner trials/folds), tuned_pipeline_fits, and cache
    hits for the entire invocation. No models or caches are saved in results.
    """
    return _cross_validate_candidates(
        candidates, data, cv=cv, base_refit=base_refit, inner_cv=inner_cv,
        validation_fraction=validation_fraction, show_progress_bar=show_progress_bar,
    )


def _cross_validate_candidates(
    candidates, data, *, cv=None, base_refit="fixed", inner_cv=None,
    validation_fraction=0.2, show_progress_bar=True, report=None,
):
    if not candidates or not isinstance(candidates, dict):
        raise ValueError("candidates must be a nonempty mapping of names to configurations")
    if any(not isinstance(name, str) or not name for name in candidates):
        raise ValueError("candidate names must be nonempty strings")
    if base_refit not in {"fixed", "retune"}:
        raise ValueError("base_refit must be 'fixed' or 'retune'")
    objectives = set()
    for candidate in candidates.values():
        if isinstance(candidate, ModelRecipe):
            objectives.add(resolve_objective(getattr(candidate.model, "objective", None), data))
        elif isinstance(candidate, EnsemblePipeline):
            _validate_ensemble_pipelines(candidate.fitted_pipelines, require_training=False)
            objective = candidate.fitted_pipelines[0].fitted_model.objective
            resolve_objective(objective, data)
            objectives.add(objective)
            if candidate.validation_data is not None:
                raise ValueError("Ensemble CV rejects external validation_data; use its internal validation split")
        else:
            raise TypeError("candidates must be ModelRecipe or EnsemblePipeline configurations")
    if len(objectives) != 1:
        raise ValueError("all candidates must use the same objective")
    data = replace(data, objective=objectives.pop())
    data.validate()
    if report is None:
        report = CrossValidationReport(
            next(iter(candidates.values())), data,
            cv=cv or CVConfig(), show_progress_bar=show_progress_bar,
        )
    else:
        report = replace(report, data=data)
    report._validate()
    clean, fold_ids, benchmark, _ = report._prepare_data(None, "raw")
    config = report.cv or CVConfig(
        report.n_folds, report.seed, "predefined" if fold_ids is not None else "random",
    )
    fold_labels, splits = resolve_folds(clean, config)
    predefined = config.folds == "predefined" or (config.folds == "auto" and fold_ids is not None)
    # Preflight every requested split and sample before fitting any candidates.
    for fold_number, (label, (train, _)) in enumerate(zip(fold_labels, splits)):
        for name, candidate in candidates.items():
            try:
                _prepare_candidate(
                    candidate, clean, train, base_refit, inner_cv,
                    validation_fraction, fold_number,
                )
            except Exception as exc:
                raise ValueError(f"Candidate {name!r}, outer fold {label!r}: {exc}") from exc
    counters = dict(pipeline_fits=0, tuned_pipeline_fits=0, fit_cache_hits=0, prediction_cache_hits=0)
    outputs = {
        name: dict(rows=[], predictions=np.full(clean.n_rows, np.nan), params={})
        for name in candidates
    }
    row_folds = [None] * clean.n_rows
    # Fit regular recipes first, allowing their resolved native params to serve
    # matching ensemble templates regardless of dictionary insertion order.
    order = sorted(candidates, key=lambda name: isinstance(candidates[name], EnsemblePipeline))
    progress = tqdm(
        enumerate(zip(fold_labels, splits)), total=len(splits), desc="Cross-validation",
        unit="fold", disable=not report.show_progress_bar,
    )
    for fold_number, (label, (train, held)) in progress:
        label = label.item() if isinstance(label, np.generic) else label
        context = _FitContext(clean, counters)
        for index in held:
            row_folds[int(index)] = label
        for name in order:
            try:
                candidate = candidates[name]
                plan = _prepare_candidate(
                    candidate, clean, train, base_refit, inner_cv,
                    validation_fraction, fold_number,
                )
                if isinstance(plan, _FitSpec):
                    entry = context.fit(plan, train)
                    predicted = context.predict(entry, held)
                    params = entry[1].fitted_model.params
                else:
                    predicted, params = _evaluate_ensemble(candidate, plan, context, held)
                if not np.all(np.isfinite(predicted)):
                    raise ValueError("CV response predictions must be finite")
                output = outputs[name]
                output["predictions"][held] = predicted
                output["params"][label] = deepcopy(params)
                output["rows"].extend(report._fold_metrics(
                    label, clean.objective, slice_model_data(clean, held),
                    pl.Series(predicted), benchmark, held,
                ))
            except Exception as exc:
                raise RuntimeError(f"Candidate {name!r}, outer fold {label!r}: {exc}") from exc
        # No estimator or prediction cache survives the outer fold.
        context.fits.clear()
        context.predictions.clear()
    cv_config = {
        "folds": "predefined" if predefined else "random", "n_splits": len(splits),
        "seed": None if predefined else config.seed, "execution": counters,
    }
    return {
        name: _assemble_cv_result(
            data=clean, rows=output["rows"], predictions=output["predictions"],
            row_folds=row_folds, fold_params=output["params"], benchmark=benchmark,
            fold_col=(report.fold_col or "cv_fold") if predefined else None,
            feature_names=list(clean.feature_names), cv_config=deepcopy(cv_config),
        )
        for name, output in outputs.items()
    }
