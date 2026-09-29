"""Budgeted four-family model search; see model_search_template.md for setup.

    python examples/model_search_template.py search --config examples/model_search_config.json
    python examples/model_search_template.py evaluate --config examples/model_search_config.json

Only `evaluate` reads the final test file. No improvement is guaranteed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import itertools
import json
from pathlib import Path

import numpy as np
import optuna
import polars as pl
from scipy.optimize import minimize
from scipy.special import xlogy

from ins_gbm import (
    CVConfig, CatBoostModel, HyperparameterTuner, LightGBMModel, ModelRecipe,
    OneHotEncoder, RandomForestModel, XGBoostModel, load_model, load_model_data,
)
from ins_gbm.data.model_data import slice_model_data
from ins_gbm.evaluation.metrics import compute_metrics


FAMILIES = {
    "lightgbm": LightGBMModel, "xgboost": XGBoostModel,
    "catboost": CatBoostModel, "random_forest": RandomForestModel,
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


@dataclass
class SearchModel:
    """Override search ranges without changing library source."""
    family: str
    trees: list[int]

    @property
    def objective(self):
        return None  # Resolve Poisson/Gamma from ModelData.

    def capabilities(self):
        return FAMILIES[self.family]().capabilities()

    def fit(self, data, params=None):
        return FAMILIES[self.family]().fit(data, params=params)

    def default_search_space(self):
        space = FAMILIES[self.family]().default_search_space()
        integer, real = optuna.distributions.IntDistribution, optuna.distributions.FloatDistribution
        tree_key = "iterations" if self.family == "catboost" else "n_estimators"
        space[tree_key] = integer(*self.trees)
        if self.family == "lightgbm":
            space.update(num_leaves=integer(15, 127), min_child_samples=integer(50, 500))
        elif self.family == "xgboost":
            space.update(max_depth=integer(3, 8), min_child_weight=real(1, 100, log=True))
        elif self.family == "catboost":
            space.update(depth=integer(4, 8))
        else:
            space.update(max_depth=integer(6, 14), min_samples_leaf=integer(20, 300),
                         max_features=real(0.1, 0.6))
        return space


@dataclass
class SelectedColumns:
    columns: list[str]

    def selected_features(self):
        return self.columns


@dataclass
class ProtectedRawSelector:
    """Rank raw groups by summed gain, retaining ALL existing raw features.

    Called after encoding, within each training fold. All one-hot levels of a
    selected categorical feature are retained. Caps apply to NEW raw features.
    This is a screening heuristic; high-cardinality groups can attract more gain.
    The unpruned full-pool candidate is always evaluated too.
    """
    raw_names: list[str]
    categorical: list[str]
    protected: list[str]
    caps: list[int]
    threads: int
    seed: int
    trees: int = 100

    def owner(self, encoded):
        matches = [raw for raw in self.raw_names if
                   (raw not in self.categorical and encoded == raw)
                   or (raw in self.categorical and encoded.startswith(raw + "__"))]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous encoded name {encoded!r}; rename raw columns to avoid '__' collisions")
        return matches[0]

    def fit(self, data):
        owners = {name: self.owner(name) for name in data.feature_names}
        current = data
        for cap in self.caps:
            fitted = LightGBMModel().fit(current, params={
                "n_estimators": self.trees, "num_leaves": 31, "min_child_samples": 50,
                "num_threads": self.threads, "seed": self.seed, "verbose": -1,
            })
            scores = dict.fromkeys(self.raw_names, 0.0)
            for name, gain in fitted.feature_importance("gain").iter_rows():
                scores[owners[name]] += gain
            available = {owners[name] for name in current.feature_names}
            extras = [name for name in self.raw_names if name in available and name not in self.protected]
            extras.sort(key=lambda name: -scores[name])
            keep = set(self.protected) | set(extras[:cap])
            columns = [name for name in current.feature_names if owners[name] in keep]
            current = current.with_features(current.features.select(columns))
        return SelectedColumns(list(current.feature_names))


def base_params(family, seed, threads):
    if family == "lightgbm":
        # subsample needs a positive bagging frequency to have an effect.
        return {"seed": seed, "num_threads": threads, "verbose": -1, "bagging_freq": 1}
    if family == "xgboost":
        return {"seed": seed, "nthread": threads, "tree_method": "hist", "verbosity": 0}
    if family == "catboost":
        return {"random_seed": seed, "thread_count": threads, "verbose": False,
                "allow_writing_files": False, "bootstrap_type": "Bernoulli"}
    return {"random_state": seed, "n_jobs": threads}


def make_recipe(config, candidate, training, trials, seed, *, screening=False):
    selector = None
    if candidate["strategy"] == "protected_selection":
        selector = ProtectedRawSelector(
            raw_names=config["candidate_features"], categorical=training.schema.categorical,
            protected=config["existing_features"], caps=config["selection_new_feature_caps"],
            threads=config["threads_per_model"], seed=seed,
            trees=config.get("selection_trees", 100),
        )
    params = base_params(candidate["family"], seed, config["threads_per_model"])
    if candidate["strategy"] == "known_baseline":
        params.update(config["known_baseline_params"])
        trials = 0
    trees = config["screen_tree_range" if screening else "final_tree_range"]
    return ModelRecipe(
        model=SearchModel(candidate["family"], trees), encoder=OneHotEncoder(),
        selection=selector, selection_scope="fold", params=params,
        tuning=HyperparameterTuner(
            n_trials=trials, cv=CVConfig(n_splits=config["inner_folds"], seed=config["fold_seed"], folds="auto"),
            seed=seed, metric=None, n_jobs=1, show_progress_bar=False,
        ) if trials else None,
    )


def load_data(config, split):
    data = load_model_data(
        config[f"{split}_path"], target=config["target"], objective=config["objective"],
        exposure=config.get("exposure"), weight=config.get("weight"),
        feature_cols=config["candidate_features"],
        cv_fold=config.get("fold_column") if split == "train" else None,
    )
    # Integer-coded categories must be explicitly identified. String categories
    # are inferred. Casting is row-local and learns nothing from held-out rows.
    if config.get("categorical_features"):
        data = data.with_features(data.features.with_columns(
            pl.col(name).cast(pl.String) for name in config["categorical_features"]
        ))
        from ins_gbm.data.schema import infer_schema
        data.schema = infer_schema(data.features, data.feature_names)
    return data


def deviance(data, predictions):
    """Same pooled deviance as the library, without computing/sorting Gini."""
    y = data.target.to_numpy().astype(float)
    mu = np.asarray(predictions, dtype=float)
    if mu.shape != y.shape or not np.isfinite(mu).all() or np.any(mu <= 0):
        raise ValueError("Expected one finite, positive response prediction per row")
    w = np.ones(len(y)) if data.weight is None else data.weight.to_numpy()
    if data.objective == "poisson":
        denominator = np.sum(w if data.exposure is None else w * data.exposure.to_numpy())
        return float(np.sum(2 * w * (xlogy(y, y / mu) - y + mu)) / denominator)
    return float(np.sum(2 * w * (-np.log(y / mu) + y / mu - 1)) / np.sum(w))


def candidate_list(config):
    strategies = {"existing": config["existing_features"], "full": config["candidate_features"]}
    if config["selection_new_feature_caps"]:
        strategies["protected_selection"] = config["candidate_features"]
    for label, features in config.get("additional_feature_groups", {}).items():
        strategies[f"add_{label}"] = list(dict.fromkeys(config["existing_features"] + features))
    candidates = [dict(name=f"{family}_{strategy}", family=family, strategy=strategy, features=features)
                  for family in FAMILIES for strategy, features in strategies.items()]
    if config.get("known_baseline_params"):
        candidates.append(dict(name="known_baseline", family=config["baseline_family"],
                               strategy="known_baseline", features=config["existing_features"]))
    return candidates


def blend_options(data, predictions):
    """All pairs + all-model simplex, equal weights, and every single model.

    Fit nonnegative response-scale weights against the selected deviance, using
    multiple starts. This searches these blend forms, not every possible ensemble.
    Gamma deviance is not globally convex in predictions; no global guarantee.
    """
    n = predictions.shape[1]
    options = [(f"single_{i}", row) for i, row in enumerate(np.eye(n))]
    subsets = list(itertools.combinations(range(n), 2))
    if n > 2:
        subsets.append(tuple(range(n)))
    for subset in subsets:
        matrix = predictions[:, subset]
        starts = [np.full(len(subset), 1 / len(subset)), *np.eye(len(subset))]
        solutions = list(starts)
        for start in starts:
            result = minimize(lambda w: deviance(data, matrix @ w), start,
                              method="SLSQP", bounds=[(0, 1)] * len(subset),
                              constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                              options={"maxiter": 300, "ftol": 1e-10})
            if result.success and np.isfinite(result.x).all():
                weights = np.maximum(result.x, 0)
                solutions.append(weights / weights.sum())
        best = min(solutions, key=lambda w: deviance(data, matrix @ w))
        for label, weights in [("optimized", best), ("equal", starts[0])]:
            full = np.zeros(n)
            full[list(subset)] = weights
            options.append((f"{label}_{'_'.join(map(str, subset))}", full))
    return options


def search(config):
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite a completed experiment.
    write_json(out / "config.json", config)
    training, validation = load_data(config, "train"), load_data(config, "validation")
    if training.cv_fold is not None and training.cv_fold.n_unique() < 3:
        raise ValueError("Nested tuning with predefined folds needs at least 3 fold IDs")
    encoder = OneHotEncoder().fit(training.features, training.schema)
    width = len(encoder.output_feature_names())
    gb = training.n_rows * width * 4 / 1e9
    print(f"{training.n_rows:,} training rows; {len(training.feature_names):,} raw / {width:,} encoded columns.")
    print(f"One dense float32 matrix alone: {gb:.2f} GB; actual peak memory is several times larger.")
    if gb > config.get("max_dense_matrix_gb", 4):
        raise ValueError("Encoded matrix exceeds configured budget; review high-cardinality columns or raise limit")
    del encoder
    seed = config["model_seeds"][0]
    # Same training-only row sample for all screening candidates. Never use this
    # sample to report final performance. Group folds remain attached to rows.
    size = min(config["screen_rows"], training.n_rows)
    indices = np.sort(np.random.default_rng(config["fold_seed"]).choice(training.n_rows, size, replace=False))
    sample = slice_model_data(training, indices)
    candidates = candidate_list(config)
    scores = {}
    for candidate in candidates:
        print("Screening", candidate["name"], flush=True)
        recipe = make_recipe(config, candidate, sample, config["screen_trials"], seed, screening=True)
        fitted = recipe.fit(sample, feature_names=candidate["features"])
        scores[candidate["name"]] = deviance(validation, fitted.predict(validation).to_numpy())
        pl.DataFrame([{"candidate": name, "validation_deviance": score} for name, score in scores.items()]).write_csv(out / "screening.csv")
        del fitted
        gc.collect()
    del sample

    # Give EVERY family a full-data finalist, even if its screening result loses.
    finalists = []
    for family in FAMILIES:
        family_candidates = [c for c in candidates if c["family"] == family and c["strategy"] != "known_baseline"]
        finalists.extend(sorted(family_candidates, key=lambda c: scores[c["name"]])[:config["finalists_per_family"]])
    # Always refine the baseline's existing feature set and retain known params.
    finalists.extend(c for c in candidates if c["name"] == f"{config['baseline_family']}_existing" or c["strategy"] == "known_baseline")
    finalists = list({c["name"]: c for c in finalists}.values())
    members, validation_predictions, oof_predictions, refined = [], [], [], []
    outer_cv = CVConfig(n_splits=config["outer_folds"], seed=config["fold_seed"], folds="auto")
    for candidate in finalists:
        for model_seed in config["model_seeds"]:
            name = f"{candidate['name']}_seed{model_seed}"
            print("Full-data tuning and nested OOF:", name, flush=True)
            recipe = make_recipe(config, candidate, training, config["final_trials"], model_seed)
            fitted = recipe.fit(training, feature_names=candidate["features"])
            prediction = fitted.predict(validation).to_numpy()
            folder = out / "models" / name
            fitted.save(str(folder))
            write_json(folder / "selected_encoded_features.json", fitted.selected_features or fitted.fitted_model.feature_names)
            refined.append({"candidate": name, "validation_deviance": deviance(validation, prediction)})
            del fitted
            gc.collect()
            # Fresh tuning AND supervised selection in each outer training fold.
            # Same fold assignments for every member. No globally chosen params
            # are reused to generate the OOF predictions used for blending.
            result = recipe.cross_validate(training, cv=outer_cv, feature_names=candidate["features"])
            result.save(str(folder / "cv_report"))
            result.fold_metrics.write_csv(folder / "outer_fold_metrics.csv")
            result.summary.write_csv(folder / "outer_cv_summary.csv")
            oof = result.predictions["gbm"].to_numpy()
            pl.DataFrame({"oof": oof}).write_parquet(folder / "oof.parquet")
            members.append({"name": name, "candidate": candidate, "path": str(folder.resolve())})
            validation_predictions.append(prediction)
            oof_predictions.append(oof)
            pl.DataFrame(refined).write_csv(out / "finalists.csv")
            del result
            gc.collect()

    validation_matrix = np.column_stack(validation_predictions)
    oof_matrix = np.column_stack(oof_predictions)
    options = blend_options(training, oof_matrix)
    rows = [{"name": name, "oof_deviance": deviance(training, oof_matrix @ weights),
             "validation_deviance": deviance(validation, validation_matrix @ weights),
             "weights": weights.tolist()} for name, weights in options]
    # OOF scores here TRAIN the blend; they are not independent blend test scores.
    winner = min(rows, key=lambda row: row["validation_deviance"])
    blends = [row for row in rows if np.count_nonzero(np.array(row["weights"]) > 1e-8) > 1]
    best_blend = min(blends, key=lambda row: row["validation_deviance"]) if blends else winner
    baseline_indices = [i for i, member in enumerate(members) if
                        member["candidate"]["family"] == config["baseline_family"] and
                        member["candidate"]["strategy"] in {"existing", "known_baseline"}]
    baseline_index = min(baseline_indices, key=lambda i: deviance(validation, validation_matrix[:, i]))
    baseline_weights = np.eye(len(members))[baseline_index].tolist()
    pl.DataFrame([{k: v for k, v in row.items() if k != "weights"} for row in rows]).write_csv(out / "blend_search.csv")
    manifest = {"members": members, "winner": winner, "best_blend": best_blend,
                "baseline_weights": baseline_weights, "blend_options": rows}
    write_json(out / "selection.json", manifest)
    print("Locked validation winner:", winner)
    print("Best genuine blend:", best_blend)
    print("The test file has NOT been read. Run the evaluate command once when ready.")


def evaluate(config):
    out = Path(config["output_dir"])
    # Use the SEARCH configuration, not subsequent edits to modeling settings.
    saved_config = json.loads((out / "config.json").read_text())
    manifest = json.loads((out / "selection.json").read_text())
    test = load_data(saved_config, "test")
    predictions = []
    for member in manifest["members"]:
        fitted = load_model(member["path"])
        predictions.append(fitted.predict(test).to_numpy())
        del fitted
        gc.collect()
    matrix = np.column_stack(predictions)
    choices = {"baseline": manifest["baseline_weights"], "locked_winner": manifest["winner"]["weights"],
               "locked_best_blend": manifest["best_blend"]["weights"]}
    metrics, exported = [], {}
    for label, weights in choices.items():
        predicted = pl.Series(matrix @ np.array(weights))
        exported[label] = predicted
        metrics.append(compute_metrics(objective=test.objective, actual=test.target, predicted=predicted,
                                       exposure=test.exposure, weight=test.weight).with_columns(pl.lit(label).alias("model")))
    report = pl.concat(metrics).select("model", "metric", "value")
    report.write_csv(out / "test_metrics.csv")
    pl.DataFrame(exported).write_parquet(out / "test_predictions.parquet")
    print(report)
    print("Positive relative deviance improvement favors the locked winner:",
          1 - deviance(test, exported["locked_winner"].to_numpy()) / max(deviance(test, exported["baseline"].to_numpy()), 1e-15))
    print("Do not choose another model or blend using these test results.")


def validate_config(config):
    existing, pool = config["existing_features"], config["candidate_features"]
    if not existing or not pool or len(set(pool)) != len(pool) or len(set(existing)) != len(existing):
        raise ValueError("Provide nonempty, unique raw feature lists")
    if not set(existing).issubset(pool):
        raise ValueError("The candidate pool must include all existing features")
    reserved = {config["target"], config.get("exposure"), config.get("weight"), config.get("fold_column")}
    if reserved.intersection(pool):
        raise ValueError("Targets, exposure, weights, and folds cannot be predictors")
    if config["baseline_family"] not in FAMILIES or config["objective"] not in {"poisson", "gamma"}:
        raise ValueError("Invalid baseline family or objective")
    if not set(config.get("categorical_features", [])).issubset(pool):
        raise ValueError("Categorical features must be in the candidate pool")
    for features in config.get("additional_feature_groups", {}).values():
        if not set(features).issubset(pool):
            raise ValueError("Additional groups must use candidate features")
    caps = config["selection_new_feature_caps"]
    if any(cap < 0 for cap in caps) or caps != sorted(caps, reverse=True):
        raise ValueError("Selection caps must be nonnegative and non-increasing")
    for key in ["screen_rows", "screen_trials", "final_trials", "finalists_per_family", "threads_per_model"]:
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if min(config["inner_folds"], config["outer_folds"]) < 2 or not config["model_seeds"]:
        raise ValueError("Need at least two folds and one model seed")
    paths = [Path(config[f"{split}_path"]).resolve() for split in ("train", "validation", "test")]
    if len(set(paths)) != 3:
        raise ValueError("Training, validation, and test must be separate files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["search", "evaluate"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    (search if args.action == "search" else evaluate)(config)


if __name__ == "__main__":
    main()
