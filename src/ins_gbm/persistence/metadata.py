from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ReproducibilityMetadata:
    """Records everything needed to recreate or audit a fitted pipeline."""
    package_versions: dict[str, str]
    random_seeds: dict[str, int]
    model_params: dict
    feature_names: list[str]
    selected_features: Optional[list[str]]
    objective: Literal["poisson", "gamma"]
    prediction_scale: str


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_metadata(
    fitted_model,
    selected_features: Optional[list[str]],
    split_seed: Optional[int] = None,
    tuning_seed: Optional[int] = None,
) -> ReproducibilityMetadata:
    packages = ["ins_gbm", "polars", "numpy", "scikit-learn", "optuna",
                "lightgbm", "xgboost", "catboost"]
    versions = {pkg: _package_version(pkg) for pkg in packages}

    seeds: dict[str, int] = {}
    if split_seed is not None:
        seeds["split"] = split_seed
    if tuning_seed is not None:
        seeds["tuning"] = tuning_seed

    return ReproducibilityMetadata(
        package_versions=versions,
        random_seeds=seeds,
        model_params=dict(fitted_model.params),
        feature_names=list(fitted_model.feature_names),
        selected_features=selected_features,
        objective=fitted_model.objective,
        prediction_scale="response",
    )
