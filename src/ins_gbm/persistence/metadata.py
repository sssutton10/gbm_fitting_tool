from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Literal


@dataclass
class ReproducibilityMetadata:
    """Records everything needed to recreate or audit a fitted pipeline.

    Args:
        package_versions (dict[str, str]): Installed package versions captured at fitting time.
        random_seeds (dict[str, int]): Random seeds used for splitting and tuning.
        model_params (dict): Model parameters used for fitting.
        feature_names (list[str]): Ordered names of input features to use.
        input_feature_names (list[str]): Names of features expected before fitted transforms.
        selected_features (Optional[list[str]]): Feature names retained by selection.
        selection_stages (Optional[list[dict]]): Metadata describing completed selection stages.
        objective (Literal['poisson', 'gamma']): Model objective: "poisson" or "gamma".
        prediction_scale (str): Scale of stored model predictions; currently "response".
        artifact_version (int): Persistence format version. Defaults to 2.
        selection_scope (str): Selection scope: "fixed" or "fold". Defaults to
            'fixed'.
        tuning_metric (Optional[str]): Metric used to rank tuning trials. Optional.
        model_selected_features (Optional[list[str]]): Optional feature subset applied after
            preprocessing.
    """

    package_versions: dict[str, str]
    random_seeds: dict[str, int]
    model_params: dict
    feature_names: list[str]
    input_feature_names: list[str]
    selected_features: list[str] | None
    selection_stages: list[dict] | None
    objective: Literal["poisson", "gamma"]
    prediction_scale: str
    artifact_version: int = 2
    selection_scope: str = "fixed"
    tuning_metric: str | None = None
    model_selected_features: list[str] | None = None


def _package_version(name: str) -> str:
    """Look up an installed package version for reproducibility metadata.

    Args:
        name (str): Name of the requested feature, model, metric, or stage.
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_metadata(
    fitted_model,
    selected_features: list[str] | None,
    split_seed: int | None = None,
    tuning_seed: int | None = None,
    input_feature_names: list[str] | None = None,
    selection_stages: list[dict] | None = None,
    selection_scope: str = "fixed",
    tuning_metric: str | None = None,
    model_selected_features: list[str] | None = None,
) -> ReproducibilityMetadata:
    """Collect fitted pipeline settings and package versions for persistence.

    Args:
        fitted_model (object): Fitted model to evaluate or persist.
        selected_features (Optional[list[str]]): Feature names retained by selection.
        split_seed (Optional[int]): Random seed used for data splits. Optional.
        tuning_seed (Optional[int]): Random seed used during hyperparameter tuning. Optional.
        input_feature_names (Optional[list[str]]): Names of features expected before fitted
            transforms. Optional.
        selection_stages (Optional[list[dict]]): Metadata describing completed selection stages.
            Optional.
        selection_scope (str): Selection scope: "fixed" or "fold". Defaults to
            'fixed'.
        tuning_metric (Optional[str]): Metric used to rank tuning trials. Optional.
        model_selected_features (Optional[list[str]]): Optional feature subset applied after
            preprocessing.
    """
    packages = [
        "ins_gbm",
        "polars",
        "numpy",
        "scikit-learn",
        "optuna",
        "lightgbm",
        "xgboost",
        "catboost",
    ]
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
        input_feature_names=(
            list(input_feature_names)
            if input_feature_names is not None
            else list(fitted_model.feature_names)
        ),
        selected_features=selected_features,
        model_selected_features=model_selected_features,
        selection_stages=selection_stages,
        objective=fitted_model.objective,
        prediction_scale="response",
        selection_scope=selection_scope,
        tuning_metric=tuning_metric,
    )
