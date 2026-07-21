"""Importance-based feature pruning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData
from ins_gbm.models.base import FittedModel


@dataclass
class ImportancePruner:
    """Prune features by importance score.

    Exactly one of threshold, percentile, or top_n should be set.
    - threshold: keep features with importance >= threshold
    - percentile: keep features at or above this percentile (0-100)
    - top_n: keep top-N features by importance
    """
    threshold: Optional[float] = None
    percentile: Optional[float] = None
    top_n: Optional[int] = None

    def __post_init__(self):
        n_set = sum(x is not None for x in [self.threshold, self.percentile, self.top_n])
        if n_set == 0:
            self.threshold = 0.0  # default: keep all non-zero importance features
        elif n_set > 1:
            raise ValueError("Set only one of: threshold, percentile, top_n")

    def fit(self, data: ModelData, fitted_model: FittedModel) -> "FittedImportancePruner":
        imp = fitted_model.feature_importance()
        names = imp["feature"].to_list()
        scores = imp["importance"].to_numpy().astype(float)

        if self.top_n is not None:
            order = np.argsort(-scores)
            keep = [names[i] for i in order[: self.top_n]]
        elif self.percentile is not None:
            cutoff = np.percentile(scores, 100.0 - self.percentile)
            keep = [n for n, s in zip(names, scores) if s >= cutoff]
        else:
            cutoff = self.threshold if self.threshold is not None else 0.0
            keep = [n for n, s in zip(names, scores) if s >= cutoff]

        # Preserve original feature order
        original_order = list(data.feature_names)
        keep_set = set(keep)
        selected = [f for f in original_order if f in keep_set]

        return FittedImportancePruner(selected_feature_names=selected)


@dataclass
class FittedImportancePruner:
    selected_feature_names: list[str]

    def selected_features(self) -> list[str]:
        return list(self.selected_feature_names)


@dataclass
class ImportanceSelectionStage:
    """One model fit in an ordered importance-selection workflow.

    ``model`` is an unfitted model wrapper, while ``params`` are deliberately
    independent from the final model and hyperparameter tuner.  This makes it
    possible to use a fast, shallow screen before a more realistic pruning
    model.
    """
    model: Any
    max_features: int
    importance_type: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.max_features, bool) or not isinstance(self.max_features, int):
            raise ValueError("max_features must be a positive integer")
        if self.max_features < 1:
            raise ValueError("max_features must be a positive integer")
        if not isinstance(self.params, dict):
            raise ValueError("params must be a dictionary")
        if self.name is not None and not self.name:
            raise ValueError("name must be non-empty when provided")


@dataclass
class FittedImportanceSelectionStage:
    """Auditable result of one importance-selection stage."""
    name: str
    max_features: int
    importance_type: Optional[str]
    model_framework: str
    model_params: dict[str, Any]
    ranking: pl.DataFrame
    selected_feature_names: list[str]

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_features": self.max_features,
            "importance_type": self.importance_type,
            "model_framework": self.model_framework,
            "model_params": dict(self.model_params),
            "selected_features": list(self.selected_feature_names),
        }


@dataclass
class StagedImportanceSelector:
    """Select features through one or more model-based importance stages.

    The selector operates on the columns it receives, so in ``ModelPipeline``
    it ranks encoded model columns after the encoder has been fitted.
    """
    stages: list[ImportanceSelectionStage]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("stages must contain at least one selection stage")
        if not all(isinstance(stage, ImportanceSelectionStage) for stage in self.stages):
            raise ValueError("stages must contain ImportanceSelectionStage instances")
        names = [stage.name for stage in self.stages if stage.name is not None]
        if len(names) != len(set(names)):
            raise ValueError("selection stage names must be unique")

    def fit(self, data: ModelData) -> "FittedStagedImportanceSelector":
        current = data
        fitted_stages: list[FittedImportanceSelectionStage] = []

        for index, stage in enumerate(self.stages, start=1):
            capabilities = stage.model.capabilities()
            if not capabilities.supports_feature_importance:
                raise ValueError(
                    f"Selection stage {index} model does not support feature importance"
                )

            fitted_model = stage.model.fit(current, params=dict(stage.params))
            importance = fitted_model.feature_importance(stage.importance_type)
            required_columns = {"feature", "importance"}
            if not required_columns.issubset(importance.columns):
                raise ValueError(
                    f"Selection stage {index} importance must contain "
                    "'feature' and 'importance' columns"
                )

            feature_order = list(current.feature_names)
            score_by_feature = dict(
                zip(importance["feature"].to_list(), importance["importance"].to_list())
            )
            try:
                scores = [float(score_by_feature.get(feature, 0.0)) for feature in feature_order]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Selection stage {index} importance scores must be numeric"
                ) from exc
            if not np.isfinite(scores).all():
                raise ValueError(
                    f"Selection stage {index} importance scores must be finite"
                )

            # Python's sort is stable, retaining incoming order for equal scores.
            ranked_indices = sorted(range(len(feature_order)), key=lambda i: -scores[i])
            n_keep = min(stage.max_features, len(feature_order))
            selected_set = {feature_order[i] for i in ranked_indices[:n_keep]}
            selected = [feature for feature in feature_order if feature in selected_set]
            ranking = pl.DataFrame(
                {
                    "feature": [feature_order[i] for i in ranked_indices],
                    "importance": [scores[i] for i in ranked_indices],
                    "rank": list(range(1, len(feature_order) + 1)),
                    "selected": [feature_order[i] in selected_set for i in ranked_indices],
                }
            )
            stage_name = stage.name or f"stage_{index}"
            fitted_stages.append(
                FittedImportanceSelectionStage(
                    name=stage_name,
                    max_features=stage.max_features,
                    importance_type=stage.importance_type,
                    model_framework=fitted_model.framework,
                    model_params=dict(fitted_model.params),
                    ranking=ranking,
                    selected_feature_names=selected,
                )
            )
            current = current.with_features(current.features.select(selected))

        return FittedStagedImportanceSelector(stages=fitted_stages)


@dataclass
class FittedStagedImportanceSelector:
    stages: list[FittedImportanceSelectionStage]

    def selected_features(self) -> list[str]:
        return list(self.stages[-1].selected_feature_names)

    def stage_results(self) -> list[FittedImportanceSelectionStage]:
        return list(self.stages)

    def selection_metadata(self) -> list[dict[str, Any]]:
        return [stage.metadata() for stage in self.stages]
