"""Importance-based feature pruning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

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

    Args:
        threshold (Optional[float]): Minimum importance threshold. Optional.
        percentile (Optional[float]): Importance percentile threshold. Optional.
        top_n (Optional[int]): Maximum number of highest-ranked features to keep or display.
            Optional.
    """

    threshold: float | None = None
    percentile: float | None = None
    top_n: int | None = None

    def __post_init__(self):
        """Validate and normalize values after initialization."""
        n_set = sum(
            x is not None for x in [self.threshold, self.percentile, self.top_n]
        )
        if n_set == 0:
            self.threshold = 0.0  # default: keep all non-zero importance features
        elif n_set > 1:
            raise ValueError("Set only one of: threshold, percentile, top_n")

    def fit(
        self,
        data: ModelData | FittedModel | Any = None,
        fitted_model: FittedModel | Any = None,
    ) -> FittedImportancePruner:
        """Rank the fitted model's columns, including encoded and reduced columns.

        ``fit(data, model)`` remains accepted for older callers. The data's raw
        feature names cannot describe the fitted design, so they are not used.

        Args:
            data (ModelData | FittedModel | Any): Model data to fit, transform, predict, or
                evaluate. Optional.
            fitted_model (FittedModel | Any): Fitted model to evaluate or persist. Optional.
        """
        if fitted_model is not None and data is not None:
            if not isinstance(data, ModelData):
                raise TypeError("fit(data, fitted_model) requires ModelData first")
        elif fitted_model is None:
            fitted_model = data
        fitted_model = getattr(fitted_model, "fitted_model", fitted_model)
        if not isinstance(fitted_model, FittedModel):
            raise TypeError("fit requires a FittedModel or FittedPipeline")
        imp = fitted_model.feature_importance()
        names = imp["feature"].to_list()
        scores = imp["importance"].to_numpy().astype(float)
        original_order = list(fitted_model.feature_names)
        if len(names) != len(original_order) or set(names) != set(original_order):
            raise ValueError(
                "Feature importance names do not match fitted model columns"
            )
        if not np.isfinite(scores).all():
            raise ValueError("Feature importance scores must be finite")

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
        keep_set = set(keep)
        selected = [f for f in original_order if f in keep_set]

        return FittedImportancePruner(selected_feature_names=selected)


@dataclass
class FittedImportancePruner:
    """Hold feature names retained by importance pruning.

    Args:
        selected_feature_names (list[str]): Feature names retained by selection.
    """

    selected_feature_names: list[str]

    def selected_features(self) -> list[str]:
        """Return feature names retained by the fitted selector."""
        return list(self.selected_feature_names)


@dataclass
class ImportanceSelectionStage:
    """One model fit in an ordered importance-selection workflow.

    ``model`` is an unfitted model wrapper, while ``params`` are deliberately
    independent from the final model and hyperparameter tuner.  This makes it
    possible to use a fast, shallow screen before a more realistic pruning
    model.

    Args:
        model (Any): Model wrapper or fitted model to use.
        max_features (int): Maximum features retained at this stage.
        importance_type (Optional[str]): Optional framework-specific importance measure.
        params (dict[str, Any]): Optional model or estimator parameter mapping.
        name (Optional[str]): Name of the requested feature, model, metric, or stage. Optional.
    """

    model: Any
    max_features: int
    importance_type: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize values after initialization."""
        if isinstance(self.max_features, bool) or not isinstance(
            self.max_features, int
        ):
            raise ValueError("max_features must be a positive integer")  # noqa: TRY004
        if self.max_features < 1:
            raise ValueError("max_features must be a positive integer")
        if not isinstance(self.params, dict):
            raise ValueError("params must be a dictionary")  # noqa: TRY004
        if self.name is not None and not self.name:
            raise ValueError("name must be non-empty when provided")


@dataclass
class FittedImportanceSelectionStage:
    """Auditable result of one importance-selection stage.

    Args:
        name (str): Name of the requested feature, model, metric, or stage.
        max_features (int): Maximum features retained at this stage.
        importance_type (Optional[str]): Optional framework-specific importance measure.
        model_framework (str): Framework name used at this selection stage.
        model_params (dict[str, Any]): Model parameters used for fitting.
        ranking (pl.DataFrame): Ordered table of feature importance scores.
        selected_feature_names (list[str]): Feature names retained by selection.
    """

    name: str
    max_features: int
    importance_type: str | None
    model_framework: str
    model_params: dict[str, Any]
    ranking: pl.DataFrame
    selected_feature_names: list[str]

    def metadata(self) -> dict[str, Any]:
        """Metadata."""
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

    Args:
        stages (list[ImportanceSelectionStage]): Ordered selection stages or their fitted
            results.
        candidate_features (list[str] | None): Starting features for selection; None uses
            all features. Raw categorical names include all their encoded levels.
        candidate_stage (Literal['raw', 'encoded']): Stage at which candidate names apply.
    """

    stages: list[ImportanceSelectionStage]
    candidate_features: list[str] | None = None
    candidate_stage: Literal["raw", "encoded"] = "raw"

    def __post_init__(self) -> None:
        """Validate and normalize values after initialization."""
        if not self.stages:
            raise ValueError("stages must contain at least one selection stage")
        if not all(
            isinstance(stage, ImportanceSelectionStage) for stage in self.stages
        ):
            raise ValueError("stages must contain ImportanceSelectionStage instances")
        names = [stage.name for stage in self.stages if stage.name is not None]
        if len(names) != len(set(names)):
            raise ValueError("selection stage names must be unique")
        if self.candidate_stage not in ("raw", "encoded"):
            raise ValueError("candidate_stage must be 'raw' or 'encoded'")
        if self.candidate_features is not None:
            if not self.candidate_features:
                raise ValueError("candidate_features must contain at least one feature")
            if len(set(self.candidate_features)) != len(self.candidate_features):
                raise ValueError("candidate_features must be unique")

    def fit(self, data: ModelData) -> FittedStagedImportanceSelector:
        """Fit feature selection on the supplied training data.

        For a direct fit, supply data at the stage named by ``candidate_stage``.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
        """
        current = (
            data.select_features(self.candidate_features)
            if self.candidate_features is not None
            else data
        )
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
                scores = [
                    float(score_by_feature.get(feature, 0.0))
                    for feature in feature_order
                ]
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
                    "selected": [
                        feature_order[i] in selected_set for i in ranked_indices
                    ],
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
    """Hold the results of each importance selection stage.

    Args:
        stages (list[FittedImportanceSelectionStage]): Ordered selection stages or their fitted
            results.
    """

    stages: list[FittedImportanceSelectionStage]

    def selected_features(self) -> list[str]:
        """Return feature names retained by the fitted selector."""
        return list(self.stages[-1].selected_feature_names)

    def stage_results(self) -> list[FittedImportanceSelectionStage]:
        """Stage results."""
        return list(self.stages)

    def selection_metadata(self) -> list[dict[str, Any]]:
        """Selection metadata."""
        return [stage.metadata() for stage in self.stages]
