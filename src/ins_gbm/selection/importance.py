"""Importance-based feature pruning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
