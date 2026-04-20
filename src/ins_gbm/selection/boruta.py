"""Custom Boruta variable selection using shadow-feature comparison."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional

import numpy as np
import polars as pl
from scipy.stats import binomtest

from ins_gbm.data.model_data import ModelData


@dataclass
class BorutaSelector:
    """Boruta feature selection.

    Algorithm:
    1. For each iteration, create shadow features (shuffled copies of all features).
    2. Fit a model on original + shadow features.
    3. Record whether each original feature's importance > max shadow importance (a "hit").
    4. After max_iter iterations, apply a two-sided binomial test against p=0.5:
       - confirmed: significantly more hits than expected by chance
       - rejected: significantly fewer hits than expected by chance
       - tentative: neither
    """
    base_estimator: Literal["lightgbm", "random_forest"] = "lightgbm"
    max_iter: int = 50
    alpha: float = 0.05
    seed: int = 42

    def fit(self, data: ModelData) -> "FittedBorutaSelector":
        rng = np.random.default_rng(self.seed)
        original_features = list(data.feature_names)
        n_features = len(original_features)
        n_rows = data.n_rows
        hit_counts = {f: 0 for f in original_features}

        for iteration in range(self.max_iter):
            # Build shadow features
            shadow_cols = {}
            for col in original_features:
                vals = data.features[col].to_numpy()
                shuffled = vals[rng.permutation(n_rows)]
                shadow_cols[f"shadow__{col}"] = shuffled

            shadow_df = pl.DataFrame(shadow_cols)
            aug_features = pl.concat([data.features, shadow_df], how="horizontal")
            aug_names = list(aug_features.columns)
            aug_data = replace(data, features=aug_features, feature_names=aug_names)

            fitted_model = self._fit_base(aug_data, rng)
            imp = fitted_model.feature_importance()
            imp_dict = dict(zip(imp["feature"].to_list(), imp["importance"].to_list()))

            shadow_importances = [imp_dict.get(f"shadow__{col}", 0.0) for col in original_features]
            max_shadow = max(shadow_importances) if shadow_importances else 0.0

            for col in original_features:
                if imp_dict.get(col, 0.0) > max_shadow:
                    hit_counts[col] += 1

        # Binomial test: H0 = feature hits by chance (p=0.5)
        classification = {}
        for col, hits in hit_counts.items():
            result = binomtest(hits, self.max_iter, p=0.5)
            p_val = result.pvalue
            if p_val < self.alpha and hits > self.max_iter / 2:
                classification[col] = "confirmed"
            elif p_val < self.alpha and hits <= self.max_iter / 2:
                classification[col] = "rejected"
            else:
                classification[col] = "tentative"

        return FittedBorutaSelector(
            classification_map=classification,
            original_features=original_features,
        )

    def _fit_base(self, data: ModelData, rng: np.random.Generator):
        seed = int(rng.integers(0, 2**31))
        if self.base_estimator == "lightgbm":
            from ins_gbm.models.lightgbm import LightGBMModel
            return LightGBMModel(objective=data.objective or "poisson").fit(
                data, params={"n_estimators": 30, "verbose": -1, "seed": seed}
            )
        else:
            from ins_gbm.models.random_forest import RandomForestModel
            return RandomForestModel(objective=data.objective or "poisson").fit(
                data, params={"n_estimators": 30, "random_state": seed}
            )


@dataclass
class FittedBorutaSelector:
    classification_map: dict[str, str]
    original_features: list[str]

    def selected_features(self) -> list[str]:
        """Return confirmed + tentative features."""
        return [f for f in self.original_features
                if self.classification_map.get(f) in ("confirmed", "tentative")]

    def confirmed_features(self) -> list[str]:
        return [f for f in self.original_features
                if self.classification_map.get(f) == "confirmed"]

    def classification(self) -> pl.DataFrame:
        return pl.DataFrame({
            "feature": self.original_features,
            "status": [self.classification_map[f] for f in self.original_features],
        })
