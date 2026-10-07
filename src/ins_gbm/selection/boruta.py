"""Custom Boruta variable selection using shadow-feature comparison."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import polars as pl
from scipy.stats import binomtest

from ins_gbm.data.model_data import ModelData
from ins_gbm.data.schema import FeatureSchema


@dataclass
class BorutaSelector:
    """Boruta feature selection.

    Algorithm:
    1. For each iteration, create shadow features (shuffled copies of all features).
    2. Fit a model on original + shadow features.
    3. Record whether each original feature's importance > max shadow importance (a "hit").
    4. After max_iter iterations, apply a two-sided binomial test against p=0.5,
       with Bonferroni correction across input features by default:
       - confirmed: significantly more hits than expected by chance
       - rejected: significantly fewer hits than expected by chance
       - tentative: neither

    Args:
        base_estimator (Literal['lightgbm', 'random_forest']): Base estimator: "lightgbm" or
            "random_forest". Defaults to 'lightgbm'.
        max_iter (int): Maximum number of Boruta iterations. Defaults to 50.
        alpha (float): Significance level for Boruta confirmation. Defaults to 0.05.
        seed (int): Random seed for reproducible fitting or splitting. Defaults to 42.
        candidate_features (list[str] | None): Starting features for selection; None uses
            all features. Raw categorical names include all their encoded levels.
        candidate_stage (Literal['raw', 'encoded']): Stage at which candidate names apply.
        shadow_percentile (float): Percentile of shadow importance used as the hit
            threshold. 100 uses the strongest shadow; lower values are less strict.
        base_n_estimators (int): Number of trees in each Boruta base fit.
        multiple_testing (Literal['bonferroni', 'none']): Adjustment across features.
            The binomial comparison is a screening heuristic, especially when
            shadow_percentile is reduced; it does not guarantee false discovery rates.
    """

    base_estimator: Literal["lightgbm", "random_forest"] = "lightgbm"
    max_iter: int = 50
    alpha: float = 0.05
    seed: int = 42
    candidate_features: list[str] | None = None
    candidate_stage: Literal["raw", "encoded"] = "raw"
    shadow_percentile: float = 100.0
    base_n_estimators: int = 30
    multiple_testing: Literal["bonferroni", "none"] = "bonferroni"

    def __post_init__(self) -> None:
        if self.multiple_testing not in ("bonferroni", "none"):
            raise ValueError("multiple_testing must be 'bonferroni' or 'none'")
        if self.base_estimator not in ("lightgbm", "random_forest"):
            raise ValueError("base_estimator must be 'lightgbm' or 'random_forest'")
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self.candidate_stage not in ("raw", "encoded"):
            raise ValueError("candidate_stage must be 'raw' or 'encoded'")
        if self.candidate_features is not None:
            if not self.candidate_features:
                raise ValueError("candidate_features must contain at least one feature")
            if len(set(self.candidate_features)) != len(self.candidate_features):
                raise ValueError("candidate_features must be unique")
        if not 0 < self.shadow_percentile <= 100:
            raise ValueError("shadow_percentile must be in (0, 100]")
        if isinstance(self.base_n_estimators, bool) or not isinstance(
            self.base_n_estimators, int
        ) or self.base_n_estimators < 1:
            raise ValueError("base_n_estimators must be a positive integer")

    def fit(self, data: ModelData) -> FittedBorutaSelector:
        """Fit feature selection on the supplied training data.

        For a direct fit, supply data at the stage named by ``candidate_stage``.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
        """
        if self.candidate_features is not None:
            data = data.select_features(self.candidate_features)
        data.validate(require_multiple_folds=False)
        if not data.feature_names:
            raise ValueError("Boruta requires at least one feature")
        rng = np.random.default_rng(self.seed)
        original_features = list(data.feature_names)
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
            aug_schema = data.schema
            if aug_schema is not None:
                aug_schema = FeatureSchema(
                    numeric=[
                        *aug_schema.numeric,
                        *(f"shadow__{name}" for name in aug_schema.numeric),
                    ],
                    categorical=[
                        *aug_schema.categorical,
                        *(f"shadow__{name}" for name in aug_schema.categorical),
                    ],
                    ordinal=[
                        *aug_schema.ordinal,
                        *(f"shadow__{name}" for name in aug_schema.ordinal),
                    ],
                    passthrough=[
                        *aug_schema.passthrough,
                        *(f"shadow__{name}" for name in aug_schema.passthrough),
                    ],
                )
            aug_data = replace(
                data,
                features=aug_features,
                feature_names=aug_names,
                schema=aug_schema,
            )

            fitted_model = self._fit_base(aug_data, rng)
            imp = fitted_model.feature_importance()
            imp_dict = dict(zip(imp["feature"].to_list(), imp["importance"].to_list()))

            shadow_importances = [
                imp_dict.get(f"shadow__{col}", 0.0) for col in original_features
            ]
            shadow_cutoff = (
                float(np.percentile(shadow_importances, self.shadow_percentile))
                if shadow_importances
                else 0.0
            )

            for col in original_features:
                if imp_dict.get(col, 0.0) > shadow_cutoff:
                    hit_counts[col] += 1

        # Binomial test: H0 = feature hits by chance (p=0.5)
        classification = {}
        alpha = (
            self.alpha / len(original_features)
            if self.multiple_testing == "bonferroni" else self.alpha
        )
        for col, hits in hit_counts.items():
            result = binomtest(hits, self.max_iter, p=0.5)
            p_val = result.pvalue
            if p_val < alpha and hits > self.max_iter / 2:
                classification[col] = "confirmed"
            elif p_val < alpha and hits <= self.max_iter / 2:
                classification[col] = "rejected"
            else:
                classification[col] = "tentative"

        return FittedBorutaSelector(
            classification_map=classification,
            original_features=original_features,
            hit_counts=hit_counts,
        )

    def _fit_base(self, data: ModelData, rng: np.random.Generator):
        """Fit the Boruta base estimator on original and shadow features.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
            rng (np.random.Generator): Random number generator for shadow feature permutations.
        """
        seed = int(rng.integers(0, 2**31))
        if self.base_estimator == "lightgbm":
            from ins_gbm.models.lightgbm import LightGBMModel

            return LightGBMModel(objective=data.objective or "poisson").fit(
                data,
                params={
                    "n_estimators": self.base_n_estimators,
                    "verbose": -1,
                    "seed": seed,
                },
            )
        else:
            from ins_gbm.models.random_forest import RandomForestModel

            return RandomForestModel(objective=data.objective or "poisson").fit(
                data,
                params={"n_estimators": self.base_n_estimators, "random_state": seed},
            )


@dataclass
class FittedBorutaSelector:
    """Hold Boruta feature classifications after fitting.

    Args:
        classification_map (dict[str, str]): Boruta classification for each original feature.
        original_features (list[str]): Names of features before Boruta selection.
        hit_counts (dict[str, int]): Number of iterations each feature beat the shadow cutoff.
    """

    classification_map: dict[str, str]
    original_features: list[str]
    hit_counts: dict[str, int] | None = None

    def selected_features(self) -> list[str]:
        """Return confirmed + tentative features."""
        return [
            f
            for f in self.original_features
            if self.classification_map.get(f) in ("confirmed", "tentative")
        ]

    def confirmed_features(self) -> list[str]:
        """Return features confirmed by Boruta."""
        return [
            f
            for f in self.original_features
            if self.classification_map.get(f) == "confirmed"
        ]

    def classification(self) -> pl.DataFrame:
        """Return Boruta classifications for all original features."""
        columns = {
            "feature": self.original_features,
            "status": [self.classification_map[f] for f in self.original_features],
        }
        if self.hit_counts is not None:
            columns["hits"] = [self.hit_counts[f] for f in self.original_features]
        return pl.DataFrame(columns)
