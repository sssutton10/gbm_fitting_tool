from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler


@dataclass
class UMAPReducer:
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    seed: int = 42

    def fit(self, features: pl.DataFrame, target: Optional[pl.Series] = None) -> "FittedUMAPReducer":
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn is required for UMAPReducer. "
                              "Install with: pip install umap-learn")

        X = features.to_numpy().astype(np.float64)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.seed,
        )
        reducer.fit(X_scaled)
        names = [f"umap_{i+1}" for i in range(self.n_components)]
        return FittedUMAPReducer(reducer=reducer, scaler=scaler, output_names=names,
                                 input_names=list(features.columns))


@dataclass
class FittedUMAPReducer:
    reducer: object
    scaler: StandardScaler
    output_names: list[str]
    input_names: list[str]

    def transform(self, features: pl.DataFrame) -> pl.DataFrame:
        X = features.select(self.input_names).to_numpy().astype(np.float64)
        X_scaled = self.scaler.transform(X)
        embedding = self.reducer.transform(X_scaled)
        return pl.DataFrame(dict(zip(self.output_names, embedding.T)))

    def output_feature_names(self) -> list[str]:
        return list(self.output_names)

    def component_mapping(self) -> dict[str, list[str]]:
        return {name: self.input_names for name in self.output_names}
