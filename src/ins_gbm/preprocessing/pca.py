from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAReducer:
    n_components: int = 2

    def fit(self, features: pl.DataFrame, target: pl.Series | None = None) -> "FittedPCAReducer":
        X = features.to_numpy().astype(np.float64)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=self.n_components)
        pca.fit(X_scaled)
        names = [f"pca_{i+1}" for i in range(self.n_components)]
        orig_names = list(features.columns)
        return FittedPCAReducer(pca=pca, scaler=scaler, output_names=names, input_names=orig_names)


@dataclass
class FittedPCAReducer:
    pca: PCA
    scaler: StandardScaler
    output_names: list[str]
    input_names: list[str]

    def transform(self, features: pl.DataFrame) -> pl.DataFrame:
        X = features.select(self.input_names).to_numpy().astype(np.float64)
        X_scaled = self.scaler.transform(X)
        components = self.pca.transform(X_scaled)
        return pl.DataFrame(dict(zip(self.output_names, components.T)))

    def output_feature_names(self) -> list[str]:
        return list(self.output_names)

    def component_mapping(self) -> dict[str, list[str]]:
        return {name: self.input_names for name in self.output_names}
