from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class PLSReducer:
    """Partial Least Squares dimensionality reduction (supervised).

    Requires target at fit time. Must only be fit on training data inside each
    CV fold to avoid target leakage.
    """
    n_components: int = 2

    def fit(self, features: pl.DataFrame, target: Optional[pl.Series] = None) -> "FittedPLSReducer":
        if target is None:
            raise ValueError("PLSReducer requires target at fit time (supervised method)")
        X = features.to_numpy().astype(np.float64)
        y = target.to_numpy().astype(np.float64).reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pls = PLSRegression(n_components=self.n_components)
        pls.fit(X_scaled, y)
        names = [f"pls_{i+1}" for i in range(self.n_components)]
        return FittedPLSReducer(pls=pls, scaler=scaler, output_names=names,
                                input_names=list(features.columns))


@dataclass
class FittedPLSReducer:
    pls: PLSRegression
    scaler: StandardScaler
    output_names: list[str]
    input_names: list[str]

    def transform(self, features: pl.DataFrame) -> pl.DataFrame:
        X = features.select(self.input_names).to_numpy().astype(np.float64)
        X_scaled = self.scaler.transform(X)
        result = self.pls.transform(X_scaled)
        components = result[0] if isinstance(result, tuple) else result
        return pl.DataFrame(dict(zip(self.output_names, components.T)))

    def output_feature_names(self) -> list[str]:
        return list(self.output_names)

    def component_mapping(self) -> dict[str, list[str]]:
        return {name: self.input_names for name in self.output_names}
