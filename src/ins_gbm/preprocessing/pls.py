from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import polars as pl
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from ins_gbm.data.dtypes import (
    FIT_DTYPE,
    frame_to_fit_array,
    series_to_fit_array,
)


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
        X = frame_to_fit_array(features)
        y = series_to_fit_array(target).reshape(-1, 1)
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
        X = frame_to_fit_array(features, self.input_names)
        X_scaled = self.scaler.transform(X)
        result = self.pls.transform(X_scaled)
        components = result[0] if isinstance(result, tuple) else result
        components = components.astype(FIT_DTYPE, copy=False)
        return pl.DataFrame(dict(zip(self.output_names, components.T)))

    def output_feature_names(self) -> list[str]:
        return list(self.output_names)

    def component_mapping(self) -> dict[str, list[str]]:
        return {name: self.input_names for name in self.output_names}
