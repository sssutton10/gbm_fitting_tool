from __future__ import annotations

from dataclasses import dataclass

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

    Args:
        n_components (int): Number of reduced components to produce. Defaults to 2.
    """

    n_components: int = 2

    def fit(
        self, features: pl.DataFrame, target: pl.Series | None = None
    ) -> FittedPLSReducer:
        """Fit this transformation on training features.

        Args:
            features (pl.DataFrame): Input feature frame; rows align with the target and optional
                series.
            target (Optional[pl.Series]): Observed outcome series aligned with feature rows.
                Optional.
        """
        if target is None:
            raise ValueError(
                "PLSReducer requires target at fit time (supervised method)"
            )
        X = frame_to_fit_array(features)
        y = series_to_fit_array(target).reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pls = PLSRegression(n_components=self.n_components)
        pls.fit(X_scaled, y)
        names = [f"pls_{i + 1}" for i in range(self.n_components)]
        return FittedPLSReducer(
            pls=pls,
            scaler=scaler,
            output_names=names,
            input_names=list(features.columns),
        )


@dataclass
class FittedPLSReducer:
    """Hold a fitted partial least squares reducer and its feature mapping.

    Args:
        pls (PLSRegression): Fitted partial least squares estimator.
        scaler (StandardScaler): Fitted feature scaler used before reduction.
        output_names (list[str]): Names of features produced by the transform.
        input_names (list[str]): Input feature names used to fit this transform.
    """

    pls: PLSRegression
    scaler: StandardScaler
    output_names: list[str]
    input_names: list[str]

    def transform(self, features: pl.DataFrame) -> pl.DataFrame:
        """Apply the fitted transformation to input features.

        Args:
            features (pl.DataFrame): Input feature frame; rows align with the target and optional
                series.
        """
        X = frame_to_fit_array(features, self.input_names)
        X_scaled = self.scaler.transform(X)
        result = self.pls.transform(X_scaled)
        components = result[0] if isinstance(result, tuple) else result
        components = components.astype(FIT_DTYPE, copy=False)
        return pl.DataFrame(dict(zip(self.output_names, components.T)))

    def output_feature_names(self) -> list[str]:
        """Return names of columns produced by this transform."""
        return list(self.output_names)

    def component_mapping(self) -> dict[str, list[str]]:
        """Map output components to their contributing inputs."""
        return {name: self.input_names for name in self.output_names}
