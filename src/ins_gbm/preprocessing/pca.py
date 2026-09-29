from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ins_gbm.data.dtypes import FIT_DTYPE, frame_to_fit_array


@dataclass
class PCAReducer:
    """Configure principal component analysis.

    Args:
        n_components (int): Number of reduced components to produce. Defaults to 2.
    """

    n_components: int = 2

    def fit(
        self, features: pl.DataFrame, target: pl.Series | None = None
    ) -> FittedPCAReducer:
        """Fit this transformation on training features.

        Args:
            features (pl.DataFrame): Input feature frame; rows align with the target and optional
                series.
            target (pl.Series | None): Observed outcome series aligned with feature rows. Optional.
        """
        X = frame_to_fit_array(features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=self.n_components)
        pca.fit(X_scaled)
        names = [f"pca_{i + 1}" for i in range(self.n_components)]
        orig_names = list(features.columns)
        return FittedPCAReducer(
            pca=pca, scaler=scaler, output_names=names, input_names=orig_names
        )


@dataclass
class FittedPCAReducer:
    """Hold a fitted PCA reducer and its feature mapping.

    Args:
        pca (PCA): Fitted principal component model.
        scaler (StandardScaler): Fitted feature scaler used before reduction.
        output_names (list[str]): Names of features produced by the transform.
        input_names (list[str]): Input feature names used to fit this transform.
    """

    pca: PCA
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
        components = self.pca.transform(X_scaled).astype(FIT_DTYPE, copy=False)
        return pl.DataFrame(dict(zip(self.output_names, components.T)))

    def output_feature_names(self) -> list[str]:
        """Return names of columns produced by this transform."""
        return list(self.output_names)

    def component_mapping(self) -> dict[str, list[str]]:
        """Map output components to their contributing inputs."""
        return {name: self.input_names for name in self.output_names}
