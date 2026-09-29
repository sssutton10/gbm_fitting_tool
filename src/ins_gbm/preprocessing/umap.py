from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from sklearn.preprocessing import StandardScaler

from ins_gbm.data.dtypes import FIT_DTYPE, frame_to_fit_array


@dataclass
class UMAPReducer:
    """Configure UMAP dimensionality reduction.

    Args:
        n_components (int): Number of reduced components to produce. Defaults to 2.
        n_neighbors (int): Number of neighbors used to build the UMAP graph. Defaults to 15.
        min_dist (float): Minimum spacing between UMAP embedding points. Defaults to 0.1.
        seed (int): Random seed for reproducible fitting or splitting. Defaults to 42.
    """

    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    seed: int = 42

    def fit(
        self, features: pl.DataFrame, target: pl.Series | None = None
    ) -> FittedUMAPReducer:
        """Fit this transformation on training features.

        Args:
            features (pl.DataFrame): Input feature frame; rows align with the target and optional
                series.
            target (Optional[pl.Series]): Observed outcome series aligned with feature rows.
                Optional.
        """
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn is required for UMAPReducer. "
                "Install with: pip install umap-learn"
            )

        X = frame_to_fit_array(features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.seed,
        )
        reducer.fit(X_scaled)
        names = [f"umap_{i + 1}" for i in range(self.n_components)]
        return FittedUMAPReducer(
            reducer=reducer,
            scaler=scaler,
            output_names=names,
            input_names=list(features.columns),
        )


@dataclass
class FittedUMAPReducer:
    """Hold a fitted UMAP reducer and its feature mapping.

    Args:
        reducer (object): Fitted dimensionality reducer.
        scaler (StandardScaler): Fitted feature scaler used before reduction.
        output_names (list[str]): Names of features produced by the transform.
        input_names (list[str]): Input feature names used to fit this transform.
    """

    reducer: object
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
        embedding = self.reducer.transform(X_scaled).astype(FIT_DTYPE, copy=False)
        return pl.DataFrame(dict(zip(self.output_names, embedding.T)))

    def output_feature_names(self) -> list[str]:
        """Return names of columns produced by this transform."""
        return list(self.output_names)

    def component_mapping(self) -> dict[str, list[str]]:
        """Map output components to their contributing inputs."""
        return {name: self.input_names for name in self.output_names}
