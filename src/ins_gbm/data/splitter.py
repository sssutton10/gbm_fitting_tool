from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import polars as pl

from .model_data import ModelData


@dataclass
class TrainTestSplit:
    train_ratio: float = 0.7
    seed: int = 42
    group_col: Optional[str] = None

    def split(self, data: ModelData) -> tuple[ModelData, ModelData]:
        if not 0 < self.train_ratio < 1:
            raise ValueError(
                f"train_ratio must be in (0, 1), got {self.train_ratio}"
            )

        rng = np.random.default_rng(self.seed)
        n = data.n_rows

        if self.group_col is not None:
            if self.group_col not in data.features.columns:
                raise ValueError(
                    f"group_col '{self.group_col}' not found in features"
                )
            groups = data.features[self.group_col].to_numpy()
            unique_groups = np.unique(groups)
            rng.shuffle(unique_groups)
            n_train_groups = max(1, int(len(unique_groups) * self.train_ratio))
            train_groups = set(unique_groups[:n_train_groups].tolist())
            train_mask = np.array([g in train_groups for g in groups])
        else:
            indices = np.arange(n)
            rng.shuffle(indices)
            n_train = int(n * self.train_ratio)
            train_mask = np.zeros(n, dtype=bool)
            train_mask[indices[:n_train]] = True

        return self._subset(data, train_mask), self._subset(data, ~train_mask)

    @staticmethod
    def _subset(data: ModelData, mask: np.ndarray) -> ModelData:
        s = pl.Series(mask)
        return replace(
            data,
            features=data.features.filter(s),
            target=data.target.filter(s),
            exposure=data.exposure.filter(s) if data.exposure is not None else None,
            weight=data.weight.filter(s) if data.weight is not None else None,
        )
