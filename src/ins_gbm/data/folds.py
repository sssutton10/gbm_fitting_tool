"""Shared cross-validation split configuration and resolution."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ins_gbm.data.model_data import ModelData


FoldPolicy = Literal["auto", "random", "predefined"]


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    seed: int = 42
    folds: FoldPolicy = "auto"


def resolve_folds(data: ModelData, config: CVConfig) -> tuple[list, list[tuple[np.ndarray, np.ndarray]]]:
    if config.folds not in {"auto", "random", "predefined"}:
        raise ValueError("folds must be 'auto', 'random', or 'predefined'")
    use_predefined = config.folds == "predefined" or (
        config.folds == "auto" and data.cv_fold is not None
    )
    if use_predefined:
        if data.cv_fold is None:
            raise ValueError("predefined folds require ModelData.cv_fold")
        if data.cv_fold.null_count():
            raise ValueError("cv_fold must be non-null")
        values = data.cv_fold.to_numpy()
        fold_ids = sorted(np.unique(values).tolist())
        if len(fold_ids) < 2:
            raise ValueError("cross-validation requires at least two fold IDs")
        splits = [(np.where(values != fold)[0], np.where(values == fold)[0]) for fold in fold_ids]
    else:
        if config.n_splits < 2 or config.n_splits > data.n_rows:
            raise ValueError("n_splits must be between 2 and the number of rows")
        from sklearn.model_selection import KFold
        splits = list(KFold(config.n_splits, shuffle=True, random_state=config.seed).split(range(data.n_rows)))
        fold_ids = list(range(config.n_splits))
    if any(len(train) == 0 or len(held) == 0 for train, held in splits):
        raise ValueError("every fold must have non-empty training and validation rows")
    held = np.concatenate([indices for _, indices in splits])
    if len(held) != data.n_rows or len(np.unique(held)) != data.n_rows:
        raise ValueError("cross-validation folds must hold out every row exactly once")
    return fold_ids, splits
