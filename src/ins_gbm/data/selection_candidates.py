"""Resolve a selector's optional starting feature set at the right stage."""

from dataclasses import replace
from typing import Any

from ins_gbm.data.model_data import ModelData


def select_raw_candidates(data: ModelData, selector: Any | None) -> ModelData:
    """Restrict raw inputs before encoding when requested by a selector."""
    if selector is None or getattr(selector, "candidate_stage", None) != "raw":
        return data
    names = getattr(selector, "candidate_features", None)
    return data.select_features(names) if names is not None else data


def select_encoded_candidates(data: ModelData, selector: Any | None) -> ModelData:
    """Restrict the encoded feature matrix before fitting a selector."""
    if selector is None or getattr(selector, "candidate_stage", None) != "encoded":
        return data
    names = getattr(selector, "candidate_features", None)
    return data.select_features(names) if names is not None else data


def selector_without_candidates(selector: Any) -> Any:
    """Return a built-in selector configured for already restricted input."""
    from ins_gbm.selection.boruta import BorutaSelector
    from ins_gbm.selection.importance import StagedImportanceSelector

    if isinstance(selector, (BorutaSelector, StagedImportanceSelector)) and (
        selector.candidate_features is not None
    ):
        return replace(selector, candidate_features=None)
    return selector
