from __future__ import annotations

"""Convenience helpers for constructing Optuna search spaces.

Each model's ``default_search_space()`` already returns a ready-to-use
distribution dict.  This module provides lightweight utilities for callers
who want to override or extend those defaults.
"""


def narrow_search_space(space: dict, **overrides) -> dict:
    """Return a copy of *space* with selected distributions replaced.

    Parameters
    ----------
    space : dict
        A distribution dict as returned by ``model.default_search_space()``.
    **overrides
        Keyword arguments whose keys are parameter names and values are
        replacement Optuna distributions.

    Examples
    --------
    >>> from ins_gbm.models.lightgbm import LightGBMModel
    >>> import optuna
    >>> space = LightGBMModel().default_search_space()
    >>> restricted = narrow_search_space(
    ...     space,
    ...     n_estimators=optuna.distributions.IntDistribution(50, 150),
    ... )
    """
    result = dict(space)
    result.update(overrides)
    return result
