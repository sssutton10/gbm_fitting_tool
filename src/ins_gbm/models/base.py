from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Protocol, runtime_checkable

import polars as pl

from ins_gbm.data.model_data import ModelData


PredictionType = Literal["response", "rate", "link"]
Objective = Literal["poisson", "gamma"]


@dataclass(frozen=True)
class ModelCapabilities:
    supports_poisson: bool
    supports_gamma: bool
    supports_offset: bool
    supports_sample_weight: bool
    supports_feature_importance: bool


@dataclass
class FittedModel:
    """Wrapper around a trained model with a uniform predict/importance interface."""
    model: Any
    params: dict
    framework: str
    objective: Objective
    feature_names: list[str]
    predict_fn: Callable[["ModelData", PredictionType], pl.Series]
    importance_fn: Callable[..., pl.DataFrame]
    transform_chain: Optional[Any] = None

    def predict(
        self,
        data: ModelData,
        prediction_type: PredictionType = "response",
    ) -> pl.Series:
        if prediction_type == "rate" and self.objective == "gamma":
            raise ValueError(
                "prediction_type='rate' is invalid for gamma objective"
            )
        current = (
            self.transform_chain.transform(data)
            if self.transform_chain is not None
            else data
        )
        return self.predict_fn(current, prediction_type)

    def feature_importance(self, importance_type: Optional[str] = None) -> pl.DataFrame:
        """Return feature scores, optionally using a framework-native importance type.

        The accepted names are deliberately model-framework specific.  Passing
        an unsupported name raises ``ValueError`` rather than silently using a
        different importance measure.
        """
        # Keep existing no-argument importance callbacks working for callers
        # that do not request a framework-specific measure.
        if importance_type is None:
            return self.importance_fn()
        return self.importance_fn(importance_type)


@runtime_checkable
class BaseModel(Protocol):
    """Protocol that all model wrappers must satisfy."""
    objective: Objective

    def fit(
        self,
        data: ModelData,
        params: Optional[dict] = None,
        *,
        feature_names: Optional[list[str]] = None,
        encoder: Optional[Any] = None,
        preprocessing: Optional[list[Any]] = None,
    ) -> FittedModel: ...

    def default_search_space(self) -> dict: ...

    def capabilities(self) -> ModelCapabilities: ...
