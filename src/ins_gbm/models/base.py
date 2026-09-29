from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import polars as pl

from ins_gbm.data.model_data import ModelData

PredictionType = Literal["response", "rate", "link"]
Objective = Literal["poisson", "gamma"]


def resolve_objective(
    model_objective: Objective | None,
    data: ModelData,
) -> Objective:
    """Resolve an objective from model configuration, data, then legacy default.

    Args:
        model_objective (Optional[Objective]): Optional objective configured on the model.
        data (ModelData): Model data to fit, transform, predict, or evaluate.
    """
    if (
        model_objective is not None
        and data.objective is not None
        and model_objective != data.objective
    ):
        raise ValueError(
            f"model objective {model_objective!r} conflicts with data objective {data.objective!r}"
        )
    return model_objective or data.objective or "poisson"


def validate_prediction_type(prediction_type: str, objective: Objective) -> None:
    """Validate prediction type.

    Args:
        prediction_type (str): Prediction scale: "response", "rate", or "link"; "rate" is
            unavailable for Gamma.
        objective (Objective): Model objective: "poisson" or "gamma".
    """
    if prediction_type not in {"response", "rate", "link"}:
        raise ValueError("prediction_type must be 'response', 'rate', or 'link'")
    if prediction_type == "rate" and objective == "gamma":
        raise ValueError("prediction_type='rate' is invalid for gamma objective")


@dataclass(frozen=True)
class ModelCapabilities:
    """Describe the objectives and features supported by a model wrapper.

    Args:
        supports_poisson (bool): Whether Poisson fitting is supported.
        supports_gamma (bool): Whether Gamma fitting is supported.
        supports_offset (bool): Whether model offsets are supported.
        supports_sample_weight (bool): Whether observation weights are supported.
        supports_feature_importance (bool): Whether fitted feature importance is available.
    """

    supports_poisson: bool
    supports_gamma: bool
    supports_offset: bool
    supports_sample_weight: bool
    supports_feature_importance: bool


@dataclass
class FittedModel:
    """Wrapper around a trained model with a uniform predict/importance interface.

    Args:
        model (Any): Model wrapper or fitted model to use.
        params (dict): Optional model or estimator parameter mapping.
        framework (str): Name of the underlying model framework.
        objective (Objective): Model objective: "poisson" or "gamma".
        feature_names (list[str]): Ordered names of input features to use.
        predict_fn (Callable[['ModelData', PredictionType], pl.Series]): Callable that produces predictions on a requested scale.
        importance_fn (Callable[..., pl.DataFrame]): Callable that returns fitted feature importance.
        transform_chain (Optional[Any]): Optional fitted transformations applied before prediction. Optional.
    """

    model: Any
    params: dict
    framework: str
    objective: Objective
    feature_names: list[str]
    predict_fn: Callable[[ModelData, PredictionType], pl.Series]
    importance_fn: Callable[..., pl.DataFrame]
    transform_chain: Any | None = None

    def predict(
        self,
        data: ModelData,
        prediction_type: PredictionType = "response",
    ) -> pl.Series:
        """Generate fitted model predictions on the requested scale.

        Args:
            data (ModelData): Model data to fit, transform, predict, or evaluate.
            prediction_type (PredictionType): Prediction scale: "response", "rate", or "link";
                "rate" is unavailable for Gamma. Defaults to 'response'.
        """
        validate_prediction_type(prediction_type, self.objective)
        data.validate_for_prediction()
        if data.objective is not None and data.objective != self.objective:
            raise ValueError(
                f"prediction data objective {data.objective!r} conflicts with fitted objective {self.objective!r}"
            )
        current = (
            self.transform_chain.transform(data)
            if self.transform_chain is not None
            else data
        )
        return self.predict_fn(current, prediction_type)

    def feature_importance(self, importance_type: str | None = None) -> pl.DataFrame:
        """Return feature scores, optionally using a framework-native importance type.

        The accepted names are deliberately model-framework specific.  Passing
        an unsupported name raises ``ValueError`` rather than silently using a
        different importance measure.

        Args:
            importance_type (Optional[str]): Optional framework-specific importance measure.
        """
        # Keep existing no-argument importance callbacks working for callers
        # that do not request a framework-specific measure.
        if importance_type is None:
            return self.importance_fn()
        return self.importance_fn(importance_type)


@runtime_checkable
class BaseModel(Protocol):
    """Protocol that all model wrappers must satisfy.

    Args:
        objective (Optional[Objective]): Model objective: "poisson" or "gamma".
    """

    objective: Objective | None

    def fit(
        self,
        data: ModelData,
        params: dict | None = None,
        *,
        feature_names: list[str] | None = None,
        encoder: Any | None = None,
        preprocessing: list[Any] | None = None,
    ) -> FittedModel:
        """Fit the model and return its prediction and importance interface.

        Args:
            data: Training features, target, and optional observation fields.
            params: Optional model parameter overrides.
            feature_names: Optional ordered subset of input features.
            encoder: Optional feature encoder fitted on the training rows.
            preprocessing: Optional ordered preprocessing steps.
        """
        ...

    def default_search_space(self) -> dict:
        """Return the model's default parameter search distributions."""
        ...

    def capabilities(self) -> ModelCapabilities:
        """Return supported objectives and fitting features."""
        ...
