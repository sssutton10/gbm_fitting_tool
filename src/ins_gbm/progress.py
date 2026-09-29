"""Progress callbacks and cancellation support for ModelPipeline and HyperparameterTuner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProgressEvent:
    """Report a pipeline stage and its optional progress details.

    Args:
        stage (str): Pipeline stage associated with the event.
        message (str): Human-readable progress message.
        current (Optional[int]): Current completed item count, when known. Optional.
        total (Optional[int]): Total item count, when known. Optional.
        payload (dict): Optional extra progress details.
    """

    stage: str  # "split"|"tuning"|"encode"|"select"|"preprocess"|"fit"|"evaluate"
    message: str
    current: int | None = None  # e.g. trial number
    total: int | None = None  # e.g. n_trials
    payload: dict = field(default_factory=dict)


ProgressCallback = Callable[[ProgressEvent], None]


class PipelineCancelled(Exception):
    """Raised when a pipeline run is cancelled via the should_stop callback."""
