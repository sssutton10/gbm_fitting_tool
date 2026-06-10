"""Progress callbacks and cancellation support for ModelPipeline and HyperparameterTuner."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class ProgressEvent:
    stage: str          # "split"|"tuning"|"encode"|"select"|"preprocess"|"fit"|"evaluate"
    message: str
    current: Optional[int] = None   # e.g. trial number
    total: Optional[int] = None     # e.g. n_trials
    payload: dict = field(default_factory=dict)


ProgressCallback = Callable[[ProgressEvent], None]


class PipelineCancelled(Exception):
    """Raised when a pipeline run is cancelled via the should_stop callback."""
