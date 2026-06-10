"""Tests for progress callbacks and cancellation in tuner and pipeline."""
import threading
import polars as pl
import pytest
from ins_gbm.data.loader import load_model_data
from ins_gbm.data.splitter import TrainTestSplit
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import ModelPipeline, ModelRecipe
from ins_gbm.progress import ProgressEvent, PipelineCancelled
from ins_gbm.tuning.tuner import HyperparameterTuner


# ── ProgressEvent dataclass ───────────────────────────────────────────────────

def test_progress_event_is_frozen():
    evt = ProgressEvent(stage="fit", message="fitting")
    with pytest.raises((AttributeError, TypeError)):
        evt.stage = "other"


def test_progress_event_optional_fields():
    evt = ProgressEvent(stage="fit", message="fitting", current=1, total=10)
    assert evt.current == 1
    assert evt.total == 10


def test_progress_event_payload():
    evt = ProgressEvent(stage="tuning", message="trial 1", payload={"best_value": 0.5})
    assert evt.payload["best_value"] == 0.5


# ── PipelineCancelled ────────────────────────────────────────────────────────

def test_pipeline_cancelled_is_exception():
    with pytest.raises(PipelineCancelled):
        raise PipelineCancelled("stopped")


# ── Tuner: progress callback fires per trial ─────────────────────────────────

def test_tuner_progress_callback_fires(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    events: list[ProgressEvent] = []
    tuner = HyperparameterTuner(n_trials=3, cv_folds=2, seed=42)
    tuner.tune(data, LightGBMModel(objective="poisson"), progress=events.append)
    tuning_events = [e for e in events if e.stage == "tuning"]
    assert len(tuning_events) == 3


def test_tuner_progress_callback_has_correct_fields(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    events: list[ProgressEvent] = []
    tuner = HyperparameterTuner(n_trials=2, cv_folds=2, seed=42)
    tuner.tune(data, LightGBMModel(objective="poisson"), progress=events.append)
    for evt in events:
        assert evt.total == 2
        assert evt.current is not None
        assert "best_value" in evt.payload


# ── Tuner: should_stop cancels via study.stop ─────────────────────────────────

def test_tuner_should_stop_after_n_trials(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    completed = []

    def record(evt):
        if evt.stage == "tuning":
            completed.append(evt)

    stop_after = 2
    call_count = [0]

    def should_stop():
        call_count[0] += 1
        return len(completed) >= stop_after

    tuner = HyperparameterTuner(n_trials=10, cv_folds=2, seed=42)
    _, history = tuner.tune(
        data, LightGBMModel(objective="poisson"),
        progress=record, should_stop=should_stop,
    )
    assert len(history) <= stop_after + 1  # +1 for the in-flight trial


# ── Pipeline: stage events emitted ───────────────────────────────────────────

def test_pipeline_emits_split_and_fit_events(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    events: list[ProgressEvent] = []
    pipeline = ModelPipeline(
        data=data,
        split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        progress=events.append,
    )
    pipeline.run()
    stages = [e.stage for e in events]
    assert "split" in stages
    assert "fit" in stages
    assert "evaluate" in stages


def test_pipeline_emits_tuning_events(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    events: list[ProgressEvent] = []
    recipe = ModelRecipe(
        model=LightGBMModel(objective="poisson"),
        tuning=HyperparameterTuner(n_trials=2, cv_folds=2, seed=42),
    )
    pipeline = ModelPipeline(
        data=data,
        split=TrainTestSplit(seed=42),
        recipe=recipe,
        progress=events.append,
    )
    pipeline.run()
    stages = [e.stage for e in events]
    assert "tuning" in stages


# ── Pipeline: cancellation between stages ────────────────────────────────────

def test_pipeline_cancel_between_stages_raises(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1", "x3"], objective="poisson",
    )
    stop_event = threading.Event()
    stages_seen: list[str] = []

    def record_and_stop(evt: ProgressEvent):
        stages_seen.append(evt.stage)
        if evt.stage == "split":
            stop_event.set()

    pipeline = ModelPipeline(
        data=data,
        split=TrainTestSplit(seed=42),
        recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        progress=record_and_stop,
        should_stop=stop_event.is_set,
    )
    with pytest.raises(PipelineCancelled):
        pipeline.run()

    assert "split" in stages_seen
    assert "fit" not in stages_seen


# ── Exported from __init__ ────────────────────────────────────────────────────

def test_exports_from_package():
    from ins_gbm import ProgressEvent, ProgressCallback, PipelineCancelled
    assert ProgressEvent is not None
    assert PipelineCancelled is not None
