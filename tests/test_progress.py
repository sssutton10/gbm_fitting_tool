"""Tests for progress callbacks and cancellation in tuner and pipeline."""
import threading

import pytest

from ins_gbm.data.loader import load_model_data
from ins_gbm.models.lightgbm import LightGBMModel
from ins_gbm.pipeline import ModelPipeline, ModelRecipe
from ins_gbm.progress import PipelineCancelled, ProgressEvent
from ins_gbm.tuning.tuner import HyperparameterTuner


def _data(path):
    return load_model_data(path=str(path), target="claim_count", exposure="exposure", feature_cols=["x1", "x3"], objective="poisson")


def test_progress_event_is_frozen():
    with pytest.raises((AttributeError, TypeError)):
        ProgressEvent(stage="fit", message="fitting").stage = "other"


def test_tuner_progress_callback_fires(poisson_parquet):
    events = []
    HyperparameterTuner(
        n_trials=2,
        cv_folds=2,
        seed=42,
        show_progress_bar=False,
    ).tune(
        _data(poisson_parquet),
        LightGBMModel(objective="poisson"),
        progress=events.append,
    )
    assert len([event for event in events if event.stage == "tuning"]) == 2


def test_tuner_progress_bar_shows_trial_count(poisson_parquet, capsys):
    HyperparameterTuner(
        n_trials=2,
        cv_folds=2,
        seed=42,
    ).tune(
        _data(poisson_parquet),
        LightGBMModel(objective="poisson"),
    )

    stderr = capsys.readouterr().err
    assert "Hyperparameter tuning" in stderr
    assert "2/2" in stderr


def test_pipeline_emits_fit_but_not_split_or_evaluate_events(poisson_parquet):
    events = []
    ModelPipeline(data=_data(poisson_parquet), recipe=ModelRecipe(model=LightGBMModel(objective="poisson")), progress=events.append).run()
    stages = [event.stage for event in events]
    assert "fit" in stages
    assert "split" not in stages
    assert "evaluate" not in stages


def test_pipeline_cancellation_before_fit_raises(poisson_parquet):
    stop_event = threading.Event()

    def stop_on_fit(event):
        if event.stage == "fit":
            stop_event.set()

    pipeline = ModelPipeline(
        data=_data(poisson_parquet), recipe=ModelRecipe(model=LightGBMModel(objective="poisson")),
        progress=stop_on_fit, should_stop=stop_event.is_set,
    )
    with pytest.raises(PipelineCancelled):
        pipeline.run()
