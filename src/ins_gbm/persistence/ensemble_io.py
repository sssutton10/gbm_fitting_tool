"""Compact, versioned persistence for complete ensemble results."""

import json
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ins_gbm.ensemble.pipeline import EnsembleResult

_ARTIFACT_VERSION = 1


def save_ensemble(result: "EnsembleResult", output_dir: str) -> None:
    """Save an ensemble without raw training rows or per-row OOF predictions.

    Writes ensemble.pkl and human-readable metadata.json. As with pipeline
    artifacts, only load trusted pickle files in a compatible environment.
    The original result retains its training data and diagnostics.
    """
    import cloudpickle

    from ins_gbm.ensemble.stacking import FittedStackingEnsemble

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    members = [
        replace(
            pipeline,
            raw_train_data=None,
            input_schema=pipeline._input_schema(),
            fitted_model=replace(
                pipeline.fitted_model,
                # CatBoost materializes lazy importance during serialization.
                # Copy its callback so the live result keeps its existing state.
                importance_fn=(
                    replace(pipeline.fitted_model.importance_fn)
                    if is_dataclass(pipeline.fitted_model.importance_fn)
                    else pipeline.fitted_model.importance_fn
                ),
            ),
        )
        for pipeline in result.ensemble.fitted_pipelines
    ]
    fitted = replace(result.ensemble, fitted_pipelines=members)
    if isinstance(fitted, FittedStackingEnsemble):
        fitted = replace(fitted, oof_predictions=None)
    compact = replace(result, ensemble=fitted, base_pipelines=members)
    metadata = {
        "artifact_version": _ARTIFACT_VERSION,
        "objective": members[0].fitted_model.objective,
        "prediction_scale": "response",
        "configuration": result.configuration,
        "ensemble_type": type(fitted).__name__,
        "members": [asdict(member.metadata) for member in members],
    }
    if hasattr(fitted, "weights"):
        metadata["weights"] = fitted.weights
    if isinstance(fitted, FittedStackingEnsemble):
        metadata["meta_learner"] = type(fitted.meta_learner).__name__
        metadata["distribution_aware"] = fitted.distribution_aware
    # Retain the version in the pickle as well; do not require the JSON sidecar.
    with (output / "ensemble.pkl").open("wb") as stream:
        cloudpickle.dump(
            {"artifact_version": _ARTIFACT_VERSION, "result": compact}, stream
        )
    with (output / "metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)


def load_ensemble(output_dir: str) -> "EnsembleResult":
    """Load a trusted ensemble artifact for prediction and holdout evaluation.

    Raw training rows and stacking OOF predictions are not restored.
    """
    import cloudpickle

    from ins_gbm.ensemble.pipeline import EnsembleResult

    path = Path(output_dir) / "ensemble.pkl"
    with path.open("rb") as stream:
        artifact = cloudpickle.load(stream)
    if (
        not isinstance(artifact, dict)
        or artifact.get("artifact_version") != _ARTIFACT_VERSION
    ):
        raise ValueError("Unsupported ensemble artifact version")
    result = artifact.get("result")
    if not isinstance(result, EnsembleResult):
        raise ValueError("Invalid ensemble artifact: expected an EnsembleResult")
    return result
