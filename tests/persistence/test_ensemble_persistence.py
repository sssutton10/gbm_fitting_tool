from dataclasses import replace
import json

import cloudpickle
import numpy as np
import polars as pl
import pytest

from ins_gbm import EnsemblePipeline, ModelData, ModelRecipe, load_ensemble, save_ensemble
from ins_gbm.models.lightgbm import LightGBMModel


@pytest.mark.parametrize("objective", ["poisson", "gamma"])
@pytest.mark.parametrize("method", ["blending", "stacking", "ridge", "seed_average", "bagging"])
def test_ensemble_roundtrip_is_compact_and_preserves_evaluation(tmp_path, objective, method):
    n = 36
    data = ModelData(
        features=pl.DataFrame({"x": np.arange(n) / n}),
        target=pl.Series(np.array([1., 2., 3.] * 12)),
        feature_names=["x"], exposure=pl.Series(np.linspace(.5, 2., n)),
        offset=pl.Series(np.linspace(-.2, .2, n)),
        weight=pl.Series(np.linspace(1., 3., n)),
        cv_fold=pl.Series(np.arange(n) % 3), objective=objective,
    )
    template = ModelRecipe(
        model=LightGBMModel(objective=objective),
        params={"n_estimators": 3, "num_threads": 1},
    ).fit(data)
    kwargs = {"blend_weights": [1.]} if method == "blending" else {}
    if method == "ridge":
        kwargs["meta_learner"] = "ridge"
    result = EnsemblePipeline(
        [template], method="stacking" if method == "ridge" else method,
        n_members=2, **kwargs,
    ).run()
    before = result.predict(data).to_numpy()
    if method == "bagging":
        save_ensemble(result, tmp_path)
    else:
        result.save(tmp_path)
    restored = load_ensemble(tmp_path)
    assert (tmp_path / "ensemble.pkl").exists()
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["artifact_version"] == 1
    assert metadata["objective"] == objective
    assert metadata["prediction_scale"] == "response"
    assert restored.configuration == result.configuration
    assert restored.ensemble.fitted_pipelines is restored.base_pipelines
    for original, loaded in zip(result.base_pipelines, restored.base_pipelines):
        assert original.raw_train_data is not None
        assert loaded.raw_train_data is None
        assert loaded.input_schema is not None
    if method in {"stacking", "ridge"}:
        assert result.ensemble.oof_predictions is not None
        assert restored.ensemble.oof_predictions is None
    np.testing.assert_allclose(restored.predict(data), before, rtol=1e-12)
    np.testing.assert_allclose(
        restored.evaluate(data).metrics()["value"], result.evaluate(data).metrics()["value"],
        rtol=1e-12,
    )
    # Compact members cannot accidentally re-enter training-dependent workflows.
    with pytest.raises(RuntimeError, match="Training data is not attached"):
        EnsemblePipeline([restored.base_pipelines[0]], method="seed_average").run()
    changed = replace(data, exposure=data.exposure * 2)
    np.testing.assert_allclose(restored.predict(changed), result.predict(changed), rtol=1e-12)


def test_invalid_ensemble_artifact_version(tmp_path):
    with (tmp_path / "ensemble.pkl").open("wb") as stream:
        cloudpickle.dump({"artifact_version": 999}, stream)
    with pytest.raises(ValueError, match="version"):
        load_ensemble(tmp_path)
