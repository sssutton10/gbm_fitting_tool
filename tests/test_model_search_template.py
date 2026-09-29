"""Checks for the example's custom selection and deviance-based blending."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from ins_gbm import ModelData, OneHotEncoder
from ins_gbm.evaluation.metrics import compute_metrics


spec = importlib.util.spec_from_file_location(
    "model_search_template", Path(__file__).parents[1] / "examples/model_search_template.py"
)
template = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = template
spec.loader.exec_module(template)


@pytest.mark.parametrize("objective", ["poisson", "gamma"])
def test_blend_deviance_matches_library(objective):
    data = ModelData(
        features=pl.DataFrame({"x": [1., 2., 3., 4.]}),
        target=pl.Series([0., 2., 3., 1.] if objective == "poisson" else [1., 2., 3., 1.]),
        exposure=pl.Series([0.2, 1., 0.5, 2.]), weight=pl.Series([2., 1., 0.5, 3.]),
        feature_names=["x"], objective=objective,
    )
    prediction = np.array([0.5, 1.5, 2., 3.])
    for exposure in (data.exposure, None):
        case = replace(data, exposure=exposure)
        expected = compute_metrics(objective=objective, actual=case.target,
                                   predicted=pl.Series(prediction), exposure=exposure, weight=case.weight)
        score = expected.filter(pl.col("metric") == f"{objective}_deviance")["value"][0]
        assert template.deviance(case, prediction) == pytest.approx(score)


@pytest.mark.parametrize("objective", ["poisson", "gamma"])
def test_blend_search_includes_baseline_and_improving_mixture(objective):
    data = ModelData(features=pl.DataFrame({"x": [0., 1.]}), target=pl.Series([2., 2.]),
                     feature_names=["x"], objective=objective)
    predictions = np.array([[1., 3.], [3., 1.]])
    options = template.blend_options(data, predictions)
    assert any(np.array_equal(weights, [1, 0]) for _, weights in options)
    for _, weights in options:
        assert np.all(weights >= 0)
        assert weights.sum() == pytest.approx(1)
    best = min(options, key=lambda option: template.deviance(data, predictions @ option[1]))
    assert template.deviance(data, predictions @ best[1]) == pytest.approx(0, abs=1e-8)


def test_encoded_selection_can_drop_category_levels_and_existing_features(monkeypatch):
    data = ModelData(
        features=pl.DataFrame({"existing": ["a", "b", "a"], "new": ["x", "y", "z"], "noise": [1., 2., 3.]}),
        target=pl.Series([1., 2., 1.]), feature_names=["existing", "new", "noise"], objective="poisson",
    )
    encoder = OneHotEncoder().fit(data.features, data.schema)
    encoded = data.with_features(encoder.transform(data.features))

    class RankingModel:
        def capabilities(self):
            return SimpleNamespace(supports_feature_importance=True)

        def fit(self, data, params):
            self.names = data.feature_names
            self.params = params
            self.framework = "lightgbm"
            return self

        def feature_importance(self, kind):
            # Keep only one level of 'new'; even 'existing' can disappear.
            return pl.DataFrame({"feature": self.names, "importance": [
                {"noise": 10., "new__y": 9., "existing__a": 3.}.get(name, 0.)
                for name in self.names
            ]})

    monkeypatch.setattr(template, "LightGBMModel", RankingModel)
    config = json.loads((Path(__file__).parents[1] / "examples/model_search_config.json").read_text())
    config["selection_encoded_caps"] = [4, 2]
    candidate = {"family": "lightgbm", "strategy": "encoded_selection"}
    selector = template.make_recipe(config, candidate, data, 1, 42).selection
    result = selector.fit(encoded)
    assert set(result.selected_features()) == {"noise", "new__y"}
    assert [len(stage.selected_feature_names) for stage in result.stage_results()] == [4, 2]
    config["selection_encoded_caps"] = [100]
    selector = template.make_recipe(config, candidate, data, 1, 42).selection
    assert selector.fit(encoded).selected_features() == encoded.feature_names


@pytest.mark.parametrize("caps", [[], [0], [-1], [50, 100], [1.5], [True]])
def test_invalid_encoded_caps_are_rejected(caps):
    config = json.loads((Path(__file__).parents[1] / "examples/model_search_config.json").read_text())
    config["selection_encoded_caps"] = caps
    with pytest.raises(ValueError, match="selection_encoded_caps"):
        template.validate_config(config)


def test_all_challengers_start_from_full_pool():
    config = json.loads((Path(__file__).parents[1] / "examples/model_search_config.json").read_text())
    candidates = template.candidate_list(config)
    challengers = [c for c in candidates if c["strategy"] == "encoded_selection"]
    assert {c["family"] for c in challengers} == set(template.FAMILIES)
    assert all(c["features"] == config["candidate_features"] for c in challengers)
    assert len([c for c in candidates if c["strategy"] == "existing"]) == 1
