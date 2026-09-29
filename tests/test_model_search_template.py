"""Checks for the example's custom selection and deviance-based blending."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

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


def test_protected_selection_retains_all_category_levels(monkeypatch):
    data = ModelData(
        features=pl.DataFrame({"existing": ["a", "b", "a"], "new": ["x", "y", "z"], "noise": [1., 2., 3.]}),
        target=pl.Series([1., 2., 1.]), feature_names=["existing", "new", "noise"], objective="poisson",
    )
    encoder = OneHotEncoder().fit(data.features, data.schema)
    encoded = data.with_features(encoder.transform(data.features))

    class RankingModel:
        def fit(self, data, params):
            self.names = data.feature_names
            return self

        def feature_importance(self, kind):
            # Each new categorical indicator has modest gain; their group wins.
            return pl.DataFrame({"feature": self.names, "importance": [
                10. if name == "noise" else (5. if name.startswith("new__") else 0.)
                for name in self.names
            ]})

    monkeypatch.setattr(template, "LightGBMModel", RankingModel)
    selector = template.ProtectedRawSelector(
        raw_names=data.feature_names, categorical=["existing", "new"],
        protected=["existing"], caps=[2, 1], threads=1, seed=42,
    )
    selected = selector.fit(encoded).selected_features()
    assert set(selected) == {"existing__a", "existing__b", "new__x", "new__y", "new__z"}
    with pytest.raises(ValueError, match="Ambiguous"):
        template.ProtectedRawSelector(["a", "a__b"], ["a", "a__b"], [], [1], 1, 42).owner("a__b__c")
