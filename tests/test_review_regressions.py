from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from ins_gbm import CVConfig, LightGBMModel, ModelData, ModelRecipe, RandomForestModel
from ins_gbm.ensemble.stacking import StackingEnsemble


def poisson_data(n=80):
    return ModelData(
        features=pl.DataFrame({"x": np.arange(n, dtype=float)}),
        target=pl.Series("target", np.tile([0.0, 0.0, 0.0, 1.0], n // 4)),
        exposure=pl.Series("exposure", np.ones(n)),
        feature_names=["x"],
        objective="poisson",
    )


def test_lightgbm_wrapper_response_matches_native_response():
    data = poisson_data()
    fitted = LightGBMModel().fit(
        data, params={"n_estimators": 20, "num_threads": 1, "min_data_in_leaf": 10}
    )
    native = fitted.model.predict(data.features.to_numpy())
    np.testing.assert_allclose(fitted.predict(data).to_numpy(), native)
    np.testing.assert_allclose(
        np.exp(fitted.predict(data, "link").to_numpy()), native
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("target", pl.Series([None] * 80, dtype=pl.Float64), "target"),
        ("weight", pl.Series([-1.0] * 80), "weight"),
        ("offset", pl.Series([float("nan")] * 80), "offset"),
    ],
)
def test_invalid_row_fields_are_rejected(field, value, match):
    with pytest.raises(ValueError, match=match):
        replace(poisson_data(), **{field: value}).validate()


def test_recipe_api_uses_predefined_model_data_folds():
    data = replace(poisson_data(), cv_fold=pl.Series([0] * 40 + [1] * 40))
    result = ModelRecipe(
        RandomForestModel(), params={"n_estimators": 2, "max_depth": 1}
    ).cross_validate(data, cv=CVConfig(folds="auto"))
    assert result.fold_metrics["fold"].unique().sort().to_list() == [0, 1]


def test_stacking_refits_with_effective_model_parameters():
    class RecordingRF(RandomForestModel):
        calls = []

        def fit(self, data, params=None, **kwargs):
            self.calls.append(dict(params or {}))
            return super().fit(data, params=params, **kwargs)

    model = RecordingRF()
    fitted = ModelRecipe(
        model, params={"n_estimators": 2, "max_depth": 1}
    ).fit(poisson_data())
    model.calls.clear()
    StackingEnsemble(cv_folds=2).fit([fitted])
    assert len(model.calls) == 2
    assert all(call["n_estimators"] == 2 for call in model.calls)


def test_dataframe_prediction_and_save_conveniences(tmp_path):
    data = poisson_data()
    fitted = ModelRecipe(
        RandomForestModel(), params={"n_estimators": 2, "max_depth": 1}
    ).fit(data)
    prediction = fitted.predict(data.features, exposure=data.exposure)
    assert len(prediction) == data.n_rows
    fitted.save(str(tmp_path / "model"))
    from ins_gbm import load_model
    restored = load_model(str(tmp_path / "model"))
    np.testing.assert_allclose(restored.predict(data).to_numpy(), prediction.to_numpy())
    with pytest.raises(ValueError, match="offset"):
        fitted.predict(
            data.features,
            exposure=data.exposure,
            offset=pl.Series([float("nan")] * data.n_rows),
        )
