import polars as pl
import pytest

from ins_gbm.data.model_data import ModelData
from ins_gbm.data.schema import infer_schema
from ins_gbm.models.random_forest import RandomForestModel
from ins_gbm.preprocessing.encoder import OneHotEncoder
from ins_gbm.preprocessing.pca import PCAReducer
from ins_gbm.preprocessing.pls import PLSReducer
from ins_gbm.preprocessing.steps import PreprocessingStep


def _raw_data(frame: pl.DataFrame) -> ModelData:
    feature_names = ["x1", "x2", "x3"]
    return ModelData(
        features=frame.select(feature_names),
        target=frame["claim_count"],
        exposure=frame["exposure"],
        weight=None,
        feature_names=feature_names,
        schema=infer_schema(frame, feature_names),
        objective="poisson",
    ).validate()


def test_direct_fit_builds_encoder_matrix_and_predicts_from_raw_data(poisson_raw):
    data = _raw_data(poisson_raw)
    original_columns = list(data.features.columns)

    fitted = RandomForestModel(objective="poisson").fit(
        data,
        params={"n_estimators": 5},
        feature_names=["x1", "x2"],
        encoder=OneHotEncoder(),
    )

    assert data.features.columns == original_columns
    assert fitted.transform_chain.input_feature_names == ["x1", "x2"]
    assert fitted.feature_names[0] == "x1"
    assert any(name.startswith("x2__") for name in fitted.feature_names)
    assert fitted.predict(data).len() == data.n_rows


def test_direct_fit_supports_multiple_targeted_preprocessors(poisson_raw):
    data = _raw_data(poisson_raw)
    fitted = RandomForestModel(objective="poisson").fit(
        data,
        params={"n_estimators": 5},
        encoder=OneHotEncoder(),
        preprocessing=[
            PreprocessingStep(
                name="x1_pca",
                preprocessor=PCAReducer(n_components=1),
                feature_names=["x1"],
            ),
            PreprocessingStep(
                name="x3_pls",
                preprocessor=PLSReducer(n_components=1),
                feature_names=["x3"],
            ),
        ],
    )

    assert fitted.feature_names[:2] == ["x1_pca__pca_1", "x3_pls__pls_1"]
    assert any(name.startswith("x2__") for name in fitted.feature_names)
    assert len(fitted.transform_chain.preprocessors) == 2
    assert fitted.predict(data, prediction_type="rate").len() == data.n_rows


def test_direct_fit_encoder_requires_schema(poisson_raw):
    data = _raw_data(poisson_raw)
    data.schema = None

    with pytest.raises(ValueError, match="encoder requires ModelData.schema"):
        RandomForestModel(objective="poisson").fit(
            data,
            params={"n_estimators": 1},
            encoder=OneHotEncoder(),
        )


@pytest.mark.parametrize(
    ("dependency", "model_path", "class_name", "params"),
    [
        (
            "lightgbm",
            "ins_gbm.models.lightgbm",
            "LightGBMModel",
            {"n_estimators": 3, "verbose": -1},
        ),
        (
            "xgboost",
            "ins_gbm.models.xgboost",
            "XGBoostModel",
            {"n_estimators": 3},
        ),
        (
            "catboost",
            "ins_gbm.models.catboost",
            "CatBoostModel",
            {"iterations": 3},
        ),
    ],
)
def test_gbm_wrappers_fit_and_predict_raw_encoded_data(
    poisson_raw, dependency, model_path, class_name, params
):
    import importlib

    pytest.importorskip(dependency)
    model_class = getattr(importlib.import_module(model_path), class_name)
    data = _raw_data(poisson_raw)

    fitted = model_class(objective="poisson").fit(
        data,
        params=params,
        encoder=OneHotEncoder(),
    )

    assert any(name.startswith("x2__") for name in fitted.feature_names)
    assert fitted.predict(data).len() == data.n_rows
