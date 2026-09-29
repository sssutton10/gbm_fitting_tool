import pytest

from ins_gbm.models.base import FittedModel, ModelCapabilities


def test_model_capabilities_fields():
    """Verify model capabilities fields."""
    caps = ModelCapabilities(
        supports_poisson=True,
        supports_gamma=True,
        supports_offset=True,
        supports_sample_weight=True,
        supports_feature_importance=True,
    )
    assert caps.supports_poisson is True
    assert caps.supports_offset is True


def test_model_capabilities_frozen():
    """Verify model capabilities frozen."""
    caps = ModelCapabilities(
        supports_poisson=True,
        supports_gamma=False,
        supports_offset=False,
        supports_sample_weight=True,
        supports_feature_importance=True,
    )
    with pytest.raises((AttributeError, TypeError)):
        caps.supports_poisson = False


def test_fitted_model_predict_rate_invalid_for_gamma(gamma_raw):
    """predict with prediction_type='rate' must raise for Gamma.

    Args:
        gamma_raw (object): The gamma raw.
    """
    # Build a minimal FittedModel stub to test the validation path
    import polars as pl

    from ins_gbm.data.model_data import ModelData

    class _StubModel:
        """Stubmodel."""

    data = ModelData(
        features=gamma_raw.select(["x1"]),
        target=gamma_raw["severity"],
        exposure=None,
        weight=gamma_raw["weight"],
        feature_names=["x1"],
        objective="gamma",
    )

    fitted = FittedModel(
        model=_StubModel(),
        params={},
        framework="stub",
        objective="gamma",
        feature_names=["x1"],
        predict_fn=lambda d, pt: pl.Series([1.0] * d.n_rows),
        importance_fn=lambda: pl.DataFrame({"feature": ["x1"], "importance": [1.0]}),
    )
    with pytest.raises(ValueError, match="(?i)rate.*gamma"):
        fitted.predict(data, prediction_type="rate")
