import numpy as np
import polars as pl
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def poisson_raw(rng):
    """Raw synthetic policy-level frequency data as a Polars DataFrame."""
    n = 400
    x1 = rng.normal(0, 1, n)
    x2 = rng.choice(["A", "B", "C"], n)
    x3 = rng.uniform(0, 1, n)
    exposure = rng.uniform(0.5, 2.0, n)
    log_rate = 0.3 * x1 - 0.2 * x3 + np.log(exposure) - 1.0
    claim_count = rng.poisson(np.exp(log_rate)).astype(float)
    return pl.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "exposure": exposure,
        "claim_count": claim_count,
    })


@pytest.fixture
def gamma_raw(rng):
    """Raw synthetic policy-level severity data (positive targets)."""
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.choice(["A", "B"], n)
    log_sev = 0.5 * x1 + 7.0
    severity = rng.gamma(shape=2.0, scale=np.exp(log_sev) / 2.0, size=n)
    weight = rng.uniform(1, 5, n)
    return pl.DataFrame({
        "x1": x1,
        "x2": x2,
        "severity": severity,
        "weight": weight,
    })


@pytest.fixture
def poisson_parquet(tmp_path, poisson_raw):
    path = tmp_path / "poisson.parquet"
    poisson_raw.write_parquet(path)
    return path


@pytest.fixture
def gamma_parquet(tmp_path, gamma_raw):
    path = tmp_path / "gamma.parquet"
    gamma_raw.write_parquet(path)
    return path
