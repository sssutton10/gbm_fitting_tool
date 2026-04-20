# GBM Fitting Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Polars-native Python library for fitting insurance frequency (Poisson) and severity (Gamma) GBM models with variable selection, dimensionality reduction, hyperparameter tuning, ensembles, and actuarial evaluation.

**Architecture:** Layered library. `ModelData` is the central data container. A `ModelRecipe` bundles encoder → selector → preprocessor → model → tuner. `ModelPipeline` executes the full train/tune/fit/evaluate cycle and returns a `FittedPipeline`. `EnsemblePipeline` composes `FittedPipeline` objects into stacking or blending ensembles. All public APIs accept and return Polars; numpy conversion happens only at the model boundary.

**Tech Stack:** Python 3.11+, Polars, LightGBM, XGBoost, CatBoost, scikit-learn, Optuna, matplotlib, SHAP, umap-learn, scipy, pytest

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/gbm_fitting/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`
- Create: `README.md`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "gbm_fitting"
version = "0.1.0"
description = "Polars-native GBM fitting library for insurance frequency and severity models"
requires-python = ">=3.11"
dependencies = [
    "polars>=0.20",
    "numpy>=1.26",
    "scipy>=1.11",
    "scikit-learn>=1.4",
    "optuna>=3.5",
    "matplotlib>=3.8",
    "pyarrow>=14.0",
]

[project.optional-dependencies]
lightgbm = ["lightgbm>=4.2"]
xgboost = ["xgboost>=2.0"]
catboost = ["catboost>=1.2"]
explain = ["shap>=0.44"]
umap = ["umap-learn>=0.5"]
all = [
    "lightgbm>=4.2",
    "xgboost>=2.0",
    "catboost>=1.2",
    "shap>=0.44",
    "umap-learn>=0.5",
]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
```

- [ ] **Step 2: Create package skeleton**

```python
# src/gbm_fitting/__init__.py
"""GBM Fitting: Polars-native GBM library for insurance modeling."""
__version__ = "0.1.0"
```

- [ ] **Step 3: Create .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
htmlcov/
*.parquet
output/
.venv/
venv/
```

- [ ] **Step 4: Create test fixtures**

```python
# tests/conftest.py
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
```

- [ ] **Step 5: Create README.md**

```markdown
# GBM Fitting

Polars-native Python library for insurance frequency and severity GBM modeling.

## Install

```bash
pip install -e ".[all,dev]"
```

## Run tests

```bash
pytest
```
```

- [ ] **Step 6: Install and verify**

Run: `pip install -e ".[all,dev]"`
Expected: Successful install of all extras.

Run: `pytest --collect-only`
Expected: "no tests collected" (no tests yet).

- [ ] **Step 7: Commit**

```bash
git init
git add pyproject.toml src/ tests/ .gitignore README.md
git commit -m "chore: project scaffolding and fixtures"
```

---

## Task 2: FeatureSchema and ModelData

**Files:**
- Create: `src/gbm_fitting/data/__init__.py`
- Create: `src/gbm_fitting/data/schema.py`
- Create: `src/gbm_fitting/data/model_data.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_schema.py`
- Create: `tests/data/test_model_data.py`

- [ ] **Step 1: Write failing schema test**

```python
# tests/data/test_schema.py
import polars as pl
from gbm_fitting.data.schema import FeatureSchema, infer_schema


def test_feature_schema_defaults():
    s = FeatureSchema(numeric=["x1"], categorical=["x2"])
    assert s.numeric == ["x1"]
    assert s.categorical == ["x2"]
    assert s.ordinal == []
    assert s.passthrough == []


def test_infer_schema_from_dataframe():
    df = pl.DataFrame({
        "num1": [1.0, 2.0],
        "int1": [1, 2],
        "cat1": ["a", "b"],
        "bool1": [True, False],
    })
    s = infer_schema(df, feature_cols=["num1", "int1", "cat1", "bool1"])
    assert set(s.numeric) == {"num1", "int1"}
    assert set(s.categorical) == {"cat1", "bool1"}
```

- [ ] **Step 2: Verify it fails**

Run: `pytest tests/data/test_schema.py -v`
Expected: FAIL — ImportError on `gbm_fitting.data.schema`.

- [ ] **Step 3: Implement FeatureSchema**

```python
# src/gbm_fitting/data/__init__.py
```

```python
# src/gbm_fitting/data/schema.py
from dataclasses import dataclass, field
from typing import Iterable

import polars as pl


@dataclass
class FeatureSchema:
    numeric: list[str]
    categorical: list[str]
    ordinal: list[str] = field(default_factory=list)
    passthrough: list[str] = field(default_factory=list)

    def all_features(self) -> list[str]:
        return [*self.numeric, *self.categorical, *self.ordinal, *self.passthrough]


_NUMERIC_DTYPES = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                   pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
_CATEGORICAL_DTYPES = (pl.Utf8, pl.String, pl.Categorical, pl.Enum, pl.Boolean)


def infer_schema(df: pl.DataFrame, feature_cols: Iterable[str]) -> FeatureSchema:
    numeric: list[str] = []
    categorical: list[str] = []
    for col in feature_cols:
        dtype = df.schema[col]
        if isinstance(dtype, _NUMERIC_DTYPES) or dtype in _NUMERIC_DTYPES:
            numeric.append(col)
        elif isinstance(dtype, _CATEGORICAL_DTYPES) or dtype in _CATEGORICAL_DTYPES:
            categorical.append(col)
        else:
            raise ValueError(f"Unsupported dtype {dtype} for column {col}")
    return FeatureSchema(numeric=numeric, categorical=categorical)
```

- [ ] **Step 4: Verify schema tests pass**

Run: `pytest tests/data/test_schema.py -v`
Expected: PASS

- [ ] **Step 5: Write failing ModelData tests**

```python
# tests/data/__init__.py
```

```python
# tests/data/test_model_data.py
import polars as pl
import pytest
from gbm_fitting.data.model_data import ModelData
from gbm_fitting.data.schema import FeatureSchema


def test_model_data_poisson_valid(poisson_raw):
    data = ModelData(
        features=poisson_raw.select(["x1", "x2", "x3"]),
        target=poisson_raw["claim_count"],
        exposure=poisson_raw["exposure"],
        weight=None,
        feature_names=["x1", "x2", "x3"],
        schema=FeatureSchema(numeric=["x1", "x3"], categorical=["x2"]),
        objective="poisson",
    )
    assert data.n_rows == 400
    assert data.feature_names == ["x1", "x2", "x3"]


def test_model_data_poisson_requires_exposure(poisson_raw):
    with pytest.raises(ValueError, match="exposure is required"):
        ModelData(
            features=poisson_raw.select(["x1"]),
            target=poisson_raw["claim_count"],
            exposure=None,
            weight=None,
            feature_names=["x1"],
            objective="poisson",
        ).validate()


def test_model_data_gamma_positive_target(gamma_raw):
    bad_target = gamma_raw["severity"].clone()
    bad_target[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        ModelData(
            features=gamma_raw.select(["x1"]),
            target=bad_target,
            exposure=None,
            weight=gamma_raw["weight"],
            feature_names=["x1"],
            objective="gamma",
        ).validate()


def test_model_data_row_count_mismatch(poisson_raw):
    with pytest.raises(ValueError, match="row count"):
        ModelData(
            features=poisson_raw.select(["x1"]),
            target=poisson_raw["claim_count"].head(10),
            exposure=poisson_raw["exposure"],
            weight=None,
            feature_names=["x1"],
            objective="poisson",
        ).validate()


def test_model_data_duplicate_feature_names(poisson_raw):
    with pytest.raises(ValueError, match="unique"):
        ModelData(
            features=poisson_raw.select(["x1", "x2"]),
            target=poisson_raw["claim_count"],
            exposure=poisson_raw["exposure"],
            weight=None,
            feature_names=["x1", "x1"],
            objective="poisson",
        ).validate()
```

- [ ] **Step 6: Verify it fails**

Run: `pytest tests/data/test_model_data.py -v`
Expected: FAIL — ImportError.

- [ ] **Step 7: Implement ModelData**

```python
# src/gbm_fitting/data/model_data.py
from dataclasses import dataclass, field
from typing import Literal, Optional

import polars as pl

from .schema import FeatureSchema


Objective = Literal["poisson", "gamma"]


@dataclass
class ModelData:
    features: pl.DataFrame
    target: pl.Series
    exposure: Optional[pl.Series]
    weight: Optional[pl.Series]
    feature_names: list[str]
    schema: Optional[FeatureSchema] = None
    objective: Optional[Objective] = None

    @property
    def n_rows(self) -> int:
        return self.features.height

    def validate(self) -> "ModelData":
        n = self.n_rows
        if self.target.len() != n:
            raise ValueError(f"target row count {self.target.len()} != features {n}")
        if self.exposure is not None and self.exposure.len() != n:
            raise ValueError(f"exposure row count {self.exposure.len()} != features {n}")
        if self.weight is not None and self.weight.len() != n:
            raise ValueError(f"weight row count {self.weight.len()} != features {n}")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        missing = [f for f in self.feature_names if f not in self.features.columns]
        if missing:
            raise ValueError(f"features missing columns: {missing}")
        if self.exposure is not None:
            if self.exposure.null_count() > 0:
                raise ValueError("exposure must be non-null")
            if (self.exposure <= 0).any():
                raise ValueError("exposure must be positive")
        if self.objective == "poisson":
            if self.exposure is None:
                raise ValueError("exposure is required for Poisson objective")
            if (self.target < 0).any():
                raise ValueError("Poisson target must be non-negative")
        if self.objective == "gamma":
            if (self.target <= 0).any():
                raise ValueError("Gamma target must be strictly positive")
        return self

    def with_features(self, features: pl.DataFrame) -> "ModelData":
        """Return a copy with swapped features (e.g., after encoding)."""
        from dataclasses import replace
        return replace(self, features=features, feature_names=features.columns)
```

- [ ] **Step 8: Run tests and verify pass**

Run: `pytest tests/data/ -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/gbm_fitting/data/ tests/data/
git commit -m "feat(data): add FeatureSchema and ModelData with validation"
```

---

## Task 3: Parquet Loader and Train/Test Splitter

**Files:**
- Create: `src/gbm_fitting/data/loader.py`
- Create: `src/gbm_fitting/data/splitter.py`
- Create: `tests/data/test_loader.py`
- Create: `tests/data/test_splitter.py`

- [ ] **Step 1: Write failing loader tests**

```python
# tests/data/test_loader.py
from gbm_fitting.data.loader import load_model_data


def test_load_poisson_from_parquet(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x2", "x3"],
        objective="poisson",
    )
    assert data.n_rows == 400
    assert data.feature_names == ["x1", "x2", "x3"]
    assert data.exposure is not None
    assert data.objective == "poisson"
    assert data.schema is not None
    assert set(data.schema.numeric) == {"x1", "x3"}
    assert data.schema.categorical == ["x2"]


def test_load_gamma_from_parquet(gamma_parquet):
    data = load_model_data(
        path=str(gamma_parquet),
        target="severity",
        weight="weight",
        feature_cols=["x1", "x2"],
        objective="gamma",
    )
    assert data.n_rows == 300
    assert data.weight is not None
    assert data.exposure is None
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/data/test_loader.py -v`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement loader**

```python
# src/gbm_fitting/data/loader.py
from typing import Optional

import polars as pl

from .model_data import ModelData, Objective
from .schema import FeatureSchema, infer_schema


def load_model_data(
    path: str,
    target: str,
    exposure: Optional[str] = None,
    weight: Optional[str] = None,
    feature_cols: Optional[list[str]] = None,
    schema: Optional[FeatureSchema] = None,
    objective: Optional[Objective] = None,
) -> ModelData:
    df = pl.read_parquet(path)

    if feature_cols is None:
        reserved = {target}
        if exposure is not None:
            reserved.add(exposure)
        if weight is not None:
            reserved.add(weight)
        feature_cols = [c for c in df.columns if c not in reserved]

    features = df.select(feature_cols)
    target_series = df[target]
    exposure_series = df[exposure] if exposure else None
    weight_series = df[weight] if weight else None

    if schema is None:
        schema = infer_schema(df, feature_cols)

    data = ModelData(
        features=features,
        target=target_series,
        exposure=exposure_series,
        weight=weight_series,
        feature_names=list(feature_cols),
        schema=schema,
        objective=objective,
    )
    return data.validate()
```

- [ ] **Step 4: Verify loader tests pass**

Run: `pytest tests/data/test_loader.py -v`
Expected: PASS

- [ ] **Step 5: Write failing splitter tests**

```python
# tests/data/test_splitter.py
import pytest
from gbm_fitting.data.loader import load_model_data
from gbm_fitting.data.splitter import TrainTestSplit


def test_default_split_70_30(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x2", "x3"],
        objective="poisson",
    )
    splitter = TrainTestSplit(train_ratio=0.7, seed=42)
    train, test = splitter.split(data)
    assert abs(train.n_rows / data.n_rows - 0.7) < 0.02
    assert train.n_rows + test.n_rows == data.n_rows


def test_split_is_reproducible(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet),
        target="claim_count",
        exposure="exposure",
        feature_cols=["x1", "x2", "x3"],
        objective="poisson",
    )
    splitter = TrainTestSplit(train_ratio=0.7, seed=42)
    t1, _ = splitter.split(data)
    t2, _ = splitter.split(data)
    assert t1.target.to_list() == t2.target.to_list()


def test_split_invalid_ratio(poisson_parquet):
    data = load_model_data(
        path=str(poisson_parquet), target="claim_count",
        exposure="exposure", feature_cols=["x1"], objective="poisson",
    )
    with pytest.raises(ValueError, match="train_ratio"):
        TrainTestSplit(train_ratio=1.5).split(data)
```

- [ ] **Step 6: Verify failure, implement splitter**

Run: `pytest tests/data/test_splitter.py -v` → FAIL

```python
# src/gbm_fitting/data/splitter.py
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import polars as pl

from .model_data import ModelData


@dataclass
class TrainTestSplit:
    train_ratio: float = 0.7
    seed: int = 42
    group_col: Optional[str] = None

    def split(self, data: ModelData) -> tuple[ModelData, ModelData]:
        if not 0 < self.train_ratio < 1:
            raise ValueError(f"train_ratio must be in (0, 1), got {self.train_ratio}")

        rng = np.random.default_rng(self.seed)
        n = data.n_rows

        if self.group_col is not None:
            if self.group_col not in data.features.columns:
                raise ValueError(f"group_col {self.group_col} not in features")
            groups = data.features[self.group_col].to_numpy()
            unique_groups = np.unique(groups)
            rng.shuffle(unique_groups)
            n_train_groups = int(len(unique_groups) * self.train_ratio)
            train_groups = set(unique_groups[:n_train_groups])
            train_mask = np.array([g in train_groups for g in groups])
        else:
            indices = np.arange(n)
            rng.shuffle(indices)
            n_train = int(n * self.train_ratio)
            train_mask = np.zeros(n, dtype=bool)
            train_mask[indices[:n_train]] = True

        return self._subset(data, train_mask), self._subset(data, ~train_mask)

    @staticmethod
    def _subset(data: ModelData, mask: np.ndarray) -> ModelData:
        mask_series = pl.Series(mask)
        return replace(
            data,
            features=data.features.filter(mask_series),
            target=data.target.filter(mask_series),
            exposure=data.exposure.filter(mask_series) if data.exposure is not None else None,
            weight=data.weight.filter(mask_series) if data.weight is not None else None,
        )
```

- [ ] **Step 7: Run full data-layer tests**

Run: `pytest tests/data/ -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/gbm_fitting/data/ tests/data/
git commit -m "feat(data): add parquet loader and train/test splitter"
```

---
