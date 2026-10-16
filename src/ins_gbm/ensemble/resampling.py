"""Whole-pipeline seed averaging and bootstrap aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np

from ins_gbm.data.model_data import slice_model_data
from ins_gbm.ensemble._utils import _refit_pipeline
from ins_gbm.ensemble.blending import FittedBlendingEnsemble

if TYPE_CHECKING:
    from ins_gbm.pipeline import FittedPipeline


def _member_params(params: dict, framework: str, seed: int) -> dict:
    """Override seed aliases, including LightGBM's independently seeded components."""
    params = dict(params)
    if framework == "lightgbm":
        for name in ("random_state", "random_seed", "data_seed"):
            params.pop(name, None)
        for name in (
            "seed", "data_random_seed", "feature_fraction_seed", "bagging_seed",
            "drop_seed", "extra_seed", "objective_seed",
        ):
            params[name] = seed
    elif framework == "xgboost":
        params.pop("random_state", None)
        params["seed"] = seed
    elif framework == "catboost":
        params.pop("random_state", None)
        params["random_seed"] = seed
    elif framework == "random_forest":
        params["random_state"] = seed
    else:
        raise ValueError(f"Seed averaging/bagging does not support framework {framework!r}")
    return params


@dataclass
class ResamplingEnsemble:
    """Refit one fitted pipeline with new seeds or sampled training rows.

    Model parameters are reused without tuning; transforms and configured
    selection are fitted independently. Members are averaged on the response
    scale: rates for Poisson rate targets with exposure omitted (the default),
    severity for Gamma. Optional Poisson count targets with explicit exposure
    produce expected counts.

    Args:
        method: "seed_average" uses every row; "bagging" samples rows uniformly.
        n_members: Positive number of members, fitted sequentially.
        seed: Nonnegative master seed for reproducible sampling and model seeds.
        sample_fraction: Fraction of rows drawn per bag, in (0, 1].
        bootstrap: Sample with replacement; False samples without replacement.
    """

    method: str = "seed_average"
    n_members: int = 5
    seed: int = 42
    sample_fraction: float = 1.0
    bootstrap: bool = True

    def fit(
        self, fitted_pipelines: list[FittedPipeline]
    ) -> tuple[FittedBlendingEnsemble, dict]:
        """Return an equal-weight fitted ensemble and reproducibility settings."""
        if len(fitted_pipelines) != 1:
            raise ValueError(f"{self.method} requires exactly one fitted template pipeline")
        template = fitted_pipelines[0]
        data = template._require_raw_train_data()
        samples, settings = self.prepare_samples(data)
        member_seeds = settings["member_seeds"]
        framework = template.fitted_model.framework
        # Fail unsupported backends before starting member fits.
        _member_params(template.fitted_model.params, framework, member_seeds[0])
        members = []
        for index, (seed, indices) in enumerate(samples):
            try:
                sample = slice_model_data(data, indices)
                params = _member_params(template.fitted_model.params, framework, seed)
                member = _refit_pipeline(template, sample, params)
                member.metadata.random_seeds["member"] = seed
                members.append(member)
            except Exception as exc:
                raise RuntimeError(
                    f"{self.method} member {index + 1} (seed={seed}) failed: {exc}"
                ) from exc
        fitted = FittedBlendingEnsemble(
            weights=[1.0 / self.n_members] * self.n_members,
            fitted_pipelines=members,
        )
        return fitted, settings

    def prepare_samples(self, data):
        """Validate settings and return ordered sample indices without fitting."""
        if self.method not in {"seed_average", "bagging"}:
            raise ValueError("method must be 'seed_average' or 'bagging'")
        if (
            isinstance(self.n_members, bool)
            or not isinstance(self.n_members, Integral)
            or self.n_members < 1
        ):
            raise ValueError("n_members must be a positive integer")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        if self.method == "bagging":
            if (
                isinstance(self.sample_fraction, bool)
                or not isinstance(self.sample_fraction, Real)
                or not 0 < self.sample_fraction <= 1
            ):
                raise ValueError("sample_fraction must be in (0, 1]")
            if not isinstance(self.bootstrap, bool):
                raise ValueError("bootstrap must be a bool")
        data.validate(require_multiple_folds=False)
        sample_size = (
            int(np.floor(self.sample_fraction * data.n_rows))
            if self.method == "bagging" else data.n_rows
        )
        if sample_size < 1:
            raise ValueError("sample_fraction must select at least one row")

        # Separate sampling and model RNG streams so the two methods share seeds.
        model_rng, sampling_rng = [
            np.random.default_rng(child)
            for child in np.random.SeedSequence(self.seed).spawn(2)
        ]
        member_seeds = []
        used = set()
        while len(member_seeds) < self.n_members:
            seed = int(model_rng.integers(0, np.iinfo(np.int32).max))
            if seed not in used:
                member_seeds.append(seed)
                used.add(seed)
        n_rows = data.n_rows
        # Generate one sample at a time, including during CV preflight, so large
        # portfolios do not retain n_members copies of row-index arrays.
        samples = (
            (seed, sampling_rng.choice(n_rows, sample_size, replace=self.bootstrap)
             if self.method == "bagging" else np.arange(n_rows))
            for seed in member_seeds
        )
        settings = {
            "method": self.method,
            "n_members": int(self.n_members),
            "seed": int(self.seed),
            "member_seeds": member_seeds,
        }
        if self.method == "bagging":
            settings.update(
                sample_fraction=float(self.sample_fraction), bootstrap=self.bootstrap
            )
        return samples, settings
