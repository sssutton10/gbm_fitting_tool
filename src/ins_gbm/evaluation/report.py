"""EvaluationReport: single-model and comparison reporting."""
from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
import polars as pl

from ins_gbm.data.model_data import ModelData
from ins_gbm.evaluation import metrics as _m
from ins_gbm.evaluation import plots as _p
from ins_gbm.models.base import FittedModel


@dataclass
class EvaluationReport:
    fitted_model: FittedModel
    evaluation_data: ModelData
    train_data: Optional[ModelData] = None
    _comparison_models: Optional[dict[str, tuple[FittedModel, ModelData, ModelData]]] = None
    comparison_predictions: Optional[dict[str, pl.Series]] = None

    @property
    def is_comparison_mode(self) -> bool:
        return self._comparison_models is not None

    def metrics(self) -> pl.DataFrame:
        if self._comparison_models is not None:
            return self._comparison_metrics()
        if self.comparison_predictions is not None:
            return self._predictions_comparison_metrics()
        return self._single_metrics()

    def _single_metrics(self) -> pl.DataFrame:
        return _m.compute_metrics(
            objective=self.fitted_model.objective,
            actual=self.evaluation_data.target,
            predicted=self.fitted_model.predict(self.evaluation_data, prediction_type="response"),
            exposure=self.evaluation_data.exposure,
            weight=self.evaluation_data.weight,
        )

    def _predictions_comparison_metrics(self) -> pl.DataFrame:
        rows = []
        for row in self._single_metrics().iter_rows(named=True):
            rows.append({"model": "GBM", **row})
        for name, preds in self.comparison_predictions.items():
            for row in _m.compute_metrics(
                objective=self.fitted_model.objective,
                actual=self.evaluation_data.target,
                predicted=preds,
                exposure=self.evaluation_data.exposure,
                weight=self.evaluation_data.weight,
            ).iter_rows(named=True):
                rows.append({"model": name, **row})
            score = self._prediction_double_lift_score(name)
            for model in ("GBM", name):
                rows.append({
                    "model": model,
                    "metric": "double_lift_score",
                    "value": score,
                })
        return pl.DataFrame(rows)

    def plot_double_lift(
        self,
        name: Optional[str] = None,
        output_path: Optional[str] = None,
        *,
        other_name: Optional[str] = None,
        n_bins: int = 10,
    ):
        """Plot double lift for a benchmark or a pair of named models.

        In benchmark mode, pass the benchmark ``name``. In named-model
        comparison mode, pass ``name`` and ``other_name``; when exactly two
        models are present, both names may be omitted.
        """
        if self._comparison_models is not None:
            # Accept ``plot_double_lift("a", "b")`` while preserving the
            # historical benchmark-mode positional output_path argument.
            if (
                other_name is None
                and output_path in self._comparison_models
            ):
                other_name = output_path
                output_path = None
            name, other_name = self._resolve_comparison_names(name, other_name)
            fitted_a, _, evaluation_a = self._comparison_models[name]
            fitted_b, _, evaluation_b = self._comparison_models[other_name]
            self._validate_comparable_evaluations(evaluation_a, evaluation_b)
            predicted_a = fitted_a.predict(
                evaluation_a, prediction_type="response"
            )
            predicted_b = fitted_b.predict(
                evaluation_b, prediction_type="response"
            )
            actual, predicted_a, predicted_b, weights = (
                _m._double_lift_metric_inputs(
                    fitted_a.objective,
                    evaluation_a.target,
                    predicted_a,
                    predicted_b,
                    evaluation_a.exposure,
                    evaluation_a.weight,
                )
            )
            return _p.plot_double_lift(
                actual,
                predicted_a,
                predicted_b,
                weights=weights,
                n_bins=min(n_bins, len(actual)),
                labels=(name, other_name),
                output_path=output_path,
            )

        if self.comparison_predictions is None or name not in self.comparison_predictions:
            raise KeyError(
                f"No comparison prediction named {name!r}. "
                f"Available: {list(self.comparison_predictions or {})}"
            )
        gbm_preds = self.fitted_model.predict(self.evaluation_data, prediction_type="response")
        actual, gbm_preds, benchmark_preds, weights = (
            _m._double_lift_metric_inputs(
                self.fitted_model.objective,
                self.evaluation_data.target,
                gbm_preds,
                self.comparison_predictions[name],
                self.evaluation_data.exposure,
                self.evaluation_data.weight,
            )
        )
        return _p.plot_double_lift(
            actual,
            gbm_preds,
            benchmark_preds,
            weights=weights,
            n_bins=min(n_bins, len(actual)),
            labels=("GBM", name),
            output_path=output_path,
        )

    def double_lift_score(
        self,
        name: Optional[str] = None,
        *,
        other_name: Optional[str] = None,
        n_bins: int = 10,
        deviation: str = "absolute",
    ) -> float:
        """Return the signed score for a benchmark or named-model pair."""
        if self._comparison_models is None:
            if (
                self.comparison_predictions is None
                or name not in self.comparison_predictions
            ):
                raise KeyError(
                    f"No comparison prediction named {name!r}. "
                    f"Available: {list(self.comparison_predictions or {})}"
                )
            return self._prediction_double_lift_score(
                name,
                n_bins=n_bins,
                deviation=deviation,
            )

        name, other_name = self._resolve_comparison_names(name, other_name)
        fitted_a, _, evaluation_a = self._comparison_models[name]
        fitted_b, _, evaluation_b = self._comparison_models[other_name]
        self._validate_comparable_evaluations(evaluation_a, evaluation_b)
        predicted_a = fitted_a.predict(evaluation_a, prediction_type="response")
        predicted_b = fitted_b.predict(evaluation_b, prediction_type="response")
        actual, predicted_a, predicted_b, weights = (
            _m._double_lift_metric_inputs(
                fitted_a.objective,
                evaluation_a.target,
                predicted_a,
                predicted_b,
                evaluation_a.exposure,
                evaluation_a.weight,
            )
        )
        table = _m.double_lift_table(
            actual,
            predicted_a,
            predicted_b,
            weights=weights,
            n_bins=min(n_bins, len(actual)),
        )
        return _m.double_lift_score(table, deviation=deviation)

    def plot_lift(self, output_path: Optional[str] = None):
        return _p.plot_lift(
            self.evaluation_data.target,
            self.fitted_model.predict(self.evaluation_data, prediction_type="response"),
            weights=(
                self.evaluation_data.exposure
                if self.evaluation_data.exposure is not None
                else self.evaluation_data.weight
            ),
            output_path=output_path,
        )

    def plot_ave(self, output_path: Optional[str] = None):
        return _p.plot_ave(
            self.evaluation_data.target,
            self.fitted_model.predict(self.evaluation_data, prediction_type="response"),
            weights=(
                self.evaluation_data.exposure
                if self.evaluation_data.exposure is not None
                else self.evaluation_data.weight
            ),
            output_path=output_path,
        )

    def plot_calibration(self, output_path: Optional[str] = None):
        return _p.plot_calibration(
            self.evaluation_data.target,
            self.fitted_model.predict(self.evaluation_data, prediction_type="response"),
            output_path=output_path,
        )

    def plot_feature_importance(self, output_path: Optional[str] = None):
        return _p.plot_feature_importance(self.fitted_model.feature_importance(), output_path=output_path)

    def export(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        if self._comparison_models is not None:
            self._export_comparison(output_dir)
            return
        self.metrics().write_csv(os.path.join(output_dir, "metrics.csv"))
        self.plot_lift(output_path=os.path.join(output_dir, "lift.png"))
        self.plot_ave(output_path=os.path.join(output_dir, "ave.png"))
        self.plot_calibration(output_path=os.path.join(output_dir, "calibration.png"))
        self.plot_feature_importance(output_path=os.path.join(output_dir, "feature_importance.png"))
        if self.comparison_predictions:
            for name in self.comparison_predictions:
                self.plot_double_lift(name, output_path=os.path.join(output_dir, f"double_lift_GBM_vs_{name}.png"))

    @classmethod
    def compare(
        cls,
        models: dict[str, tuple[FittedModel, ModelData, ModelData]],
    ) -> "EvaluationReport":
        """Compare fitted models paired with their transformed evaluation data."""
        first_name = next(iter(models))
        first_fitted, first_train, first_evaluation = models[first_name]
        obj = cls(
            fitted_model=first_fitted,
            evaluation_data=first_evaluation,
            train_data=first_train,
        )
        obj._comparison_models = models
        return obj

    def _comparison_metrics(self) -> pl.DataFrame:
        rows = []
        for name, (fitted, train, evaluation) in self._comparison_models.items():
            sub = EvaluationReport(
                fitted_model=fitted,
                evaluation_data=evaluation,
                train_data=train,
            )
            for row in sub._single_metrics().iter_rows(named=True):
                rows.append({"model": name, **row})
        for name_a, name_b in combinations(self._comparison_models, 2):
            score = self.double_lift_score(name_a, other_name=name_b)
            for model in (name_a, name_b):
                rows.append({
                    "model": model,
                    "metric": "double_lift_score",
                    "value": score,
                })
        return pl.DataFrame(rows)

    def _export_comparison(self, output_dir: str) -> None:
        self._comparison_metrics().write_csv(os.path.join(output_dir, "metrics.csv"))
        model_items = list(self._comparison_models.items())
        for (name_a, _), (name_b, _) in combinations(model_items, 2):
            self.plot_double_lift(
                name_a,
                other_name=name_b,
                output_path=os.path.join(output_dir, f"double_lift_{name_a}_vs_{name_b}.png"),
            )

    def _prediction_double_lift_score(
        self,
        name: str,
        *,
        n_bins: int = 10,
        deviation: str = "absolute",
    ) -> float:
        gbm_preds = self.fitted_model.predict(
            self.evaluation_data, prediction_type="response"
        )
        actual, gbm_preds, benchmark_preds, weights = (
            _m._double_lift_metric_inputs(
                self.fitted_model.objective,
                self.evaluation_data.target,
                gbm_preds,
                self.comparison_predictions[name],
                self.evaluation_data.exposure,
                self.evaluation_data.weight,
            )
        )
        table = _m.double_lift_table(
            actual,
            gbm_preds,
            benchmark_preds,
            weights=weights,
            n_bins=min(n_bins, len(actual)),
        )
        return _m.double_lift_score(table, deviation=deviation)

    def _resolve_comparison_names(
        self,
        name: Optional[str],
        other_name: Optional[str],
    ) -> tuple[str, str]:
        available = list(self._comparison_models)
        if name is None and other_name is None and len(available) == 2:
            return available[0], available[1]
        if name is not None and other_name is None and len(available) == 2:
            other_name = next(candidate for candidate in available if candidate != name)
        if (
            name not in self._comparison_models
            or other_name not in self._comparison_models
            or name == other_name
        ):
            raise KeyError(
                "Choose two distinct comparison model names. "
                f"Available: {available}"
            )
        return name, other_name

    @staticmethod
    def _validate_comparable_evaluations(
        evaluation_a: ModelData,
        evaluation_b: ModelData,
    ) -> None:
        if evaluation_a.objective != evaluation_b.objective:
            raise ValueError("Compared models must have the same objective")
        actual_a = evaluation_a.target.to_numpy()
        actual_b = evaluation_b.target.to_numpy()
        if (
            len(actual_a) != len(actual_b)
            or not np.array_equal(actual_a, actual_b, equal_nan=True)
        ):
            raise ValueError(
                "Compared models must use aligned evaluation rows and targets"
            )
        for field in ("exposure", "weight"):
            values_a = getattr(evaluation_a, field)
            values_b = getattr(evaluation_b, field)
            if (values_a is None) != (values_b is None):
                raise ValueError(
                    f"Compared models must use the same {field} values"
                )
            if values_a is not None and not np.array_equal(
                values_a.to_numpy(),
                values_b.to_numpy(),
                equal_nan=True,
            ):
                raise ValueError(
                    f"Compared models must use the same {field} values"
                )
