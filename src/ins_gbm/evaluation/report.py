"""EvaluationReport: single-model and comparison reporting."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from itertools import combinations

import polars as pl

from ins_gbm.data.model_data import ModelData
from ins_gbm.models.base import FittedModel
from ins_gbm.evaluation import metrics as _m
from ins_gbm.evaluation import plots as _p


@dataclass
class EvaluationReport:
    fitted_model: FittedModel
    test_data: ModelData
    train_data: ModelData
    _comparison_models: Optional[dict[str, tuple[FittedModel, ModelData, ModelData]]] = None

    # ── Single-model API ──────────────────────────────────────────────────────

    def metrics(self) -> pl.DataFrame:
        if self._comparison_models is not None:
            return self._comparison_metrics()
        return self._single_metrics()

    def _single_metrics(self) -> pl.DataFrame:
        return _m.compute_metrics(
            objective=self.fitted_model.objective,
            actual=self.test_data.target,
            predicted=self.fitted_model.predict(self.test_data, prediction_type="response"),
            exposure=self.test_data.exposure,
            weight=self.test_data.weight,
        )

    def plot_lift(self, output_path: Optional[str] = None):
        actual = self.test_data.target
        predicted = self.fitted_model.predict(self.test_data, prediction_type="response")
        return _p.plot_lift(actual, predicted,
                            weights=self.test_data.exposure if self.test_data.exposure is not None else self.test_data.weight,
                            output_path=output_path)

    def plot_ave(self, output_path: Optional[str] = None):
        actual = self.test_data.target
        predicted = self.fitted_model.predict(self.test_data, prediction_type="response")
        return _p.plot_ave(actual, predicted,
                           weights=self.test_data.exposure if self.test_data.exposure is not None else self.test_data.weight,
                           output_path=output_path)

    def plot_calibration(self, output_path: Optional[str] = None):
        actual = self.test_data.target
        predicted = self.fitted_model.predict(self.test_data, prediction_type="response")
        return _p.plot_calibration(actual, predicted, output_path=output_path)

    def plot_feature_importance(self, output_path: Optional[str] = None):
        return _p.plot_feature_importance(self.fitted_model.feature_importance(),
                                          output_path=output_path)

    def export(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        if self._comparison_models is not None:
            self._export_comparison(output_dir)
            return
        self.metrics().write_csv(os.path.join(output_dir, "metrics.csv"))
        self.plot_lift(output_path=os.path.join(output_dir, "lift.png"))
        self.plot_ave(output_path=os.path.join(output_dir, "ave.png"))
        self.plot_calibration(output_path=os.path.join(output_dir, "calibration.png"))
        self.plot_feature_importance(
            output_path=os.path.join(output_dir, "feature_importance.png"))

    # ── Comparison mode ───────────────────────────────────────────────────────

    @classmethod
    def compare(
        cls,
        models: dict[str, tuple[FittedModel, ModelData, ModelData]],
        test_data: ModelData,
    ) -> "EvaluationReport":
        # Use the first model as the primary; store all for comparison
        first_name = next(iter(models))
        first_fitted, first_train, first_test = models[first_name]
        obj = cls(fitted_model=first_fitted, test_data=first_test, train_data=first_train)
        obj._comparison_models = models
        return obj

    def _comparison_metrics(self) -> pl.DataFrame:
        rows = []
        for name, (fitted, train, test) in self._comparison_models.items():
            sub = EvaluationReport(fitted_model=fitted, test_data=test, train_data=train)
            for row in sub._single_metrics().iter_rows(named=True):
                rows.append({"model": name, **row})
        return pl.DataFrame(rows)

    def _export_comparison(self, output_dir: str) -> None:
        self._comparison_metrics().write_csv(os.path.join(output_dir, "metrics.csv"))
        # Double-lift charts for all pairs
        model_items = list(self._comparison_models.items())
        for (name_a, (fitted_a, _, test_a)), (name_b, (fitted_b, _, test_b)) in \
                combinations(model_items, 2):
            actual = test_a.target
            pred_a = fitted_a.predict(test_a, prediction_type="response")
            pred_b = fitted_b.predict(test_b, prediction_type="response")
            path = os.path.join(output_dir, f"double_lift_{name_a}_vs_{name_b}.png")
            _p.plot_double_lift(actual, pred_a, pred_b,
                                labels=(name_a, name_b),
                                output_path=path)
