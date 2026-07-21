"""EvaluationReport: single-model and comparison reporting."""
from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import polars as pl

from ins_gbm.data.model_data import ModelData
from ins_gbm.evaluation import metrics as _m
from ins_gbm.evaluation import plots as _p
from ins_gbm.models.base import FittedModel


@dataclass
class EvaluationReport:
    fitted_model: FittedModel
    evaluation_data: ModelData
    train_data: ModelData
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
        return pl.DataFrame(rows)

    def plot_double_lift(self, name: str, output_path: Optional[str] = None):
        if self.comparison_predictions is None or name not in self.comparison_predictions:
            raise KeyError(
                f"No comparison prediction named {name!r}. "
                f"Available: {list(self.comparison_predictions or {})}"
            )
        gbm_preds = self.fitted_model.predict(self.evaluation_data, prediction_type="response")
        weights = (
            self.evaluation_data.exposure
            if self.evaluation_data.exposure is not None
            else self.evaluation_data.weight
        )
        return _p.plot_double_lift(
            self.evaluation_data.target,
            gbm_preds,
            self.comparison_predictions[name],
            weights=weights,
            labels=("GBM", name),
            output_path=output_path,
        )

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
        return pl.DataFrame(rows)

    def _export_comparison(self, output_dir: str) -> None:
        self._comparison_metrics().write_csv(os.path.join(output_dir, "metrics.csv"))
        model_items = list(self._comparison_models.items())
        for (name_a, (fitted_a, _, evaluation_a)), (name_b, (fitted_b, _, evaluation_b)) in combinations(model_items, 2):
            _p.plot_double_lift(
                evaluation_a.target,
                fitted_a.predict(evaluation_a, prediction_type="response"),
                fitted_b.predict(evaluation_b, prediction_type="response"),
                labels=(name_a, name_b),
                output_path=os.path.join(output_dir, f"double_lift_{name_a}_vs_{name_b}.png"),
            )
