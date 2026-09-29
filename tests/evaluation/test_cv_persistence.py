from dataclasses import replace
import json

import numpy as np
import polars as pl
import pytest

from ins_gbm import (
    CVConfig, LightGBMModel, ModelRecipe, compare_cv_double_lift,
    compare_reports, load_cv_result, load_model_data,
)
from ins_gbm.evaluation.metrics import (
    _double_lift_metric_inputs, double_lift_score, double_lift_table,
)


def _data(poisson_parquet):
    return load_model_data(
        str(poisson_parquet), target="claim_count", exposure="exposure",
        feature_cols=["x1", "x3"], objective="poisson",
    )


def _cv(data, estimators):
    return ModelRecipe(
        model=LightGBMModel(objective="poisson"),
        params={"n_estimators": estimators, "verbose": -1},
    ).cross_validate(data, cv=CVConfig(n_splits=3, seed=13, folds="random"))


def test_saved_cv_metrics_and_double_lift_round_trip(poisson_parquet, tmp_path):
    base_data = _data(poisson_parquet)
    data = replace(
        base_data,
        weight=pl.Series("weight", np.linspace(0.5, 2.0, base_data.n_rows)),
    )
    reference = _cv(data, 6)
    candidate = _cv(data, 12)
    reference.save(str(tmp_path / "cv"))
    restored = load_cv_result(str(tmp_path / "cv"))

    assert restored.fold_metrics.equals(reference.fold_metrics)
    assert restored.summary.equals(reference.summary)
    assert restored.predictions.equals(reference.predictions)
    assert restored.row_folds.equals(reference.row_folds)
    assert restored.fold_params == reference.fold_params
    assert restored.actual is None
    assert restored.exposure is None
    assert restored.weight is None
    metric_comparison = compare_reports({"saved": restored, "candidate": candidate})
    assert {"metric", "saved", "candidate", "preferred"}.issubset(
        metric_comparison.columns
    )
    assert "double_lift_score" not in metric_comparison["metric"].to_list()

    expected = compare_cv_double_lift(reference, candidate)
    actual = compare_cv_double_lift(restored, candidate)
    assert actual["score"].to_list() == pytest.approx(expected["score"].to_list())
    assert actual["scope"].to_list() == ["overall", "fold", "fold", "fold"]
    a, p_a, p_b, w = _double_lift_metric_inputs(
        data.objective, data.target, reference.predictions["gbm"],
        candidate.predictions["gbm"], data.exposure, data.weight,
    )
    direct_score = double_lift_score(
        double_lift_table(a, p_a, p_b, weights=w, n_bins=10)
    )
    assert actual["score"][0] == pytest.approx(direct_score)
    for row in actual.filter(pl.col("scope") == "fold").iter_rows(named=True):
        indices = np.flatnonzero(reference.row_folds.to_numpy() == int(row["fold"]))
        taken = indices.tolist()
        fold_score = double_lift_score(double_lift_table(
            a.gather(taken), p_a.gather(taken), p_b.gather(taken),
            weights=w.gather(taken) if w is not None else None, n_bins=10,
        ))
        assert row["score"] == pytest.approx(fold_score)

    candidate.save(str(tmp_path / "candidate_cv"))
    loaded_candidate = load_cv_result(str(tmp_path / "candidate_cv"))
    assert compare_cv_double_lift(restored, loaded_candidate, data=data)[
        "score"
    ].to_list() == pytest.approx(expected["score"].to_list())
    with pytest.raises(ValueError, match="Evaluation data is needed"):
        compare_cv_double_lift(restored, loaded_candidate)

    saved_columns = pl.read_parquet(tmp_path / "cv" / "predictions.parquet").columns
    assert saved_columns == ["gbm", "_cv_fold"]
    metadata = json.loads((tmp_path / "cv" / "metadata.json").read_text())
    assert "target" not in metadata
    assert "exposure" not in metadata
    assert "weight" not in metadata
    assert metadata["row_count"] == data.n_rows


def test_cv_double_lift_rejects_misalignment(poisson_parquet, tmp_path):
    data = _data(poisson_parquet)
    reference = _cv(data, 6)
    candidate = _cv(data, 8)
    reference.save(str(tmp_path / "cv"))
    saved = load_cv_result(str(tmp_path / "cv"))

    reordered_data = replace(data, target=data.target.reverse())
    with pytest.raises(ValueError, match="does not match"):
        compare_cv_double_lift(saved, candidate, data=reordered_data)
    changed_weights = replace(data, weight=pl.Series([2.0] * data.n_rows))
    with pytest.raises(ValueError, match="does not match"):
        compare_cv_double_lift(saved, candidate, data=changed_weights)
    with pytest.raises(ValueError, match="different row or fold alignment"):
        compare_cv_double_lift(
            saved, replace(candidate, row_folds=candidate.row_folds.reverse())
        )
    with pytest.raises(ValueError, match="does not match"):
        compare_cv_double_lift(saved, replace(candidate, objective="gamma"))
