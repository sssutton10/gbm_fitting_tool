"""Importable subprocess entry point for JournalStorage-backed tuning."""

from __future__ import annotations

import sys

import cloudpickle
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from ins_gbm.tuning.tuner import _evaluate_trial


def main() -> int:
    payload_path, journal_path, study_name, n_trials, seed = sys.argv[1:]
    with open(payload_path, "rb") as payload_file:
        config = cloudpickle.load(payload_file)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = JournalStorage(JournalFileBackend(file_path=journal_path))
    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=int(seed)),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: _evaluate_trial(trial, config),
        n_trials=int(n_trials),
        n_jobs=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
