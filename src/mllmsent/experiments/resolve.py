"""Flat, diffable projection of an ExperimentSpec.

The column set mirrors tools/legacy_resolve.py so the two can be diffed line for
line; that diff is the proof that replacing path-parsing with experiments.yaml
did not move any existing result or checkpoint.
"""

from __future__ import annotations

from pathlib import Path

from mllmsent.experiments.matrix import Matrix
from mllmsent.experiments.spec import ExperimentSpec

COLUMNS = (
    "qualified_id",
    "tree",
    "dir_name",
    "log_dir",
    "checkpoint_path",
    "dataset_csv",
    "model_name",
    "model_path",
    "learning_rate",
    "batch_size",
    "epochs",
    "max_len",
    "patience",
    "freeze_backbone",
    "fine_tuning",
    "num_classes",
    "objective",
    "validation",
)


def checkpoint_path(spec: ExperimentSpec, checkpoint_root: Path) -> str:
    if spec.writes_per_fold_checkpoints:
        return f"{spec.checkpoint_dir(checkpoint_root)}/best_model_fold{{fold}}.pt"
    if spec.backbone == "swin":
        return "none"
    return f"{spec.checkpoint_dir(checkpoint_root)}/{spec.checkpoint_name}"


def project(spec: ExperimentSpec, matrix: Matrix, relative: bool = True) -> dict[str, str]:
    root = matrix.paths.root
    checkpoint_root = matrix.paths.checkpoint_root
    data_root = matrix.paths.data_root

    def shorten(value: str) -> str:
        if not relative:
            return value
        prefix = f"{root}/"
        return value[len(prefix) :] if value.startswith(prefix) else value

    hyperparameters = spec.hyperparameters
    return {
        "qualified_id": spec.qualified_id,
        "tree": spec.tree,
        "dir_name": spec.dir_name,
        "log_dir": spec.log_dir,
        "checkpoint_path": shorten(checkpoint_path(spec, checkpoint_root)),
        "dataset_csv": shorten(str(spec.dataset_csv(data_root))),
        "model_name": spec.backbone_profile.model_name,
        "model_path": spec.backbone_profile.model_path,
        "learning_rate": repr(hyperparameters.learning_rate),
        "batch_size": str(hyperparameters.batch_size),
        "epochs": str(hyperparameters.epochs),
        "max_len": str(hyperparameters.max_len),
        "patience": str(hyperparameters.patience),
        "freeze_backbone": str(spec.freeze_backbone),
        "fine_tuning": spec.fine_tuning,
        "num_classes": str(spec.num_classes),
        "objective": spec.objective,
        "validation": spec.validation,
    }
