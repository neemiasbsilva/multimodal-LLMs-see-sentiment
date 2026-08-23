"""Downloading the published inputs and results back into the working tree.

hub/datasets.py maps data/ and output/ onto the dataset repository for upload;
this module walks the same map backwards. A fresh clone carries neither folder
— both are gitignored — so this is how an experiment's dataset CSV and every
published result reach the paths experiments.yaml resolves them to.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from mllmsent.experiments.matrix import Matrix

DATA_GROUPS = ("inputs", "captions", "splits")
RESULT_GROUP = "results"
GROUPS = (*DATA_GROUPS, RESULT_GROUP)
REPORTS_TREE = "reports"


def group_root(matrix: Matrix, group: str) -> Path:
    if group in DATA_GROUPS:
        return matrix.paths.data_root
    return matrix.paths.output_root


def local_path(matrix: Matrix, repo_file: str) -> Path | None:
    group, _, remainder = repo_file.partition("/")
    if not remainder:
        return None
    if group in DATA_GROUPS:
        return matrix.paths.data_root / remainder
    if group != RESULT_GROUP:
        return None
    tree, _, tail = remainder.partition("/")
    if tree == REPORTS_TREE:
        return matrix.paths.output_root / REPORTS_TREE / tail
    return matrix.paths.results_root / remainder


def download_snapshot(repo_id: str, groups, revision: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"{group}/**" for group in groups],
        )
    )


def download_dataset(
    repo_id: str,
    matrix: Matrix,
    groups=GROUPS,
    revision: str | None = None,
    force: bool = False,
    on_event=print,
) -> dict:
    snapshot = download_snapshot(repo_id, groups, revision)
    written = {group: 0 for group in groups}
    kept = 0

    for group in groups:
        for source in sorted((snapshot / group).rglob("*")):
            if not source.is_file():
                continue
            target = local_path(matrix, source.relative_to(snapshot).as_posix())
            if target is None:
                continue
            if target.is_file() and not force:
                kept += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            written[group] += 1
        on_event(f"{group:>10}: {written[group]} files -> {group_root(matrix, group)}")

    return {"written": sum(written.values()), "kept": kept, "groups": written}
