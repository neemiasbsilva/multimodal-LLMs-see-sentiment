"""Staging layout for the dataset repository.

Nothing is copied: the staging tree is symlinks into the existing folders, so
building it is instant and costs no disk. upload_folder follows them.
"""

from __future__ import annotations

from pathlib import Path

from mllmsent.experiments.matrix import Matrix

INPUT_DIRS = (
    "percept_dataset",
    "gpt4-openai-classify",
    "gemini-classify",
    "deepseek",
    "phi4-classify",
    "gemma4-classify",
    "minigpt4-classify",
    "gpt4-openai-regression",
    "raw",
)

CAPTION_DIRS = (
    "gpt4-openai-only",
    "gemini-only",
    "deepseek-only",
    "phi4-only",
    "gemma4-only",
)

SPLIT_DIRS = ("train", "validation", "test")

RESULT_TREES = (
    "experiments-finetuning",
    "experiments-not-finetuning",
    "experiments-twitter",
    "experiments-swin",
    "experiments-vader",
    "exps-tuning-handle-subjectivity",
    "exps-tuning-handle-subjectivity-no-sigma",
)

RESULT_SUFFIXES = (".csv", ".txt", ".yaml", ".json")

RESULT_EXCLUDED_PARTS = ("checkpoint-", "trained_model", "runs")


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source.resolve())


def _link_tree(source: Path, target: Path) -> int:
    count = 0
    for path in sorted(source.rglob("*")):
        if path.is_file():
            _link(path, target / path.relative_to(source))
            count += 1
    return count


def is_publishable_result(path: Path) -> bool:
    if path.suffix not in RESULT_SUFFIXES:
        return False
    return not any(
        part.startswith(RESULT_EXCLUDED_PARTS[0]) or part in RESULT_EXCLUDED_PARTS[1:]
        for part in path.parts
    )


def build_staging(matrix: Matrix, staging: Path, on_event=print) -> dict:
    data_root = matrix.paths.data_root
    results_root = matrix.paths.results_root
    output_root = matrix.paths.output_root
    counts: dict[str, int] = {}

    for group, names in (("inputs", INPUT_DIRS), ("captions", CAPTION_DIRS)):
        total = 0
        for name in names:
            source = data_root / name
            if source.is_dir():
                total += _link_tree(source, staging / group / name)
        counts[group] = total

    splits = 0
    for name in SPLIT_DIRS:
        source = data_root / name
        if source.is_dir():
            splits += _link_tree(source, staging / "splits" / name)
    counts["splits"] = splits

    results = 0
    for tree in RESULT_TREES:
        source = results_root / tree
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if path.is_file() and is_publishable_result(path):
                _link(path, staging / "results" / tree / path.relative_to(source))
                results += 1
    counts["results"] = results

    reports = output_root / "reports"
    report_count = 0
    if reports.is_dir():
        for path in sorted(reports.rglob("*")):
            if path.is_file() and path.suffix in (".csv", ".json", ".tex", ".png"):
                if "attention_flow" in path.parts:
                    continue
                _link(path, staging / "results" / "reports" / path.relative_to(reports))
                report_count += 1
    counts["reports"] = report_count

    for group, count in counts.items():
        on_event(f"{group:>10}: {count} files")
    return counts


def dataset_chunks(staging: Path) -> list[Path]:
    return [path for path in sorted(staging.iterdir()) if path.is_dir()]
