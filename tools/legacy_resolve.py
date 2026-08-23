"""Reference implementation of the pre-refactor experiment derivation.

Walks every config.yaml on disk and reproduces, verbatim, the values that
scripts/train_gpu0.py and scripts/train_gpu1.py compute by string-splitting the
config file path. Its TSV output is the baseline the new experiments.yaml matrix
must reproduce exactly.

Throwaway: delete once `mllmsent matrix resolve` is trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_TREES = (
    "experiments-finetuning",
    "experiments-not-finetuning",
    "experiments-twitter",
    "experiments-swin",
    "exps-tuning-handle-subjectivity",
    "exps-tuning-handle-subjectivity-no-sigma",
)

TRACK_BY_TREE = {
    "experiments-finetuning": "finetuning",
    "experiments-not-finetuning": "not-finetuning",
    "experiments-twitter": "twitter",
    "experiments-swin": "swin",
    "exps-tuning-handle-subjectivity": "subjectivity",
    "exps-tuning-handle-subjectivity-no-sigma": "subjectivity-no-sigma",
}

DATASET_TYPE_BY_PREFIX = {
    "openai": "gpt4-openai-classify",
    "deepseek": "deepseek",
    "gemini": "gemini",
    "phi4": "phi4-classify",
    "gemma4": "gemma4-classify",
}
DEFAULT_DATASET_TYPE = "minigpt4-classify"

MLLM_BY_PREFIX = {
    "openai": "openai",
    "deepseek": "deepseek",
    "gemini": "gemini",
    "phi4": "phi4",
    "gemma4": "gemma4",
}
DEFAULT_MLLM = "minigpt4"

BACKBONE_BY_MODEL_NAME = {
    "distil-bert": "distilbert",
    "distil-bert-pooling-self-attention": "distilbert",
    "modern-bert": "modernbert",
    "llama-qlora": "llama3-qlora",
    "bart": "bart",
    "swin": "swin",
}

ARCH_TOKEN_BY_BACKBONE = {
    "distilbert": "distilbert",
    "modernbert": "modernbert",
    "llama3-qlora": "llamaqlora",
    "bart": "bart",
    "swin": "swin",
}

NUM_CLASSES_BY_PROBLEM = {"p5": 5, "p3": 3, "p2plus": 2, "p2neg": 2}

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


def trainer_lineage(backbone: str, mllm: str) -> str:
    """Which of train_gpu0.py / train_gpu1.py actually produced these results.

    Derived from the run-*.sh call graph: bart and llama3-qlora were always
    launched through train_gpu0; modernbert through train_gpu1 except for the
    phi4 and gemma4 families, which run-phi4.sh and run-gemma4.sh send through
    train_gpu0. distilbert predates the split (run-not-finetune-distil-bert.sh
    calls the long-deleted scripts/train.py) and is treated as gpu1 lineage.
    """
    if backbone in ("bart", "llama3-qlora"):
        return "gpu0"
    if backbone == "modernbert" and mllm in ("phi4", "gemma4"):
        return "gpu0"
    return "gpu1"


def resolve_fine_tuning(tree: str, backbone: str, lineage: str) -> str:
    if backbone == "llama3-qlora":
        return "finetuned" if tree == "experiments-finetuning" else "not_finetuned"
    if lineage == "gpu0" and backbone in ("distilbert", "modernbert"):
        return "finetuned" if tree == "experiments-finetuning" else "not_finetuned"
    return "not_finetuned" if tree == "experiments-not-finetuning" else "finetuned"


def resolve_patience(tree: str, lineage: str) -> int:
    if tree != "experiments-not-finetuning":
        return 10
    return 50 if lineage == "gpu0" else 25


def iter_config_paths():
    for tree in EXPERIMENT_TREES:
        root = REPO_ROOT / tree
        if not root.is_dir():
            continue
        for config_path in sorted(root.glob("*/config.yaml")):
            yield config_path


def problem_from_substring(text: str) -> str:
    for problem in ("p2plus", "p2neg", "p5", "p3"):
        if problem in text:
            return problem
    return "p5"


def resolve(config_path: Path) -> dict[str, str]:
    config = yaml.safe_load(config_path.read_text())

    parts = config_path.parts
    dir_name = parts[-2]
    tree = parts[-3]
    track = TRACK_BY_TREE[tree]
    objective = "regression" if tree.startswith("exps-tuning") else "classification"

    tokens = dir_name.split("-")
    prefix = tokens[0]

    # train_handle_subjectivity_non_sigma.py does not path-parse: it detects the
    # problem by substring and drops sigma entirely.
    if track == "subjectivity-no-sigma":
        alpha_version = None
        experiment_group = problem_from_substring(str(config_path))
        prefix = "openai"
    else:
        alpha_version = int(tokens[-1][-1])
        experiment_group = tokens[-2]

    dataset_type = DATASET_TYPE_BY_PREFIX.get(prefix, DEFAULT_DATASET_TYPE)
    mllm = MLLM_BY_PREFIX.get(prefix, DEFAULT_MLLM)

    model_name = config["model_name"]
    backbone = BACKBONE_BY_MODEL_NAME[model_name]
    arch_token = ARCH_TOKEN_BY_BACKBONE[backbone]

    lineage = trainer_lineage(backbone, mllm)
    fine_tuning = resolve_fine_tuning(tree, backbone, lineage)
    patience = 10 if objective == "regression" else resolve_patience(tree, lineage)
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")

    if objective == "regression":
        # train_handle_subjectivity*.py write one file per fold, ignoring the
        # best_checkpoint_* scheme entirely.
        checkpoint_path = f"{checkpoint_dir}/best_model_fold{{fold}}.pt"
    elif backbone == "swin":
        # swin_train.py:124 calls save_checkpoint with 5 of 8 required args, so
        # no checkpoint is ever written.
        checkpoint_path = "none"
    else:
        checkpoint_path = (
            f"{checkpoint_dir}/best_checkpoint_{dataset_type}_{arch_token}"
            f"_{experiment_group}_sigma{alpha_version}_{fine_tuning}.pt"
        )

    if track == "subjectivity-no-sigma":
        dataset_csv = f"data/gpt4-openai-regression/percept_dataset_regression_{experiment_group}.csv"
        sigma_token = "none"
    else:
        dataset_csv = (
            f"data/{dataset_type}/percept_dataset_alpha{alpha_version}_{experiment_group}.csv"
        )
        sigma_token = str(alpha_version)

    return {
        "qualified_id": f"{track}/{mllm}-{backbone}-{experiment_group}-sigma{sigma_token}",
        "tree": tree,
        "dir_name": dir_name,
        "log_dir": config["log_dir"],
        "checkpoint_path": checkpoint_path,
        "dataset_csv": dataset_csv,
        "model_name": model_name,
        "model_path": config["model_path"],
        "learning_rate": repr(float(config["learning_rate"])),
        "batch_size": str(config["batch_size"]),
        "epochs": str(config["epochs"]),
        "max_len": str(config.get("max_len")),
        "patience": str(patience),
        "freeze_backbone": str(tree == "experiments-not-finetuning"),
        "fine_tuning": fine_tuning,
        "num_classes": str(NUM_CLASSES_BY_PROBLEM[experiment_group]),
        "objective": objective,
        "validation": "twitter_captions" if tree == "experiments-twitter" else "kfold",
    }


def main() -> int:
    rows = [resolve(path) for path in iter_config_paths()]

    seen: dict[str, str] = {}
    for row in rows:
        key = row["qualified_id"]
        if key in seen:
            print(f"duplicate qualified_id: {key}", file=sys.stderr)
        seen[key] = row["dir_name"]

    print("\t".join(COLUMNS))
    for row in rows:
        print("\t".join(row[column] for column in COLUMNS))

    print(f"resolved {len(rows)} experiments", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
