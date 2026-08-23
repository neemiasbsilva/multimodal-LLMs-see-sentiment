"""Conversion of the fp32 pickled checkpoints into safetensors for the Hub.

Every checkpoint in checkpoints/ is a bare torch state_dict saved by
utils_dl.save_checkpoint. safetensors refuses tensors that share storage, which
BART does for its tied embeddings, so aliases are detected, recorded in
config.json and cloned apart; load_state_dict re-ties them on load.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from mllmsent.experiments.matrix import Matrix
from mllmsent.experiments.spec import ExperimentSpec
from mllmsent.labels import id2label, label2id

CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"
MANIFEST_FILENAME = "MANIFEST.json"

DTYPES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_subpath(spec: ExperimentSpec) -> str:
    return "/".join(
        (
            spec.mllm_profile.dataset_dir,
            spec.backbone,
            spec.problem,
            f"sigma{spec.sigma_token}",
            spec.fine_tuning,
        )
    )


def read_fold_scores(log_dir: Path) -> dict[str, object]:
    summary = log_dir / "test_logs.csv"
    if not summary.is_file():
        return {}
    scores, accuracies = [], []
    with open(summary, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("f1_score"):
                scores.append(float(row["f1_score"]))
            if row.get("accuracy"):
                accuracies.append(float(row["accuracy"]))
    if not scores:
        return {}
    return {
        "kfold_f1": [round(value, 6) for value in scores],
        "mean_f1": round(sum(scores) / len(scores), 6),
        "mean_accuracy": (
            round(sum(accuracies) / len(accuracies), 6) if accuracies else None
        ),
        "folds": len(scores),
    }


def build_config(
    spec: ExperimentSpec,
    source: Path,
    tied_weights: dict[str, str],
    dtype_name: str,
    source_digest: str,
    log_dir: Path,
) -> dict:
    return {
        "mllmsent_format": 1,
        "spec_id": spec.id,
        "qualified_id": spec.qualified_id,
        "track": spec.track,
        "base_model": spec.backbone_profile.model_path,
        "caption_mllm": spec.mllm,
        "text_backbone": spec.backbone,
        "problem": spec.problem,
        "sigma": spec.sigma,
        "fine_tuning": spec.fine_tuning,
        "frozen_backbone": spec.freeze_backbone,
        "num_classes": spec.num_classes,
        "id2label": {str(index): name for index, name in id2label(spec.problem).items()},
        "label2id": label2id(spec.problem),
        "max_len": spec.hyperparameters.max_len,
        "learning_rate": spec.hyperparameters.learning_rate,
        "batch_size": spec.hyperparameters.batch_size,
        "torch_dtype": DTYPES[dtype_name],
        "source_checkpoint": source.name,
        "source_sha256": source_digest,
        "tied_weights": tied_weights,
        **read_fold_scores(log_dir),
    }


def already_converted(destination: Path, source_digest: str) -> bool:
    config_path = destination / CONFIG_FILENAME
    if not (config_path.is_file() and (destination / WEIGHTS_FILENAME).is_file()):
        return False
    try:
        return json.loads(config_path.read_text()).get("source_sha256") == source_digest
    except (OSError, json.JSONDecodeError):
        return False


def convert_checkpoint(
    spec: ExperimentSpec,
    source: Path,
    destination: Path,
    dtype_name: str,
    log_dir: Path,
    force: bool = False,
) -> dict | None:
    import torch
    from safetensors.torch import save_file

    source_digest = sha256_file(source)
    if not force and already_converted(destination, source_digest):
        return None

    state = torch.load(source, map_location="cpu", weights_only=True)
    target_dtype = getattr(torch, DTYPES[dtype_name])

    storage_owner: dict[tuple[int, int], str] = {}
    tied_weights: dict[str, str] = {}
    converted: dict = {}

    for key, tensor in state.items():
        identity = (tensor.untyped_storage().data_ptr(), tensor.storage_offset())
        if identity in storage_owner:
            # Stored once and rebuilt by hub.pull.load_state_dict; keeping the
            # duplicate would add ~200 MB per BART checkpoint.
            tied_weights[key] = storage_owner[identity]
            continue
        storage_owner[identity] = key
        value = tensor.to(target_dtype) if tensor.is_floating_point() else tensor
        converted[key] = value.detach().clone().contiguous()

    destination.mkdir(parents=True, exist_ok=True)
    save_file(converted, destination / WEIGHTS_FILENAME, metadata={"format": "pt"})
    config = build_config(
        spec, source, tied_weights, dtype_name, source_digest, log_dir
    )
    (destination / CONFIG_FILENAME).write_text(json.dumps(config, indent=2) + "\n")
    return config


def convert_all(
    matrix: Matrix,
    staging: Path,
    dtype_name: str = "fp16",
    force: bool = False,
    specs=None,
    on_event=print,
) -> dict:
    checkpoint_root = matrix.paths.checkpoint_root
    results_root = matrix.paths.results_root
    manifest: dict[str, dict] = {}
    converted_count = skipped_count = missing_count = 0

    for spec in specs if specs is not None else matrix.specs:
        if spec.writes_per_fold_checkpoints or spec.backbone == "swin":
            continue
        source = spec.resolve_checkpoint(checkpoint_root)
        if not source.is_file():
            missing_count += 1
            continue

        subpath = repo_subpath(spec)
        destination = staging / subpath
        config = convert_checkpoint(
            spec,
            source,
            destination,
            dtype_name,
            results_root / spec.log_dir,
            force=force,
        )
        if config is None:
            config = json.loads((destination / CONFIG_FILENAME).read_text())
            skipped_count += 1
            on_event(f"skip    {subpath}")
        else:
            converted_count += 1
            size_mb = (destination / WEIGHTS_FILENAME).stat().st_size / 1048576
            on_event(f"convert {subpath}  ({size_mb:.0f} MB)")

        manifest[spec.id + "-" + spec.fine_tuning] = {
            "path": f"{subpath}/{WEIGHTS_FILENAME}",
            "spec_id": spec.id,
            "track": spec.track,
            "base_model": config["base_model"],
            "num_classes": config["num_classes"],
            "torch_dtype": config["torch_dtype"],
            "source_sha256": config["source_sha256"],
            "mean_f1": config.get("mean_f1"),
        }

    staging.mkdir(parents=True, exist_ok=True)
    (staging / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "converted": converted_count,
        "skipped": skipped_count,
        "missing": missing_count,
        "entries": len(manifest),
    }
