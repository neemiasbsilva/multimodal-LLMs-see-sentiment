"""The single training entry point, driven by an ExperimentSpec.

Replaces scripts/train_gpu0.py, scripts/train_gpu1.py and "scripts/train copy.py",
which were ~90% identical forks. The three values that genuinely differed
between the forks — the fine_tuning rule, the early-stopping patience and the
device pin — are now spec fields rather than literals, so one code path
reproduces what both forks produced.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import KFold
from tqdm import tqdm

from mllmsent.data.experiment_data import (
    fold_frame,
    load_experiment_frame,
    load_twitter_validation,
)
from mllmsent.data.loader import data_loader
from mllmsent.experiments.spec import ExperimentSpec
from mllmsent.models.registry import get_builder
from mllmsent.training.loops import (
    compute_metrics,
    compute_val_loss_and_preds,
    compute_val_metrics,
    log_metrics,
    save_metrics_to_csv,
    train_one_epoch,
    update_metrics_df,
    validate_one_epoch,
)

KFOLD_SPLITS = 5
KFOLD_SEED = 42
HEAD_NAMES = ("classifier", "score", "lm_head", "classification_head")


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def freeze_backbone(model) -> None:
    for name, module in model.named_children():
        if name in HEAD_NAMES:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = False


def class_weights_for(frame: pd.DataFrame, device: str) -> torch.Tensor:
    counts = frame.sentiment.value_counts().sort_index().to_list()
    return torch.Tensor([1 / count for count in counts]).type(torch.float).to(device)


def save_checkpoint(model, spec: ExperimentSpec, checkpoint_root: Path) -> Path:
    directory = spec.checkpoint_dir(checkpoint_root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / spec.checkpoint_name
    torch.save(model.state_dict(), path)
    return path


def fit(
    model,
    spec: ExperimentSpec,
    class_weights,
    optimizer,
    train_dl,
    val_dl,
    log_dir: Path,
    fold: int,
    device: str,
    max_epochs: int | None = None,
):
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    model_name = spec.backbone_profile.model_name

    if spec.freeze_backbone:
        print("freezing backbone")
        freeze_backbone(model)

    torch.manual_seed(KFOLD_SEED)
    np.random.seed(KFOLD_SEED)

    log_file = log_dir / f"training_logs_{fold + 1:02d}.txt"
    log_file.write_text("")

    epochs = max_epochs or spec.hyperparameters.epochs
    patience = spec.hyperparameters.patience
    metrics = pd.DataFrame([])
    best_f1 = 0.0
    stalled = 0

    for epoch in tqdm(range(epochs)):
        train_loss, train_preds, train_targets = train_one_epoch(
            model, train_dl, optimizer, loss_fn, device, model_name
        )
        train_accuracy, train_f1 = compute_metrics(train_preds, train_targets)
        train_loss /= len(train_dl)

        val_loss, val_preds, val_targets = validate_one_epoch(
            model, val_dl, loss_fn, device, model_name
        )
        val_accuracy, val_f1 = compute_metrics(val_preds, val_targets)
        val_loss /= len(val_dl)

        log_metrics(
            epoch, epochs, train_loss, train_accuracy, train_f1,
            val_loss, val_accuracy, val_f1, str(log_file),
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            stalled = 0
        else:
            stalled += 1

        if stalled >= patience:
            print(f"validation F1 flat for {patience} epochs; stopping")
            break

        metrics = pd.concat(
            [
                metrics,
                pd.DataFrame(
                    {
                        "epoch": [epoch + 1],
                        "train_accuracy": [train_accuracy],
                        "train_f1_score": [train_f1],
                        "val_accuracy": [val_accuracy],
                        "val_f1_score": [val_f1],
                    }
                ),
            ],
            axis=0,
        )
        metrics.to_csv(log_dir / f"training_logs_{fold + 1:02d}.csv", index=False)

    return model, loss_fn


def evaluate_fold(spec, model, val_dl, loss_fn, fold, metrics, device, started):
    model.eval()
    total_loss, preds, targets = compute_val_loss_and_preds(
        model, val_dl, loss_fn, device, spec.backbone_profile.model_name
    )
    accuracy, f1, _ = compute_val_metrics(preds, targets, total_loss, val_dl)
    metrics = update_metrics_df(metrics, fold, accuracy, f1, started)
    return metrics, preds, targets, f1


def train(
    spec: ExperimentSpec,
    matrix,
    folds: int = KFOLD_SPLITS,
    max_epochs: int | None = None,
    resume: bool = False,
) -> dict:
    if spec.objective == "regression":
        from mllmsent.training.regression import train_regression

        return train_regression(spec, matrix, folds=folds, max_epochs=max_epochs)

    if spec.backbone == "llama3-qlora":
        from mllmsent.training.qlora import train_qlora

        return train_qlora(spec, matrix, folds=folds, max_epochs=max_epochs)

    device = resolve_device()
    print(f"{spec.qualified_id} on {device}")

    log_dir = matrix.paths.results_root / spec.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    frame = load_experiment_frame(spec, matrix.paths.data_root)
    builder = get_builder(spec.backbone)
    hyperparameters = spec.hyperparameters
    train_params = {"batch_size": hyperparameters.batch_size, "shuffle": True}
    val_params = {"batch_size": hyperparameters.batch_size, "shuffle": False}

    twitter_val = (
        load_twitter_validation(spec, matrix.paths.twitter_captions_root)
        if spec.uses_twitter_validation
        else None
    )
    modality = spec.backbone_profile.modality
    image_dir = str(matrix.paths.image_root) if modality == "image" else None

    kfold = KFold(n_splits=folds, shuffle=True, random_state=KFOLD_SEED)
    metrics = pd.DataFrame([])
    best_f1 = 0.0

    for fold, (train_idx, val_idx) in enumerate(kfold.split(frame)):
        print(f"fold {fold + 1}/{folds}")
        started = time.time()

        train_df = fold_frame(frame, train_idx, modality)
        val_df = (
            twitter_val
            if twitter_val is not None
            else fold_frame(frame, val_idx, modality)
        )

        model = builder.build_model(spec.backbone_profile.model_path, spec.num_classes)
        tokenizer = builder.build_tokenizer(spec.backbone_profile.model_path)
        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=hyperparameters.learning_rate, weight_decay=1e-6
        )
        train_dl = data_loader(
            train_df, tokenizer, hyperparameters.max_len, train_params, image_dir
        )
        val_dl = data_loader(
            val_df, tokenizer, hyperparameters.max_len, val_params, image_dir
        )

        model, loss_fn = fit(
            model, spec, class_weights_for(train_df, device), optimizer,
            train_dl, val_dl, log_dir, fold, device, max_epochs=max_epochs,
        )

        metrics, preds, targets, fold_f1 = evaluate_fold(
            spec, model, val_dl, loss_fn, fold, metrics, device, started
        )
        save_metrics_to_csv(metrics, str(log_dir))

        if fold_f1 > best_f1:
            best_f1 = fold_f1
            saved = save_checkpoint(model, spec, matrix.paths.checkpoint_root)
            print(f"  saved {saved.name} (F1 {fold_f1:.4f})")

        if twitter_val is None:
            columns = {
                "id": frame["id"].iloc[val_idx].to_list(),
                "target": targets,
                "prediction": preds,
            }
            if modality != "image":
                columns["text"] = frame["text"].iloc[val_idx].to_list()
                columns = {
                    "id": columns["id"],
                    "text": columns["text"],
                    "target": targets,
                    "prediction": preds,
                }
            predictions = pd.DataFrame(columns)
        else:
            predictions = pd.DataFrame(
                {
                    "text": val_df["text"].to_list(),
                    "target": val_df["sentiment"].to_list(),
                    "prediction": preds,
                }
            )
        predictions.to_csv(log_dir / f"test_logs_{fold + 1:02d}.csv", index=False)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    scores = metrics["f1_score"].to_numpy()
    mean_f1 = float(np.mean(scores))
    interval = (
        stats.t.interval(0.95, len(scores) - 1, loc=mean_f1, scale=stats.sem(scores))
        if len(scores) > 1
        else (mean_f1, mean_f1)
    )
    print(f"mean F1 {mean_f1 * 100:.2f}%  95% CI {interval}")
    return {"mean_f1": mean_f1, "confidence_interval": interval, "folds": len(scores)}
