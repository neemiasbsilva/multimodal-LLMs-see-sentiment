"""Regression track for the subjectivity experiments.

Predicts the mean annotator score on a continuous scale instead of a class, so
it swaps cross-entropy for MSE, reports MSE/MAE/Pearson/R2 instead of
accuracy/F1, and keeps one checkpoint per fold rather than one best overall.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm

from mllmsent.data.experiment_data import fold_frame, load_experiment_frame
from mllmsent.data.loader import data_loader
from mllmsent.experiments.spec import ExperimentSpec

KFOLD_SEED = 42


def compute_regression_metrics(preds, targets) -> tuple[float, float, float, float]:
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    try:
        pearson, _ = stats.pearsonr(preds, targets)
        if np.isnan(pearson):
            pearson = 0.0
    except (ValueError, FloatingPointError):
        pearson = 0.0
    return float(mse), float(mae), float(pearson), float(r2)


def _forward(model, batch, device):
    ids = batch["ids"].to(device)
    mask = batch["mask"].to(device)
    targets = batch["targets"].to(device).float()
    outputs = model(ids, mask, batch["token_type_id"].to(device))
    if hasattr(outputs, "logits"):
        outputs = outputs.logits
    return outputs.squeeze(-1), targets


def run_epoch(model, dataloader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    preds, targets = [], []
    with torch.set_grad_enabled(training):
        for batch in dataloader:
            outputs, batch_targets = _forward(model, batch, device)
            loss = loss_fn(outputs, batch_targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            preds.extend(outputs.detach().cpu().tolist())
            targets.extend(batch_targets.detach().cpu().tolist())

    return total_loss / max(len(dataloader), 1), preds, targets


def fit_fold(model, spec, optimizer, train_dl, val_dl, log_dir, checkpoint_dir, fold, device, max_epochs):
    loss_fn = torch.nn.MSELoss()
    epochs = max_epochs or spec.hyperparameters.epochs
    patience = spec.hyperparameters.patience

    torch.manual_seed(KFOLD_SEED)
    np.random.seed(KFOLD_SEED)

    log_file = log_dir / f"training_logs_{fold + 1:02d}.txt"
    log_file.write_text("")

    curves = pd.DataFrame([])
    best_pearson = -1.0
    stalled = 0

    for epoch in tqdm(range(epochs)):
        train_loss, _, _ = run_epoch(model, train_dl, loss_fn, device, optimizer)
        val_loss, preds, targets = run_epoch(model, val_dl, loss_fn, device)
        mse, mae, pearson, r2 = compute_regression_metrics(preds, targets)

        with open(log_file, "a") as handle:
            handle.write(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | MAE: {mae:.4f} | "
                f"Pearson: {pearson:.4f} | R2: {r2:.4f}\n"
            )

        if pearson > best_pearson:
            best_pearson = pearson
            stalled = 0
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / f"best_model_fold{fold}.pt")
        else:
            stalled += 1

        curves = pd.concat(
            [
                curves,
                pd.DataFrame(
                    {
                        "epoch": [epoch + 1],
                        "train_loss": [train_loss],
                        "val_loss": [val_loss],
                        "val_mae": [mae],
                        "val_pearson": [pearson],
                        "val_r2": [r2],
                    }
                ),
            ],
            axis=0,
        )
        curves.to_csv(log_dir / f"training_logs_{fold + 1:02d}.csv", index=False)

        if stalled >= patience:
            print(f"validation Pearson flat for {patience} epochs; stopping")
            break

    return model, loss_fn


def train_regression(spec: ExperimentSpec, matrix, folds: int = 5, max_epochs=None) -> dict:
    from transformers import AutoTokenizer

    from mllmsent.models.regression import ModernBERTModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{spec.qualified_id} (regression) on {device}")

    log_dir = matrix.paths.results_root / spec.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = spec.checkpoint_dir(matrix.paths.checkpoint_root)

    frame = load_experiment_frame(spec, matrix.paths.data_root)
    if "sentiment_score" in frame.columns:
        frame = frame.rename(columns={"sentiment_score": "sentiment"})
    frame["sentiment"] = frame["sentiment"].astype(float)

    hyperparameters = spec.hyperparameters
    train_params = {"batch_size": hyperparameters.batch_size, "shuffle": True}
    val_params = {"batch_size": hyperparameters.batch_size, "shuffle": False}

    kfold = KFold(n_splits=folds, shuffle=True, random_state=KFOLD_SEED)
    metrics = pd.DataFrame([])

    for fold, (train_idx, val_idx) in enumerate(kfold.split(frame)):
        print(f"fold {fold + 1}/{folds}")
        started = time.time()

        model = ModernBERTModel(spec.backbone_profile.model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(spec.backbone_profile.model_path)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=hyperparameters.learning_rate, weight_decay=1e-6
        )

        train_df = fold_frame(frame, train_idx)
        val_df = fold_frame(frame, val_idx)
        train_dl = data_loader(train_df, tokenizer, hyperparameters.max_len, train_params)
        val_dl = data_loader(val_df, tokenizer, hyperparameters.max_len, val_params)

        model, loss_fn = fit_fold(
            model, spec, optimizer, train_dl, val_dl,
            log_dir, checkpoint_dir, fold, device, max_epochs,
        )

        _, preds, targets = run_epoch(model, val_dl, loss_fn, device)
        mse, mae, pearson, r2 = compute_regression_metrics(preds, targets)
        metrics = pd.concat(
            [
                metrics,
                pd.DataFrame(
                    {
                        "kfold": [fold + 1],
                        "val_mse": [mse],
                        "val_mae": [mae],
                        "val_pearson": [pearson],
                        "val_r2": [r2],
                        "time_sec": [int(time.time() - started)],
                    }
                ),
            ],
            axis=0,
        )
        metrics.to_csv(log_dir / "test_logs.csv", index=False)

        pd.DataFrame(
            {
                "id": frame["id"].iloc[val_idx].to_list(),
                "text": frame["text"].iloc[val_idx].to_list(),
                "target_score": targets,
                "predicted_score": preds,
            }
        ).to_csv(log_dir / f"test_logs_{fold + 1:02d}.csv", index=False)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"mean Pearson {metrics['val_pearson'].mean():.4f}")
    return {
        "mean_pearson": float(metrics["val_pearson"].mean()),
        "mean_r2": float(metrics["val_r2"].mean()),
        "folds": len(metrics),
    }
