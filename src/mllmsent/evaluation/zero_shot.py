#!/usr/bin/env python3
"""Zero-shot classification k-fold evaluation using a HuggingFace NLI pipeline.

Replaces: notebooks/zero-shot-bart-large-mnli.ipynb

Iterates over every CSV in the given data directories, runs 5-fold cross-
validation with the specified zero-shot classification model, and saves per-
dataset metric CSVs.

Example
-------
python scripts/zero_shot_kfold.py \\
    --data-dirs data/gpt4-openai-classify data/deepseek \\
    --model MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \\
    --output-dir experiments-zero-shot/results \\
    --device cuda
"""
import argparse
import os
import sys
import warnings

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from transformers import pipeline
from tqdm import tqdm
from mllmsent.evaluation.metrics import compute_ci

warnings.filterwarnings("ignore")

LABELS_P5 = ["Positive", "SlightlyPositive", "Neutral", "SlightlyNegative", "Negative"]
LABELS_P3 = ["Positive", "Neutral", "Negative"]
LABELS_P2 = ["Negative", "Positive"]

SENT_MAP_P5 = {
    "Positive": 4,
    "SlightlyPositive": 3,
    "Neutral": 2,
    "SlightlyNegative": 1,
    "Negative": 0,
}
SENT_MAP_P3 = {"Negative": 1, "Neutral": 0, "Positive": 2}


def _labels_and_map(n_classes: int, filename: str) -> tuple:
    if n_classes == 5:
        return LABELS_P5, SENT_MAP_P5
    if n_classes == 3:
        return LABELS_P3, SENT_MAP_P3
    # Binary: distinguish P2+ (Neutral→Positive) from P2- (Neutral→Negative)
    if filename.endswith("p2plus.csv"):
        return LABELS_P2, {"Negative": 1, "Positive": 0}
    return LABELS_P2, {"Negative": 0, "Positive": 1}


def _source_flag(data_path: str) -> str:
    """Derive a dataset-source tag from the file path."""
    parent = os.path.basename(os.path.dirname(data_path))
    # Paths with more than two hyphenated parts in the folder name → openai
    if len(parent.split("-")) > 2:
        return "openai"
    return "opensource"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero-shot NLI k-fold evaluation on percept datasets"
    )
    p.add_argument(
        "--data-dirs",
        nargs="+",
        required=True,
        help="Directories containing percept dataset CSV files",
    )
    p.add_argument(
        "--model",
        default="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        help="HuggingFace zero-shot classification model",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory where per-dataset metric CSVs are written",
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    classifier = pipeline(
        "zero-shot-classification", model=args.model, device=args.device
    )
    kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    data_paths = []
    for d in args.data_dirs:
        data_paths.extend(
            os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".csv")
        )

    for data_path in tqdm(data_paths, desc="Datasets"):
        df = pd.read_csv(data_path)
        if len(df) < args.n_splits:
            print(f"Skipping {data_path} (fewer than {args.n_splits} samples)")
            continue

        fname = os.path.basename(data_path)
        n_classes = df["sentiment"].nunique()
        labels, smap = _labels_and_map(n_classes, fname)

        rows = []
        for _, val_idx in kfold.split(df):
            val_df = df.iloc[val_idx]
            target = val_df["sentiment"].tolist()
            pred = [smap[classifier(t, labels)["labels"][0]] for t in val_df["text"]]
            rows.append(
                {
                    "accuracy": accuracy_score(target, pred),
                    "f1_score": f1_score(target, pred, average="weighted"),
                }
            )

        df_metrics = pd.DataFrame(rows)
        flag = _source_flag(data_path)
        out_path = os.path.join(args.output_dir, f"{flag}-{fname}")
        df_metrics.to_csv(out_path, index=False)

        mean_f1, ci = compute_ci(df_metrics["f1_score"].tolist())
        print(
            f"{fname}: mean_f1={mean_f1:.4f}  "
            f"95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]  ±{abs(ci[1]-mean_f1):.4f}"
        )


if __name__ == "__main__":
    main()
