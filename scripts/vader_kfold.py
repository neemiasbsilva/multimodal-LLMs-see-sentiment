#!/usr/bin/env python3
"""VADER baseline k-fold evaluation on all percept dataset variants.

Replaces: notebooks/vader.ipynb

Iterates over every CSV in the given data directories, skips P5 datasets
(VADER cannot distinguish 5 classes), runs 5-fold cross-validation with
NLTK's SentimentIntensityAnalyzer, and saves per-dataset metric CSVs.

Example
-------
python scripts/vader_kfold.py \\
    --data-dirs data/gpt4-openai-classify data/deepseek \\
    --output-dir experiments-vader/results
"""
import argparse
import os
import sys

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.evaluation import compute_ci


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VADER baseline k-fold evaluation on percept datasets"
    )
    p.add_argument(
        "--data-dirs",
        nargs="+",
        required=True,
        help="Directories containing percept dataset CSV files",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory where per-dataset metric CSVs are written",
    )
    p.add_argument("--n-splits", type=int, default=5)
    return p.parse_args()


def _sent_map(n_classes: int, filename: str) -> dict | None:
    """Return VADER polarity-key → integer-label mapping, or None if unsupported."""
    if n_classes == 3:
        return {"neg": 1, "neu": 0, "pos": 2}
    if n_classes == 2:
        if filename.endswith("p2plus.csv"):
            return {"neg": 1, "neu": 0, "pos": 0}
        return {"neg": 0, "neu": 0, "pos": 1}
    return None  # P5 not supported


def _source_flag(data_path: str) -> str:
    if "openai" in data_path:
        return "openai"
    if "deepseek" in data_path:
        return "deepseek"
    return "percept"


def main() -> None:
    args = parse_args()
    nltk.download("vader_lexicon", quiet=True)
    os.makedirs(args.output_dir, exist_ok=True)

    data_paths = []
    for d in args.data_dirs:
        data_paths.extend(
            os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".csv")
        )

    analyzer = SentimentIntensityAnalyzer()
    kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    for data_path in tqdm(data_paths, desc="Datasets"):
        df = pd.read_csv(data_path)
        if len(df) < args.n_splits:
            print(f"Skipping {data_path} (fewer than {args.n_splits} samples)")
            continue

        n_classes = df["sentiment"].nunique()
        fname = os.path.basename(data_path)
        smap = _sent_map(n_classes, fname)
        if smap is None:
            print(f"Skipping {data_path} (P5 not supported by VADER)")
            continue

        rows = []
        for _, val_idx in kfold.split(df):
            val_df = df.iloc[val_idx]
            target = val_df["sentiment"].tolist()
            pred = []
            for text in val_df["text"]:
                scores = analyzer.polarity_scores(text)
                del scores["compound"]
                max_key = max(scores, key=scores.get)
                pred.append(smap[max_key])
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
