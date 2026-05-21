#!/usr/bin/env python3
"""Build classification and regression datasets from raw captions + vote JSON.

Replaces: notebooks/preprocess_raw_data.ipynb

Example
-------
# Build all classification variants for the GPT-4 OpenAI captions
python scripts/preprocess_data.py \\
    --caption-file data/raw/image_caption_gpt4_openai.csv \\
    --caption-col text \\
    --caption-delimiter ";" \\
    --dataset-json data/raw/dataset.json \\
    --output-dir data/gpt4-openai-classify \\
    --regression-dir data/gpt4-openai-regression \\
    --alphas 3 4 5 \\
    --problems p5 p3 p2plus p2neg

# DeepSeek captions (comma-delimited, column named 'text')
python scripts/preprocess_data.py \\
    --caption-file data/raw/image_caption_deepseek.csv \\
    --dataset-json data/raw/dataset.json \\
    --output-dir data/deepseek \\
    --alphas 3 4 5 \\
    --problems p5 p3 p2plus p2neg
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import build_percept_dataset, build_regression_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess raw captions + vote data into classification/regression CSVs"
    )
    p.add_argument("--caption-file", required=True, help="Path to the caption CSV")
    p.add_argument(
        "--caption-col",
        default="text",
        help="Column name for captions (default: text; legacy files may use 'caption')",
    )
    p.add_argument(
        "--caption-delimiter", default=",", help="CSV delimiter for caption file"
    )
    p.add_argument("--dataset-json", required=True, help="Path to dataset.json vote file")
    p.add_argument(
        "--output-dir", required=True, help="Directory for classification CSVs"
    )
    p.add_argument(
        "--regression-dir",
        default=None,
        help="Directory for regression CSVs (omit to skip regression)",
    )
    p.add_argument(
        "--alphas",
        nargs="+",
        type=int,
        default=[3, 4, 5],
        help="Alpha (frequency) thresholds to generate",
    )
    p.add_argument(
        "--problems",
        nargs="+",
        default=["p5", "p3", "p2plus", "p2neg"],
        choices=["p5", "p3", "p2plus", "p2neg"],
        help="Problem variants to generate",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df_captions = pd.read_csv(args.caption_file, delimiter=args.caption_delimiter)
    # Normalise legacy 'caption' column name
    if "caption" in df_captions.columns and args.caption_col not in df_captions.columns:
        df_captions = df_captions.rename(columns={"caption": "text"})
        args.caption_col = "text"

    with open(args.dataset_json) as f:
        data_json = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    for problem in args.problems:
        # Classification datasets (one per alpha threshold)
        for alpha in args.alphas:
            df = build_percept_dataset(
                df_captions, data_json, problem, alpha, args.caption_col
            )
            out = os.path.join(
                args.output_dir, f"percept_dataset_alpha{alpha}_{problem}.csv"
            )
            df.to_csv(out, index=False)
            print(f"Saved {out}  ({len(df)} samples)")

        # Regression dataset (alpha-independent: uses mean of all votes)
        if args.regression_dir:
            os.makedirs(args.regression_dir, exist_ok=True)
            df_reg = build_regression_dataset(
                df_captions, data_json, problem, args.caption_col
            )
            out_reg = os.path.join(
                args.regression_dir, f"percept_dataset_regression_{problem}.csv"
            )
            df_reg.to_csv(out_reg, index=False)
            print(f"Saved {out_reg}  ({len(df_reg)} samples)")


if __name__ == "__main__":
    main()
