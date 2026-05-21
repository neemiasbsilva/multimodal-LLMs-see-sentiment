#!/usr/bin/env python3
"""K-fold evaluation for MLLM sentiment classification (Tasks 1, 2a, 2b).

Replaces: notebooks/phi4-sentiment-kfold.ipynb
          notebooks/gemma4-sentiment-kfold.ipynb
          notebooks/deepseek-sentiment-kfold.ipynb
          notebooks/openai-sentiment-kfold.ipynb

Task 1  – Direct MLLM image classification; reads responses from data/{model}-only/.
Task 2a – ModernBERT frozen backbone; reads experiment logs from experiments-not-finetuning/.
Task 2b – ModernBERT fine-tuned; reads experiment logs from experiments-finetuning/.

Response file naming convention (data/{model}-only/):
    alpha3p3.csv, alpha3p5.csv, alpha5p3.csv, alpha5p5.csv

Experiment log path convention:
    experiments-{not-}finetuning/{model}-modernbert-experiment-{prob}-{alpha}/logs/test_logs.csv

Example
-------
# Phi-4: all three tasks
python scripts/kfold_eval.py --model phi4 --tasks 1 2a 2b

# Gemma-4: Task 1 only
python scripts/kfold_eval.py --model gemma4 --tasks 1

# DeepSeek: Task 1, save CSVs
python scripts/kfold_eval.py --model deepseek --tasks 1 --output-dir reports/kfold

# OpenAI: response column is named 'response' instead of 'sentiment'
python scripts/kfold_eval.py --model openai --tasks 1 --response-col response
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.evaluation import (
    compute_ci,
    kfold_log_eval,
    kfold_task1_eval,
    print_task_summary,
    results_table,
)

# ── Model configuration ────────────────────────────────────────────────────────

# Subdirectory inside data/ that holds the model's raw response CSVs
MODEL_DATA_DIRS: dict = {
    "phi4": "phi4-only",
    "gemma4": "gemma4-only",
    "deepseek": "deepseek-only",
    "openai": "gpt4-openai-only",
}

# Prefix used in experiment directory names
MODEL_EXP_PREFIX: dict = {
    "phi4": "phi4-modernbert-experiment",
    "gemma4": "gemma4-modernbert-experiment",
    "deepseek": "deepseek-modernbert-experiment",
    "openai": "openai-modernbert-experiment",
}

GT_DIR_NAME = "gpt4-openai-classify"

# (problem, alpha_key, human-readable label) tuples evaluated for every model
CONFIGS_ORDER: list = [
    ("p3", "alpha3", "P3 + alpha3 (σ3)"),
    ("p5", "alpha3", "P5 + alpha3 (σ3)"),
    ("p3", "alpha5", "P3 + alpha5 (σ5)"),
    ("p5", "alpha5", "P5 + alpha5 (σ5)"),
]


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="K-fold evaluation for MLLM Tasks 1, 2a, 2b"
    )
    p.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_DATA_DIRS.keys()),
        help="Which MLLM to evaluate",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["1", "2a", "2b"],
        choices=["1", "2a", "2b"],
        help="Which tasks to run (default: all)",
    )
    p.add_argument(
        "--base-dir",
        default="/mnt/raid5/neemias/PerceptSent-LLM-approach",
        help="Project root directory",
    )
    p.add_argument(
        "--response-col",
        default="sentiment",
        help="Column name in model response CSV (use 'response' for openai)",
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of K-fold splits",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Save per-task summary CSVs here (optional)",
    )
    return p.parse_args()


# ── Task runners ───────────────────────────────────────────────────────────────


def run_task1(args: argparse.Namespace, base: str) -> dict:
    model_dir = os.path.join(base, "data", MODEL_DATA_DIRS[args.model])
    gt_dir = os.path.join(base, "data", GT_DIR_NAME)
    all_results: dict = {}

    for prob, alpha_key, label in CONFIGS_ORDER:
        gt_path = os.path.join(gt_dir, f"percept_dataset_{alpha_key}_{prob}.csv")
        resp_path = os.path.join(model_dir, f"{alpha_key}{prob}.csv")

        if not os.path.exists(gt_path) or not os.path.exists(resp_path):
            print(f"[Task 1] Skipping '{label}': file not found")
            print(f"         gt  : {gt_path}")
            print(f"         resp: {resp_path}")
            continue

        df_gt = pd.read_csv(gt_path)
        df_resp = pd.read_csv(resp_path)
        df_folds = kfold_task1_eval(
            df_gt, df_resp, args.response_col, args.n_splits
        )

        f1s = df_folds["f1_score"].tolist()
        mean_f1, ci = compute_ci(f1s)
        mean_acc = float(np.mean(df_folds["accuracy"].tolist()))
        all_results[label] = dict(mean_f1=mean_f1, mean_acc=mean_acc, ci=ci, df=df_folds)
        print_task_summary("Task 1", label, df_folds, mean_f1, mean_acc, ci)

    if all_results:
        print("\n--- Task 1 Summary ---")
        print(results_table(all_results).to_string())
        _save_results(args, all_results, "task1")

    return all_results


def run_task_from_logs(
    args: argparse.Namespace, base: str, task_label: str, exp_dir_name: str
) -> dict:
    exp_dir = os.path.join(base, exp_dir_name)
    all_results: dict = {}

    for prob, alpha_key, label in CONFIGS_ORDER:
        exp_name = f"{MODEL_EXP_PREFIX[args.model]}-{prob}-{alpha_key}"
        log_path = os.path.join(exp_dir, exp_name, "logs", "test_logs.csv")

        if not os.path.exists(log_path):
            print(f"[{task_label}] Skipping '{label}': {log_path} not found")
            continue

        res = kfold_log_eval(log_path)
        all_results[label] = res
        print_task_summary(
            task_label, label, res["df"], res["mean_f1"], res["mean_acc"], res["ci"]
        )

    if all_results:
        print(f"\n--- {task_label} Summary ---")
        print(results_table(all_results).to_string())
        _save_results(args, all_results, task_label.replace(" ", "_").lower())

    return all_results


def _save_results(args: argparse.Namespace, all_results: dict, tag: str) -> None:
    if not args.output_dir:
        return
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for cfg_name, res in all_results.items():
        ci = res["ci"]
        rows.append(
            {
                "config": cfg_name,
                "task": tag,
                "mean_f1": res["mean_f1"],
                "mean_acc": res["mean_acc"],
                "ci_low": ci[0],
                "ci_high": ci[1],
            }
        )
    out = os.path.join(args.output_dir, f"{args.model}_{tag}_results.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    base = args.base_dir

    t1_res = t2a_res = t2b_res = {}

    if "1" in args.tasks:
        t1_res = run_task1(args, base)

    if "2a" in args.tasks:
        t2a_res = run_task_from_logs(args, base, "Task 2a", "experiments-not-finetuning")

    if "2b" in args.tasks:
        t2b_res = run_task_from_logs(args, base, "Task 2b", "experiments-finetuning")

    # All-tasks side-by-side comparison (only when all three tasks ran)
    if t1_res and t2a_res and t2b_res:
        configs = [label for _, _, label in CONFIGS_ORDER if label in t1_res]
        comparison = pd.DataFrame(
            {
                "Config": configs,
                "Task 1 F1": [f"{t1_res[c]['mean_f1']:.4f}" for c in configs],
                "Task 2a F1": [
                    f"{t2a_res.get(c, {}).get('mean_f1', float('nan')):.4f}"
                    for c in configs
                ],
                "Task 2b F1": [
                    f"{t2b_res.get(c, {}).get('mean_f1', float('nan')):.4f}"
                    for c in configs
                ],
            }
        ).set_index("Config")
        print("\n=== All-Tasks Comparison ===")
        print(comparison.to_string())


if __name__ == "__main__":
    main()
