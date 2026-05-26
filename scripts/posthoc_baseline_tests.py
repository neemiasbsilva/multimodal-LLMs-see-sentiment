#!/usr/bin/env python3
"""Targeted post-hoc tests for MLLMsent against manuscript baselines.

This script supports comparing MLLMsent (GPT (OAI) + MBERT,
Task 2b) with Swin Transformer, ResNet CNN, and VADER baselines. Swin and
VADER are compared with paired t-tests over the shared 5 folds. ResNet is
available in this repository only as a mean +/- 95% CI, so it is compared with
a Welch t-test reconstructed from the reported CI assuming n=5 folds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.stats.hypothesis_tests import holm_correction


CONFIGS = [
    ("p5", "alpha3"),
    ("p3", "alpha3"),
    ("p5", "alpha5"),
    ("p3", "alpha5"),
]

PROBLEM_TEX = {
    "p3": r"$P_3$",
    "p5": r"$P_5$",
}

ALPHA_TEX = {
    "alpha3": r"$\sigma{=}3$",
    "alpha5": r"$\sigma{=}5$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc tests comparing MLLMsent with named baselines."
    )
    parser.add_argument(
        "--metric",
        default="f1_score",
        help="Fold-wise metric column used for paired comparisons.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Family-wise alpha for Holm-Bonferroni correction.",
    )
    parser.add_argument(
        "--output",
        default="reports/stats/posthoc_baselines_mllmsent",
        help="Output prefix. .csv and .tex are appended.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_fold_scores(path: Path, metric: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")
    df = pd.read_csv(path)
    if metric not in df.columns:
        raise ValueError(f"Column '{metric}' not found in {path}")
    if "kfold" in df.columns:
        df = df.sort_values("kfold")
    return df[metric].to_numpy(dtype=float)


def mllmsent_scores(root: Path, problem: str, alpha_tag: str, metric: str) -> np.ndarray:
    path = (
        root
        / "experiments-finetuning"
        / f"openai-modernbert-experiment-{problem}-{alpha_tag}"
        / "logs"
        / "test_logs.csv"
    )
    return load_fold_scores(path, metric)


def swin_scores(root: Path, problem: str, alpha_tag: str, metric: str) -> np.ndarray:
    path = (
        root
        / "experiments-swin"
        / f"openai-swin-experiment-{problem}-{alpha_tag}"
        / "logs"
        / "test_logs.csv"
    )
    return load_fold_scores(path, metric)


def vader_scores(root: Path, alpha_tag: str, metric: str) -> np.ndarray:
    path = root / "experiments-vader" / f"openai-percept_dataset_{alpha_tag}_p3.csv"
    return load_fold_scores(path, metric)


def load_resnet_summary(root: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """Return {(problem, alpha_tag): (mean, ci_half_width)} in 0-1 scale."""
    path = root / "reports" / "results.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ResNet summary source: {path}")

    with path.open() as f:
        payload = json.load(f)

    rows = {}
    for row in payload["comprehensive"]["records"]:
        if row.get("approach") != "ResNet - (Benchmark)":
            continue
        problem = str(row["problem"]).lower()
        threshold = int(row["threshold"])
        if problem not in {"p3", "p5"} or threshold not in {3, 5}:
            continue
        rows[(problem, f"alpha{threshold}")] = (row["f1"] / 100, row["err"] / 100)

    expected = set(CONFIGS)
    missing = expected - set(rows)
    if missing:
        missing_str = ", ".join(f"{p}-{a}" for p, a in sorted(missing))
        raise ValueError(f"Missing ResNet summary values for: {missing_str}")
    return rows


def paired_row(
    problem: str,
    alpha_tag: str,
    baseline: str,
    mllm_scores: np.ndarray,
    baseline_scores: np.ndarray,
) -> dict:
    if len(mllm_scores) != len(baseline_scores):
        raise ValueError(
            f"{baseline} fold count differs for {problem}-{alpha_tag}: "
            f"{len(mllm_scores)} vs {len(baseline_scores)}"
        )
    t_stat, p_raw = stats.ttest_rel(mllm_scores, baseline_scores)
    return {
        "problem": problem,
        "alpha_tag": alpha_tag,
        "baseline": baseline,
        "test": "paired_t_test",
        "n_mllmsent": len(mllm_scores),
        "n_baseline": len(baseline_scores),
        "mean_mllmsent": float(np.mean(mllm_scores)),
        "mean_baseline": float(np.mean(baseline_scores)),
        "mean_diff": float(np.mean(mllm_scores - baseline_scores)),
        "t_stat": float(t_stat),
        "df": float(len(mllm_scores) - 1),
        "p_raw": float(p_raw),
        "baseline_ci_half_width": np.nan,
        "note": "Fold-wise paired comparison.",
    }


def welch_from_summary_row(
    problem: str,
    alpha_tag: str,
    baseline: str,
    mllm_scores: np.ndarray,
    baseline_mean: float,
    baseline_ci_half_width: float,
) -> dict:
    n_mllm = len(mllm_scores)
    n_baseline = 5
    mllm_mean = float(np.mean(mllm_scores))
    mllm_sd = float(np.std(mllm_scores, ddof=1))

    tcrit = stats.t.ppf(0.975, n_baseline - 1)
    baseline_sd = baseline_ci_half_width * math.sqrt(n_baseline) / tcrit

    se2 = (mllm_sd**2 / n_mllm) + (baseline_sd**2 / n_baseline)
    t_stat = (mllm_mean - baseline_mean) / math.sqrt(se2)
    df_num = se2**2
    df_den = ((mllm_sd**2 / n_mllm) ** 2 / (n_mllm - 1)) + (
        (baseline_sd**2 / n_baseline) ** 2 / (n_baseline - 1)
    )
    df = df_num / df_den
    p_raw = 2 * stats.t.sf(abs(t_stat), df)

    return {
        "problem": problem,
        "alpha_tag": alpha_tag,
        "baseline": baseline,
        "test": "welch_t_from_summary_ci",
        "n_mllmsent": n_mllm,
        "n_baseline": n_baseline,
        "mean_mllmsent": mllm_mean,
        "mean_baseline": baseline_mean,
        "mean_diff": mllm_mean - baseline_mean,
        "t_stat": float(t_stat),
        "df": float(df),
        "p_raw": float(p_raw),
        "baseline_ci_half_width": baseline_ci_half_width,
        "note": "ResNet fold logs unavailable; SD reconstructed from reported 95% CI.",
    }


def build_results(root: Path, metric: str, alpha: float) -> pd.DataFrame:
    resnet = load_resnet_summary(root)
    rows = []

    for problem, alpha_tag in CONFIGS:
        mllm = mllmsent_scores(root, problem, alpha_tag, metric)
        rows.append(
            paired_row(
                problem,
                alpha_tag,
                "Swin Transformer",
                mllm,
                swin_scores(root, problem, alpha_tag, metric),
            )
        )
        baseline_mean, baseline_ci_half_width = resnet[(problem, alpha_tag)]
        rows.append(
            welch_from_summary_row(
                problem,
                alpha_tag,
                "ResNet CNN",
                mllm,
                baseline_mean,
                baseline_ci_half_width,
            )
        )

    for alpha_tag in ["alpha3", "alpha5"]:
        problem = "p3"
        rows.append(
            paired_row(
                problem,
                alpha_tag,
                "VADER",
                mllmsent_scores(root, problem, alpha_tag, metric),
                vader_scores(root, alpha_tag, metric),
            )
        )

    reject, p_holm = holm_correction([row["p_raw"] for row in rows], alpha=alpha)
    for row, is_significant, adjusted in zip(rows, reject, p_holm):
        row["p_holm"] = float(adjusted)
        row["significant"] = bool(is_significant)

    return pd.DataFrame(rows).sort_values(
        ["alpha_tag", "problem", "baseline"]
    ).reset_index(drop=True)


def fmt_percent(value: float) -> str:
    return f"{value * 100:.2f}"


def fmt_p(value: float) -> str:
    if value < 0.001:
        return r"$<$0.001"
    return f"{value:.4f}"


def blue(value: str) -> str:
    return r"\textcolor{blue}{" + value + "}"


def generate_latex(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        (
            r"\caption{\textcolor{blue}{Targeted post-hoc comparisons between "
            r"MLLMsent and the baselines named in Fig.~\ref{fig:plotBaselinesAndSentVLMs}. "
            r"Swin Transformer and VADER use paired $t$-tests over 5 folds; "
            r"ResNet CNN uses a Welch test reconstructed from the reported mean "
            r"and 95\% CI because fold-wise logs were unavailable. "
            r"$p$-values are Holm--Bonferroni adjusted across the 10 comparisons.}}"
        ),
        r"\label{tab:posthoc_baselines_mllmsent}",
        r"\begin{tabular}{lllrrrr}",
        r"\hline",
        (
            r"\textbf{Problem} & \textbf{$\sigma$} & \textbf{Baseline} & "
            r"\textbf{MLLMsent $F_1$} & \textbf{Baseline $F_1$} & "
            r"\textbf{$\Delta F_1$} & \textbf{$p_\text{Holm}$} \\"
        ),
        r"\hline",
    ]
    for _, row in df.iterrows():
        cells = [
            PROBLEM_TEX[row["problem"]],
            ALPHA_TEX[row["alpha_tag"]],
            row["baseline"],
            blue(fmt_percent(row["mean_mllmsent"])),
            blue(fmt_percent(row["mean_baseline"])),
            blue(fmt_percent(row["mean_diff"])),
            blue(fmt_p(row["p_holm"])),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def save_outputs(df: pd.DataFrame, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    tex_path = output_prefix.with_suffix(".tex")
    df.to_csv(csv_path, index=False)
    tex_path.write_text(generate_latex(df))
    print(f"CSV saved: {csv_path}")
    print(f"LaTeX saved: {tex_path}")


def main() -> None:
    args = parse_args()
    root = project_root()
    output_prefix = Path(args.output)
    if not output_prefix.is_absolute():
        output_prefix = root / output_prefix

    df = build_results(root, args.metric, args.alpha)
    save_outputs(df, output_prefix)

    print(f"Comparisons: {len(df)}")
    print(f"Baselines: {', '.join(sorted(df['baseline'].unique()))}")
    print(f"Max Holm-adjusted p-value: {df['p_holm'].max():.6g}")
    print(f"Significant comparisons: {int(df['significant'].sum())}/{len(df)}")


if __name__ == "__main__":
    main()
