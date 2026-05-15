"""
Paired t-tests with Holm–Bonferroni correction across all model pairs.

All statistical comparisons use fold-wise results from the stratified 5-fold
cross-validation procedure. Pairwise model comparisons are performed using
two-sided paired t-tests. P-values are adjusted with the Holm–Bonferroni
correction to control the family-wise error rate at alpha = 0.05.

Usage
-----
python scripts/statistical_tests.py \
    --exp-dirs experiments-finetuning experiments-not-finetuning \
    --metric f1_score \
    --output reports/stats/holm_ttests \
    --alpha 0.05
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stats.data_loader import load_results_from_dir, organize_by_setting
from utils.stats.hypothesis_tests import run_paired_ttests_with_holm
from utils.stats.reporter import (
    generate_latex_table,
    print_summary,
    results_to_dataframe,
    save_csv,
    save_latex,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Paired t-tests + Holm–Bonferroni correction across experiment results."
    )
    p.add_argument(
        "--exp-dirs",
        nargs="+",
        default=["experiments-finetuning", "experiments-not-finetuning"],
        metavar="DIR",
        help="Experiment directories (relative to project root).",
    )
    p.add_argument(
        "--metric",
        default="f1_score",
        choices=["f1_score", "accuracy"],
        help="Fold-wise metric to compare (default: f1_score).",
    )
    p.add_argument(
        "--output",
        default="reports/stats/holm_ttests",
        help="Output path prefix — .csv and .tex are appended automatically.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Family-wise error rate for Holm correction (default: 0.05).",
    )
    p.add_argument(
        "--significant-only",
        action="store_true",
        help="Include only significant comparisons in the LaTeX table.",
    )
    p.add_argument(
        "--table-width",
        default="textwidth",
        metavar="WIDTH",
        help=(
            "Width of generated LaTeX tables. "
            "Aliases: 'textwidth' (default, full-width / two-column), "
            "'columnwidth' (single-column or one-column in two-column layout), "
            "'linewidth'. Any raw LaTeX expression is passed through unchanged "
            "(e.g. '0.85\\\\textwidth')."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    all_settings = {}

    for exp_dir_rel in args.exp_dirs:
        exp_dir = os.path.join(project_root, exp_dir_rel)
        if not os.path.isdir(exp_dir):
            print(f"[WARNING] Directory not found, skipping: {exp_dir}")
            continue

        # Use the directory name as the experiment group label
        exp_label = exp_dir_rel.replace("experiments-", "")
        results = load_results_from_dir(exp_dir, metric=args.metric)
        print(f"Loaded {len(results)} experiments from '{exp_dir_rel}'")

        group_settings = organize_by_setting(results, exp_label)
        all_settings.update(group_settings)

    if not all_settings:
        print("No experiment data found. Exiting.")
        sys.exit(1)

    n_models_per_setting = {k: len(v) for k, v in all_settings.items()}
    total_comparisons_upper = sum(
        n * (n - 1) // 2 for n in n_models_per_setting.values()
    )
    print(
        f"\nSettings found: {len(all_settings)} | "
        f"Expected comparisons (upper bound): {total_comparisons_upper}"
    )

    test_results = run_paired_ttests_with_holm(all_settings, alpha=args.alpha)
    df = results_to_dataframe(test_results)

    print_summary(df)

    # Resolve absolute output paths
    out_prefix = (
        args.output
        if os.path.isabs(args.output)
        else os.path.join(project_root, args.output)
    )
    save_csv(df, out_prefix + ".csv")

    tex = generate_latex_table(
        df,
        caption=(
            "Pairwise model comparisons via two-sided paired t-tests. "
            "P-values are adjusted using the Holm--Bonferroni correction "
            r"($\alpha = 0.05$). Bold rows are significant after correction. "
            r"$\Delta\bar{F_1}$ is the mean fold-wise difference (Model A $-$ Model B)."
        ),
        label="tab:holm_ttests",
        significant_only=args.significant_only,
        table_width=args.table_width,
    )
    save_latex(tex, out_prefix + ".tex")


if __name__ == "__main__":
    main()
