import os

import pandas as pd


_PROBLEM_LABELS = {
    "p3": r"$P_3$",
    "p5": r"$P_5$",
    "p2neg": r"$P_2^-$",
    "p2plus": r"$P_2^+$",
}

_ALPHA_LABELS = {
    "alpha3": r"$\sigma{=}3$",
    "alpha5": r"$\sigma{=}5$",
}

# Display name for each known MLLM caption source.
# Models with no recognised caption prefix belong to MiniGPT-4 (Open Source).
_CAPTION_DISPLAY = {
    "openai":   "GPT (OAI)",
    "deepseek": "DeepSeek",
    "gemini":   "Gemini",
    "gemma4":   "Gemma4",
    "phi4":     "Phi-4",
}

# Display name for each known text classifier.
_CLASSIFIER_DISPLAY = {
    "bart":       "BART",
    "modernbert": "MBERT",
    "llama3":     "LLaMA",
}


def _fmt_model(name: str) -> str:
    """Convert a raw model slug to a human-readable display name.

    Format: ``{MLLM} + {Classifier}``

    Models whose first token is not a known MLLM caption source are treated as
    belonging to MiniGPT-4 (Open Source) and labelled ``GPT (OS)``.
    """
    parts = name.split("-")
    if parts[0] in _CAPTION_DISPLAY:
        mllm = _CAPTION_DISPLAY[parts[0]]
        classifier_parts = parts[1:]
    else:
        mllm = "GPT (OS)"
        classifier_parts = parts

    # Walk the classifier tokens and pick the first recognised keyword.
    classifier = next(
        (_CLASSIFIER_DISPLAY[p] for p in classifier_parts if p in _CLASSIFIER_DISPLAY),
        "-".join(classifier_parts),  # fallback: use the raw remainder
    )
    return f"{mllm} + {classifier}"


def results_to_dataframe(test_results: list) -> pd.DataFrame:
    df = pd.DataFrame(test_results)
    if df.empty:
        return df
    df = df.sort_values(["exp_label", "problem", "alpha_tag", "p_holm"]).reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No test results to display.")
        return

    total = len(df)
    n_sig = int(df["significant"].sum())
    print(f"\nTotal pairwise comparisons: {total}")
    print(f"Significant after Holm–Bonferroni correction (alpha=0.05): {n_sig}")

    for exp_label, grp in df.groupby("exp_label"):
        print(f"\n{'='*60}")
        print(f"Group: {exp_label}")
        print(f"{'='*60}")
        for _, row in grp.iterrows():
            sig_marker = "*" if row["significant"] else " "
            print(
                f"  [{sig_marker}] [{row['problem']}, {row['alpha_tag']}] "
                f"{row['model_a']} vs {row['model_b']}: "
                f"t={row['t_stat']:+.4f}, "
                f"mean_diff={row['mean_diff']*100:+.2f}%, "
                f"p_raw={row['p_raw']:.4f}, "
                f"p_holm={row['p_holm']:.4f}"
            )


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"CSV saved → {path}")


# Maps user-facing width aliases to LaTeX macros
_WIDTH_ALIASES = {
    "textwidth": r"\textwidth",
    "columnwidth": r"\columnwidth",
    "linewidth": r"\linewidth",
}


def _resolve_width(table_width: str) -> str:
    """Return the LaTeX width macro for a user-supplied alias or raw string."""
    return _WIDTH_ALIASES.get(table_width.lower().strip(), table_width)


def _table_block(
    group_rows: pd.DataFrame,
    caption: str,
    label: str,
    header: str,
    table_width: str = r"\textwidth",
) -> list:
    """Return lines for one table*/table block (one group).

    *table_width* is the LaTeX width macro used for both the float environment
    width and the tabular* column span (e.g. ``\\textwidth``, ``\\columnwidth``).
    Single-column templates use ``\\columnwidth``; two-column templates use
    ``\\textwidth`` for a full-page-wide table or ``\\columnwidth`` to stay
    within one column.
    """
    # table* spans both columns; table stays in one column
    float_env = "table*" if table_width == r"\textwidth" else "table"
    lines = [
        rf"\begin{{{float_env}}}[htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        rf"\begin{{tabular*}}{{{table_width}}}{{@{{\extracolsep{{\fill}}}}llllrrrl}}",
        r"\hline",
        header,
        r"\hline",
    ]
    for _, row in group_rows.iterrows():
        prob_tex = _PROBLEM_LABELS.get(row["problem"], row["problem"])
        alpha_tex = _ALPHA_LABELS.get(row["alpha_tag"], row["alpha_tag"])
        diff_pct = row["mean_diff"] * 100
        cells = [
            row["exp_label"],
            prob_tex,
            alpha_tex,
            _fmt_model(row["model_a"]),
            _fmt_model(row["model_b"]),
            f"{diff_pct:+.2f}",
            _fmt_pval(row["p_raw"]),
            _fmt_pval(row["p_holm"]),
        ]
        if row["significant"]:
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\hline",
        r"\end{tabular*}",
        rf"\end{{{float_env}}}",
    ]
    return lines


def generate_latex_table(
    df: pd.DataFrame,
    caption: str = "Pairwise paired t-test results with Holm–Bonferroni correction.",
    label: str = "tab:holm_ttests",
    significant_only: bool = False,
    table_width: str = "textwidth",
) -> str:
    """
    Build a LaTeX string where each (exp, problem, alpha) group is its own
    table/table* using tabular*, separated by \\newpage.

    Parameters
    ----------
    table_width : str
        Controls the width of every generated table.
        Recognised aliases: ``textwidth`` (default, two-column full-width),
        ``columnwidth`` (single-column or one-column-wide in two-column),
        ``linewidth``.  Any raw LaTeX expression (e.g. ``0.85\\textwidth``) is
        passed through unchanged.
    significant_only : bool
        When True, include only rows that survive the Holm correction.
    """
    if df.empty:
        return "% No results to tabulate.\n"

    rows = df[df["significant"]] if significant_only else df
    width_macro = _resolve_width(table_width)

    header = (
        r"\textbf{Exp.} & \textbf{Problem} & \textbf{$\sigma$} & "
        r"\textbf{Model A} & \textbf{Model B} & "
        r"\textbf{$\Delta \bar{F_1}$ (\%)} & "
        r"\textbf{$p_\text{raw}$} & \textbf{$p_\text{Holm}$} \\"
    )

    group_keys = ["exp_label", "problem", "alpha_tag"]
    groups = list(rows.groupby(group_keys, sort=False))

    blocks = []
    for i, (key, grp) in enumerate(groups):
        exp_label, problem, alpha_tag = key
        group_caption = (
            caption
            + f" Group: {exp_label}, {_PROBLEM_LABELS.get(problem, problem)},"
            f" {_ALPHA_LABELS.get(alpha_tag, alpha_tag)}."
        )
        group_label = f"{label}_{exp_label}_{problem}_{alpha_tag}"
        blocks.append(
            "\n".join(
                _table_block(grp, group_caption, group_label, header, width_macro)
            )
        )

    return "\n\\newpage\n".join(blocks) + "\n"


def _fmt_pval(p: float) -> str:
    if p < 0.001:
        return r"$<$0.001"
    return f"{p:.4f}"


def save_latex(tex_str: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(tex_str)
    print(f"LaTeX table saved → {path}")
