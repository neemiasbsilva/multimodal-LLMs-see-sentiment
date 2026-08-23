"""Visualization utilities for PerceptSent-LLM results.

All public functions accept the dict loaded from reports/results.json so
notebooks stay free of hardcoded data — they only load the JSON and call
these functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence


# ── helpers ────────────────────────────────────────────────────────────────────

def _records_to_df(results: dict) -> pd.DataFrame:
    """Return comprehensive records as a tidy DataFrame."""
    return pd.DataFrame(results["comprehensive"]["records"])


def _scatter_markers() -> list:
    return ["o", "s", "D", "^", "v", "p", "h", "*"]


# ── public plot functions ──────────────────────────────────────────────────────

def scatter_by_model(
    results: dict,
    problem: str,
    ax: plt.Axes | None = None,
    figsize: tuple = (15, 6),
) -> plt.Axes:
    """Scatter plot of σ=5 F1 scores for one classification problem.

    Parameters
    ----------
    results : dict loaded from reports/results.json
    problem : one of 'P5', 'P3', 'P2+', 'P2-'
    ax      : existing Axes; a new figure is created if None
    """
    s5 = results["scatter_sigma5"]
    caption_models = results["caption_models"]
    methods = s5["methods"]
    f1_matrix = s5["f1_scores"][problem]

    x_values = np.arange(len(caption_models))
    markers = _scatter_markers()
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for i, (method, scores) in enumerate(zip(methods, f1_matrix)):
        ax.scatter(
            x_values,
            scores,
            label=method,
            marker=markers[i % len(markers)],
            s=200,
            edgecolors="black",
            alpha=0.8,
            color=colors[i],
        )

    ax.set_xticks(x_values)
    ax.set_xticklabels(caption_models, rotation=20)
    ax.set_xlabel("Approaches")
    ax.set_ylabel("F1-score (%)")
    ax.set_title(f"F1-score — Threshold σ=5, Classification Problem {problem}")
    ax.legend(loc="lower left", bbox_to_anchor=(1, 0), fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    return ax


def scatter_grid(
    results: dict,
    figsize: tuple = (18, 14),
) -> plt.Figure:
    """2×2 grid of scatter plots for all four classification problems (σ=5)."""
    problems = results["classification_problems"]
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, problem in zip(axes.flat, problems):
        scatter_by_model(results, problem, ax=ax)
    # Share a single legend from the last axes
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    for ax in axes.flat:
        ax.get_legend().remove()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("F1-score Comparison — Threshold σ=5 (all problems)", fontsize=14)
    fig.tight_layout()
    return fig


def comprehensive_errorbar(
    results: dict,
    selected_approaches: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple = (20, 8),
) -> plt.Axes:
    """Error-bar line plot across all classification problems for every threshold.

    Parameters
    ----------
    results              : dict loaded from reports/results.json
    selected_approaches  : subset of approach names to plot; all if None
    ax                   : existing Axes; a new figure is created if None
    """
    df = _records_to_df(results)
    styles = results["comprehensive"]["approach_styles"]
    problems = results["classification_problems"]

    if selected_approaches is None:
        selected_approaches = list(styles.keys())

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for approach in selected_approaches:
        style = styles.get(approach, {"color": "black", "marker": "o", "linestyle": "-"})
        sub = df[df["approach"] == approach]

        for threshold in results["thresholds"]:
            t_sub = sub[sub["threshold"] == threshold].set_index("problem")
            scores = [t_sub.loc[p, "f1"] if p in t_sub.index else np.nan for p in problems]
            errs   = [t_sub.loc[p, "err"] if p in t_sub.index else np.nan for p in problems]

            ax.errorbar(
                problems,
                scores,
                yerr=errs,
                label=f"{approach} (σ={threshold})",
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                capsize=4,
                alpha=0.85,
            )

    ax.set_xlabel("Problem setups")
    ax.set_ylabel("F1-Score (%)")
    ax.set_ylim(0, 100)
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("F1-Score Comparison by Approach and Problem Setup")
    return ax


def confidence_band_plot(
    results: dict,
    selected_approaches: Sequence[str],
    threshold: int = 5,
    ax: plt.Axes | None = None,
    figsize: tuple = (20, 8),
) -> plt.Axes:
    """Error-bar + shaded confidence band for selected approaches at one threshold.

    Parameters
    ----------
    selected_approaches : list of approach names to compare
    threshold           : σ value to show (3, 4, or 5)
    """
    df = _records_to_df(results)
    styles = results["comprehensive"]["approach_styles"]
    problems = results["classification_problems"]

    threshold_colors = {3: "blue", 4: "red", 5: "green"}
    color = threshold_colors.get(threshold, "black")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for approach in selected_approaches:
        style = styles.get(approach, {"marker": "o"})
        sub = df[(df["approach"] == approach) & (df["threshold"] == threshold)].set_index("problem")
        scores = np.array([sub.loc[p, "f1"] if p in sub.index else np.nan for p in problems])
        errs   = np.array([sub.loc[p, "err"] if p in sub.index else np.nan for p in problems])

        ax.errorbar(problems, scores, yerr=errs,
                    fmt=style["marker"], color=color,
                    linestyle="--", linewidth=2,
                    label=f"{approach} (σ={threshold})", capsize=4)
        ax.fill_between(problems, scores - errs, scores + errs, color=color, alpha=0.15)

    ax.set_xlabel("Problem setups")
    ax.set_ylabel("F1-Score (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(problems)))
    ax.set_xticklabels(problems)
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title(f"F1-Score with Confidence Bands — σ={threshold}")
    return ax


def bar_comparison(
    results: dict,
    base_approaches: Sequence[str],
    thresholds: Sequence[int] | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple = (15, 6),
) -> plt.Axes:
    """Grouped bar chart comparing selected approaches × thresholds across problems.

    Parameters
    ----------
    base_approaches : approach names (without σ suffix) to include
    thresholds      : list of σ values to include; defaults to all three
    """
    df = _records_to_df(results)
    problems = results["classification_problems"]

    if thresholds is None:
        thresholds = results["thresholds"]

    # Build ordered series: one bar group per (approach, threshold) combo
    series_labels = [f"{a} (σ={t})" for t in thresholds for a in base_approaches]
    n_series = len(series_labels)
    bar_width = 0.12
    group_space = 0.3
    x_centers = np.arange(len(problems)) * (n_series * bar_width + group_space)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for s_idx, (threshold, approach) in enumerate(
        [(t, a) for t in thresholds for a in base_approaches]
    ):
        sub = df[(df["approach"] == approach) & (df["threshold"] == threshold)].set_index("problem")
        scores = np.array([sub.loc[p, "f1"] if p in sub.index else np.nan for p in problems])
        errs   = np.array([sub.loc[p, "err"] if p in sub.index else np.nan for p in problems])
        ax.bar(
            x_centers + s_idx * bar_width,
            scores,
            bar_width,
            yerr=errs,
            label=f"{approach} (σ={threshold})",
            capsize=5,
        )

    ax.set_xlabel("Problem Setups")
    ax.set_ylabel("F1-score (%)")
    ax.set_title("F1-Score Comparison by Approach and Problem Setup")
    ax.set_xticks(x_centers + bar_width * (n_series - 1) / 2)
    ax.set_xticklabels(problems)
    ax.legend(title="Model Configurations", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return ax
