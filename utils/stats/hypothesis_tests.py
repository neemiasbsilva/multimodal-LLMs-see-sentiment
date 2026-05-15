from itertools import combinations

import numpy as np
from scipy import stats


def holm_correction(p_values: list, alpha: float = 0.05):
    """
    Holm-Bonferroni step-down correction.

    Returns (reject_array, adjusted_p_array) aligned with the input order.
    """
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=bool), np.array([])

    p = np.asarray(p_values, dtype=float)
    sorted_idx = np.argsort(p)

    adjusted = np.empty(m)
    running_max = 0.0
    for rank, orig_idx in enumerate(sorted_idx):
        step = (m - rank) * p[orig_idx]
        running_max = max(running_max, step)
        adjusted[orig_idx] = min(running_max, 1.0)

    reject = adjusted <= alpha
    return reject, adjusted


def run_paired_ttests_with_holm(settings: dict, alpha: float = 0.05) -> list:
    """
    Run two-sided paired t-tests for every model pair within each setting,
    then apply a single global Holm-Bonferroni correction.

    Parameters
    ----------
    settings : dict[(exp_label, problem, alpha_tag)] -> dict[model -> ndarray]
    alpha    : family-wise error rate

    Returns
    -------
    List of dicts, one per comparison, with keys:
        exp_label, problem, alpha_tag, model_a, model_b,
        mean_a, mean_b, mean_diff, t_stat, p_raw, p_holm, significant
    """
    raw_rows = []

    for (exp_label, problem, alpha_tag), model_data in sorted(settings.items()):
        model_names = sorted(model_data.keys())
        for m_a, m_b in combinations(model_names, 2):
            scores_a = model_data[m_a]
            scores_b = model_data[m_b]
            if len(scores_a) != len(scores_b):
                continue
            t_stat, p_raw = stats.ttest_rel(scores_a, scores_b)
            raw_rows.append({
                "exp_label": exp_label,
                "problem": problem,
                "alpha_tag": alpha_tag,
                "model_a": m_a,
                "model_b": m_b,
                "mean_a": float(np.mean(scores_a)),
                "mean_b": float(np.mean(scores_b)),
                "mean_diff": float(np.mean(scores_a - scores_b)),
                "t_stat": float(t_stat),
                "p_raw": float(p_raw),
            })

    if not raw_rows:
        return raw_rows

    reject, p_adj = holm_correction([r["p_raw"] for r in raw_rows], alpha=alpha)

    for i, row in enumerate(raw_rows):
        row["p_holm"] = float(p_adj[i])
        row["significant"] = bool(reject[i])

    return raw_rows
