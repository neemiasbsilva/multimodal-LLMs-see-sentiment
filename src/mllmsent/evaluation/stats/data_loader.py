import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd


# Models excluded from statistical comparisons per paper scope
EXCLUDED_MODELS = {"distil"}

# Caption sources that are restricted to a single classifier in the paper
_CAPTION_CLASSIFIER_ONLY = {
    "gemini": "modernbert",
    "phi4": "modernbert",
}

# Pattern: {model_name}-{problem}-{alpha}
# problem: p3 | p5 | p2neg | p2plus
# alpha: alpha3 | alpha5
_FOLDER_RE = re.compile(r"^(.+)-(p(?:3|5|2neg|2plus))-(alpha[35])$")


def parse_experiment_name(folder_name: str):
    """Return (model_name, problem, alpha) or None if folder doesn't match."""
    m = _FOLDER_RE.match(folder_name)
    if m is None:
        return None
    raw_model, problem, alpha = m.groups()
    # Strip the training-regime keyword that adds no identity information
    model_name = raw_model.replace("-experiment", "")
    return model_name, problem, alpha


def _is_excluded(model_name: str) -> bool:
    parts = model_name.split("-")
    parts_set = set(parts)
    if parts_set & EXCLUDED_MODELS:
        return True
    # For restricted caption sources, only keep the allowed classifier
    for caption_src, required_classifier in _CAPTION_CLASSIFIER_ONLY.items():
        if parts[0] == caption_src and required_classifier not in parts_set:
            return True
    return False


def load_results_from_dir(root_dir: str, metric: str = "f1_score") -> dict:
    """
    Load fold-wise metric values for every valid experiment under *root_dir*.

    Returns
    -------
    dict[(model_name, problem, alpha)] -> np.ndarray of shape (n_folds,)
    """
    results = {}
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Experiment directory not found: {root_dir}")

    for folder in sorted(os.listdir(root_dir)):
        parsed = parse_experiment_name(folder)
        if parsed is None:
            continue
        model_name, problem, alpha = parsed
        if _is_excluded(model_name):
            continue

        log_path = os.path.join(root_dir, folder, "logs", "test_logs.csv")
        if not os.path.isfile(log_path):
            continue

        df = pd.read_csv(log_path)
        if metric not in df.columns:
            continue
        # Sort by fold to guarantee alignment across models
        df = df.sort_values("kfold").reset_index(drop=True)
        results[(model_name, problem, alpha)] = df[metric].to_numpy()

    return results


def organize_by_setting(
    results: dict, exp_label: str
) -> dict:
    """
    Group results by (exp_label, problem, alpha).

    Returns
    -------
    dict[(exp_label, problem, alpha)] -> dict[model_name -> np.ndarray]
    """
    settings = defaultdict(dict)
    for (model, problem, alpha), scores in results.items():
        settings[(exp_label, problem, alpha)][model] = scores
    return dict(settings)
