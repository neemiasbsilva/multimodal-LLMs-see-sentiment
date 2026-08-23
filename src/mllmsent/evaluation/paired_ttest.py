"""
Paired t-tests with Holm-Bonferroni correction for Task 1 (direct MLLM classification).

Compares VLMoai (GPT-4o) vs. DeepSeek across 4 problem configurations:
  <sigma3, P3>, <sigma3, P5>, <sigma5, P3>, <sigma5, P5>

Evaluation uses stratified 5-fold cross-validation (random_state=42):
each fold's 20% test partition is class-balanced, and both models are
evaluated on the same partition. The fold-wise weighted F1-scores are
the paired observations for the two-sided paired t-test.

P-values are adjusted with the Holm-Bonferroni step-down correction
to control the family-wise error rate at alpha=0.05.

Usage
-----
python scripts/task1_paired_ttest.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats

SENT_MAP_P5 = {"Positive": 4, "Slightlypositive": 3, "Neutral": 2, "Slightlynegative": 1, "Negative": 0}
SENT_MAP_P3 = {"Positive": 2, "Neutral": 0, "Negative": 1}

CONFIGS = [
    ("sigma3_P3", "data/gpt4-openai-classify/percept_dataset_alpha3_p3.csv",
     "data/gpt4-openai-only/alpha3p3.csv", "response",
     "data/deepseek-only/alpha3p3.csv"),
    ("sigma3_P5", "data/gpt4-openai-classify/percept_dataset_alpha3_p5.csv",
     "data/gpt4-openai-only/alpha3p5.csv", "sentiment",
     "data/deepseek-only/alpha3p5.csv"),
    ("sigma5_P3", "data/gpt4-openai-classify/percept_dataset_alpha5_p3.csv",
     "data/gpt4-openai-only/p3-results.csv", "response",
     "data/deepseek-only/alpha5p3.csv"),
    ("sigma5_P5", "data/gpt4-openai-classify/percept_dataset_alpha5_p5.csv",
     "data/gpt4-openai-only/p5-results.csv", "response",
     "data/deepseek-only/alpha5p5.csv"),
]


def map_string(val, sent_map):
    key = str(val).replace(".", "").strip().replace(" ", "")
    if key in sent_map:
        return sent_map[key]
    for k in sent_map:
        if k.lower() == key.lower():
            return sent_map[k]
    return None


def skfold_eval(df_gt, preds, n_splits=5):
    y = df_gt["sentiment"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1s = []
    for _, test_idx in skf.split(df_gt, y):
        test = df_gt.iloc[test_idx]
        y_true = test["sentiment"].tolist()
        y_pred = [preds.get(img_id, -1) for img_id in test["id"]]
        f1s.append(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    return np.array(f1s)


def holm_correction(p_values, alpha=0.05):
    m = len(p_values)
    p = np.asarray(p_values, dtype=float)
    sorted_idx = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, orig_idx in enumerate(sorted_idx):
        step = (m - rank) * p[orig_idx]
        running_max = max(running_max, step)
        adjusted[orig_idx] = min(running_max, 1.0)
    return adjusted <= alpha, adjusted


def main():
    all_raw_p = []
    results = []

    for label, gt_path, oai_path, oai_col, ds_path in CONFIGS:
        df_gt  = pd.read_csv(gt_path)
        df_oai = pd.read_csv(oai_path)
        df_ds  = pd.read_csv(ds_path)

        n_classes = df_gt["sentiment"].nunique()
        sent_map = SENT_MAP_P5 if n_classes == 5 else SENT_MAP_P3

        oai_preds = {}
        for _, row in df_oai.iterrows():
            mapped = map_string(row[oai_col], sent_map)
            if mapped is not None:
                oai_preds[row["id"]] = mapped

        ds_preds = dict(zip(df_ds["id"], df_ds["sentiment"].astype(int)))

        common_ids = set(df_gt["id"]) & set(oai_preds.keys()) & set(ds_preds.keys())
        df_gt_c = (df_gt[df_gt["id"].isin(common_ids)]
                   .sort_values("id").reset_index(drop=True))

        oai_f1s = skfold_eval(df_gt_c, oai_preds)
        ds_f1s  = skfold_eval(df_gt_c, ds_preds)

        t, p_raw = stats.ttest_rel(oai_f1s, ds_f1s)
        all_raw_p.append(p_raw)
        results.append((label, oai_f1s, ds_f1s, t, p_raw, len(df_gt_c)))

        print(f"{label}: n={len(df_gt_c)}")
        print(f"  OAI  folds: {np.round(oai_f1s * 100, 1).tolist()}, mean={oai_f1s.mean() * 100:.1f}%")
        print(f"  DS   folds: {np.round(ds_f1s * 100, 1).tolist()},  mean={ds_f1s.mean() * 100:.1f}%")
        print(f"  t={t:.3f}, p_raw={p_raw:.4f}")

    reject, p_adj = holm_correction(all_raw_p, alpha=0.05)

    print("\n=== Holm-corrected results (alpha=0.05) ===")
    for i, (label, oai_f1s, ds_f1s, t, p_raw, n) in enumerate(results):
        sig = "SIGNIFICANT" if reject[i] else "NOT significant"
        print(
            f"<{label}>  OAI={oai_f1s.mean() * 100:.1f}%  DS={ds_f1s.mean() * 100:.1f}%  "
            f"t={t:.3f}  p_raw={p_raw:.4f}  p_holm={p_adj[i]:.4f}  [{sig}]"
        )


if __name__ == "__main__":
    main()
