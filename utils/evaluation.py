import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

SENT_MAP_P5: dict = {
    "Positive": 4,
    "Slightlypositive": 3,
    "Neutral": 2,
    "Slightlynegative": 1,
    "Negative": 0,
}
SENT_MAP_P3: dict = {"Positive": 2, "Neutral": 0, "Negative": 1}


def get_sent_map(n_classes: int) -> dict:
    """Return the string→int mapping for P5 or P3 (fallback)."""
    return SENT_MAP_P5 if n_classes == 5 else SENT_MAP_P3


def compute_ci(values: list, confidence: float = 0.95) -> tuple:
    """Return (mean, (ci_low, ci_high)) via Student-t interval."""
    mean = float(np.mean(values))
    ci = stats.t.interval(
        confidence, len(values) - 1, loc=mean, scale=stats.sem(values)
    )
    return mean, ci


def kfold_task1_eval(
    df_gt: pd.DataFrame,
    df_resp: pd.DataFrame,
    response_col: str = "sentiment",
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """K-fold evaluation for Task 1 (MLLM direct classification).

    Matches samples by 'id', selects the intersection, then runs KFold.
    df_gt must have integer 'sentiment' labels.
    df_resp[response_col] holds raw sentiment strings (mapped via SENT_MAP_P5/P3).

    Returns DataFrame with columns: fold, accuracy, f1_score.
    """
    df_resp = df_resp.copy()
    df_resp[response_col] = (
        df_resp[response_col].astype(str).str.replace(".", "", regex=False).str.strip()
    )

    common_ids = set(df_gt["id"]) & set(df_resp["id"])
    df_gt = (
        df_gt[df_gt["id"].isin(common_ids)].sort_values("id").reset_index(drop=True)
    )
    df_resp = (
        df_resp[df_resp["id"].isin(common_ids)]
        .sort_values("id")
        .reset_index(drop=True)
    )

    sent_map = get_sent_map(df_gt["sentiment"].nunique())
    resp_lookup: dict = df_resp.set_index("id")[response_col].to_dict()

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for fold, (_, val_idx) in enumerate(kfold.split(df_gt)):
        val_gt = df_gt.iloc[val_idx]
        target = val_gt["sentiment"].tolist()

        pred = []
        for img_id in val_gt["id"]:
            raw = resp_lookup[img_id]
            key = raw.replace(" ", "")
            if key not in sent_map:
                key = next(
                    (k for k in sent_map if k.lower() == key.lower()),
                    list(sent_map.keys())[0],
                )
            pred.append(sent_map[key])

        rows.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(target, pred),
                "f1_score": f1_score(target, pred, average="weighted"),
            }
        )
    return pd.DataFrame(rows)


def kfold_log_eval(log_path: str) -> dict:
    """Read an experiment log CSV and compute aggregate statistics.

    Expected columns: kfold (or fold), accuracy, f1_score.
    Returns dict with mean_f1, mean_acc, ci (tuple), df (DataFrame).
    """
    df = pd.read_csv(log_path)
    f1s = df["f1_score"].tolist()
    accs = df["accuracy"].tolist()
    mean_f1, ci = compute_ci(f1s)
    mean_acc = float(np.mean(accs))
    return dict(mean_f1=mean_f1, mean_acc=mean_acc, ci=ci, df=df)


def print_task_summary(
    task: str,
    cfg_name: str,
    df_folds: pd.DataFrame,
    mean_f1: float,
    mean_acc: float,
    ci: tuple,
) -> None:
    f1s = df_folds["f1_score"].tolist()
    print(f"\n{'='*55}")
    print(f"  {task}  |  {cfg_name}")
    print(f"{'='*55}")
    print(df_folds.to_string(index=False))
    print(f"Max  F1  : {max(f1s):.4f}")
    print(f"Mean F1  : {mean_f1:.4f}")
    print(f"Mean Acc : {mean_acc:.4f}")
    print(f"95% CI   : [{ci[0]:.4f}, {ci[1]:.4f}]  ±{abs(ci[1] - mean_f1):.4f}")


def results_table(all_results: dict) -> pd.DataFrame:
    """Build a compact summary DataFrame from a dict of kfold result dicts."""
    rows = []
    for cfg_name, res in all_results.items():
        ci = res["ci"]
        rows.append(
            {
                "Config": cfg_name,
                "Mean F1": f"{res['mean_f1']:.4f}",
                "Mean Acc": f"{res['mean_acc']:.4f}",
                "95% CI": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
                "±": f"{abs(ci[1] - res['mean_f1']):.4f}",
            }
        )
    return pd.DataFrame(rows).set_index("Config")
