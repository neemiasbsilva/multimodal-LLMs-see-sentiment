"""
Attention Flow / Rollout visualisation for ModernBERT sentiment classifier.

Implements Attention Rollout (Abnar & Zuidema, 2020) for the FULL token×token
attention flow — not restricted to the [CLS] row.

The full rollout matrix R (seq_len × seq_len) is computed by chaining all
global-attention layers with residual connections:

    R[i, j]  =  how much of token j's identity contributed to token i
                after all transformer layers.

Visualisation (--heatmap):
    One PDF per sample named by image ID.
    Each figure shows the final rollout matrix as a token×token heatmap
    with actual token text on both axes.

Data is persisted to fold_XX/sample_rollouts.parquet so PDFs can be
regenerated without GPU via --heatmap_only.

Usage:
    # Full run (GPU):
    python scripts/mbert_attention_flow.py \\
        --alpha_version 3 --p_version p3 --fine_tuning finetuned --heatmap

    # Regenerate PDFs from saved parquet (no GPU):
    python scripts/mbert_attention_flow.py \\
        --alpha_version 3 --p_version p3 --fine_tuning finetuned --heatmap_only
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import KFold
from transformers import AutoTokenizer
from tqdm import tqdm
from mllmsent.models.heads import ModernBERTModel

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────
DATASET_TYPE = "gpt4-openai-classify"
MODEL_PATH   = "answerdotai/ModernBERT-large"
MAX_LEN      = 512
KFOLD_SEED   = 42
N_SPLITS     = 5

from mllmsent.labels import ordered_label_names

# Derived from the sentiment_values encoding the datasets were built with, so a
# plot legend can never drift from the integers in the CSVs. The previous
# hardcoded p2plus/p2neg entries were inverted against that encoding.
LABEL_NAMES = {
    problem: ordered_label_names(problem)
    for problem in ("p5", "p3", "p2plus", "p2neg")
}


# ─────────────────────────────────────────────────────────────────────────────
# Attention rollout  –  full token × token matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_full_rollout(attentions: list, seq_len: int) -> np.ndarray:
    """
    Compute the full (seq_len × seq_len) attention rollout matrix.

    R[i, j] = how much token j's identity contributed to token i after
    all transformer layers (residual-augmented, head-averaged).

    Attention tensors from ModernBERT have shape (1, heads, padded, padded);
    we slice to [:seq_len, :seq_len] to remove padding before chaining.
    """
    rollout = torch.eye(seq_len)        # identity: each token starts as itself

    for attn in attentions:
        if attn is None:
            continue
        # average over heads, strip padding
        mat = attn[0].mean(dim=0)[:seq_len, :seq_len]   # (seq_len, seq_len)

        # residual connection: A_hat = (A + I) / row_sum
        mat = mat + torch.eye(seq_len, device=mat.device)
        mat = mat / mat.sum(dim=-1, keepdim=True)

        rollout = torch.matmul(mat.cpu().float(), rollout)

    return rollout.numpy()              # (seq_len, seq_len)


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap figure  –  token × token
# ─────────────────────────────────────────────────────────────────────────────

def _clean_token(tok: str) -> str:
    return (tok.replace("Ġ", " ")
               .replace("▁", " ")
               .replace("Ċ", " ")
               .replace("##", ""))


def _build_heatmap_fig(tokens: list,
                       rollout: np.ndarray,
                       true_label: int,
                       pred_label: int,
                       label_names: list,
                       sample_id: str,
                       max_tokens: int) -> plt.Figure:
    """
    Build a token×token attention rollout heatmap figure.

    Rows  = destination tokens  (which token's representation …)
    Cols  = source tokens       (… was influenced by which token)
    Color = rollout attribution score (row-normalised for contrast)
    """
    n = min(len(tokens), max_tokens)
    mat = rollout[:n, :n]

    # Row-normalise so each destination token's distribution sums to 1 visually
    row_max = mat.max(axis=1, keepdims=True)
    mat_norm = mat / (row_max + 1e-12)

    tok_labels = [_clean_token(t) for t in tokens[:n]]

    true_name = label_names[true_label] if true_label < len(label_names) else str(true_label)
    pred_name = label_names[pred_label] if pred_label < len(label_names) else str(pred_label)
    correct   = "[OK]" if true_label == pred_label else "[X]"

    # Figure size scales with number of tokens
    fig_size = max(6, n * 0.28)
    fig, ax  = plt.subplots(figsize=(fig_size + 1.5, fig_size))

    im = ax.imshow(mat_norm, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_xticklabels(tok_labels, rotation=90, fontsize=5, fontfamily="monospace")
    ax.set_yticks(range(n))
    ax.set_yticklabels(tok_labels, fontsize=5, fontfamily="monospace")

    ax.set_xlabel("Source token  (contributes to →)", fontsize=8)
    ax.set_ylabel("Destination token  (← influenced by)", fontsize=8)
    ax.set_title(
        f"[{sample_id}]   True: {true_name}  ->  Pred: {pred_name}  {correct}\n"
        f"Full Attention Rollout Matrix (all layers chained)",
        fontsize=8,
    )

    fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02, label="Normalised attribution")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Save per-sample PDFs
# ─────────────────────────────────────────────────────────────────────────────

def save_heatmap_pdfs(sample_rows: list,
                      out_dir: str,
                      label_names: list,
                      max_tokens: int = 40) -> int:
    """
    Write one PDF per sample, named {sample_id}.pdf.
    Returns the number of files written.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    skipped = 0

    for row in sample_rows:
        rollout_data = row.get("rollout_matrix", [])
        if not rollout_data:
            skipped += 1
            continue

        rollout = np.array(rollout_data)
        if rollout.ndim != 2 or rollout.shape[0] == 0:
            skipped += 1
            continue

        sample_id = str(row["sample_id"])
        out_path  = os.path.join(out_dir, f"{sample_id}.pdf")

        fig = _build_heatmap_fig(
            row["tokens"], rollout,
            int(row["true_label"]), int(row["pred_label"]),
            label_names, sample_id, max_tokens,
        )
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        written += 1

    if skipped:
        print(f"  [WARN] {skipped}/{len(sample_rows)} samples skipped (no rollout data).")
        if written == 0:
            print("         Run without --heatmap_only to populate rollout data.")
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate mean-rollout bar chart (CLS row of mean matrix)
# ─────────────────────────────────────────────────────────────────────────────

def save_aggregate_plot(sample_rows: list,
                        output_path: str,
                        fold: int,
                        experiment_tag: str) -> None:
    """
    Average the CLS row (row 0) of each sample's rollout matrix across the fold,
    pad to MAX_LEN, and plot as a bar chart.
    """
    cls_rows = []
    for r in sample_rows:
        data = r.get("rollout_matrix", [])
        if not data:
            continue
        mat = np.array(data)
        if mat.ndim == 2 and mat.shape[0] > 0:
            padded = np.zeros(MAX_LEN)
            padded[:mat.shape[1]] = mat[0]    # CLS row
            cls_rows.append(padded)

    if not cls_rows:
        return

    mean_cls = np.stack(cls_rows).mean(axis=0)
    n = min(64, MAX_LEN)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.bar(range(n), mean_cls[:n], color="steelblue", width=0.8)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Mean CLS attribution")
    ax.set_title(f"Mean CLS Rollout Attribution – {experiment_tag} | fold {fold + 1}")
    ax.set_xlim(-0.5, n - 0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({"position": range(MAX_LEN), "mean_cls_rollout": mean_cls}
                 ).to_csv(output_path.replace(".png", ".csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Parquet persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_sample_rollouts(sample_rows: list, path: str) -> None:
    records = [
        {
            "sample_id":     r["sample_id"],
            "true_label":    r["true_label"],
            "pred_label":    r["pred_label"],
            "tokens":        json.dumps(r["tokens"]),
            "rollout_matrix": json.dumps(r.get("rollout_matrix", [])),
        }
        for r in sample_rows
    ]
    pd.DataFrame(records).to_parquet(path, index=False)


def load_sample_rollouts(path: str) -> list:
    df = pd.read_parquet(path)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "sample_id":     row["sample_id"],
            "true_label":    int(row["true_label"]),
            "pred_label":    int(row["pred_label"]),
            "tokens":        json.loads(row["tokens"]),
            "rollout_matrix": json.loads(row.get("rollout_matrix", "[]")),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Full token-to-token attention rollout for ModernBERT"
    )
    parser.add_argument("--alpha_version", type=int, required=True, choices=[3, 5])
    parser.add_argument("--p_version", required=True,
        choices=["p2", "p2plus", "p2neg", "p3", "p5"])
    parser.add_argument("--fine_tuning", default="finetuned",
        choices=["finetuned", "not_finetuned"])
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--max_tokens", type=int, default=60,
        help="Max tokens shown on each axis of the heatmap")
    parser.add_argument("--device",
        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reports_dir", default="output/reports/attention_flow")
    parser.add_argument("--heatmap", action="store_true",
        help="Generate per-sample token x token rollout heatmap PDFs")
    parser.add_argument("--heatmap_only", action="store_true",
        help="Regenerate PDFs from saved parquet (no GPU)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args        = parse_args()
    device      = torch.device(args.device)
    label_names = LABEL_NAMES.get(args.p_version, [str(i) for i in range(5)])

    experiment_tag = (
        f"gpt4-openai_modernbert_{args.p_version}_sigma{args.alpha_version}"
        f"_{args.fine_tuning}"
    )
    out_root = os.path.join(args.reports_dir, experiment_tag)
    os.makedirs(out_root, exist_ok=True)

    # ── Offline: regenerate from saved parquet ────────────────────────────────
    if args.heatmap_only:
        print("Offline mode: regenerating heatmaps from saved parquet …")
        for fold in range(N_SPLITS):
            fold_dir     = os.path.join(out_root, f"fold_{fold + 1:02d}")
            parquet_path = os.path.join(fold_dir, "sample_rollouts.parquet")
            if not os.path.exists(parquet_path):
                print(f"  [SKIP] fold {fold + 1}: {parquet_path} not found")
                continue
            rows   = load_sample_rollouts(parquet_path)
            hm_dir = os.path.join(fold_dir, "heatmaps")
            n = save_heatmap_pdfs(rows, hm_dir, label_names, args.max_tokens)
            print(f"  fold {fold + 1}: {n} PDFs -> {hm_dir}/")
        print("Done.")
        return

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    ckpt_name = (
        f"best_checkpoint_{DATASET_TYPE}_modernbert"
        f"_{args.p_version}_sigma{args.alpha_version}_{args.fine_tuning}.pt"
    )
    ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Checkpoint : {ckpt_path}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    data_path = (
        f"data/{DATASET_TYPE}/percept_dataset_alpha{args.alpha_version}"
        f"_{args.p_version}.csv"
    )
    df = pd.read_csv(data_path)
    print(f"Dataset    : {data_path}  ({len(df)} rows)")
    n_classes = df["sentiment"].nunique()

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading ModernBERT checkpoint …")
    model = ModernBERTModel(MODEL_PATH, n_classes)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    model.model.config.output_attentions = True

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # ── KFold replay ──────────────────────────────────────────────────────────
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=KFOLD_SEED)

    for fold, (_, val_idx) in enumerate(kfold.split(df)):
        print(f"\n-- Fold {fold + 1}/{N_SPLITS} --")
        fold_dir = os.path.join(out_root, f"fold_{fold + 1:02d}")
        os.makedirs(fold_dir, exist_ok=True)

        val_df      = df.iloc[val_idx].reset_index(drop=True)
        sample_rows = []

        for _, row in tqdm(val_df.iterrows(), total=len(val_df),
                           desc=f"  Fold {fold + 1}"):
            text       = str(row["text"])
            true_label = int(row["sentiment"])
            sample_id  = row.get("id", _)

            enc = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            ids     = enc["input_ids"].to(device)
            mask    = enc["attention_mask"].to(device)
            seq_len = int(mask.sum().item())

            with torch.no_grad():
                out        = model.model(ids, attention_mask=mask)
                logits     = model.classifier(out.last_hidden_state[:, 0, :])
                pred_label = int(logits.argmax(dim=-1).item())
                attentions = out.attentions

            rollout_matrix = compute_full_rollout(list(attentions), seq_len)
            tokens = tokenizer.convert_ids_to_tokens(ids[0, :seq_len].cpu().tolist())

            sample_rows.append({
                "sample_id":     sample_id,
                "true_label":    true_label,
                "pred_label":    pred_label,
                "tokens":        tokens,
                "rollout_matrix": rollout_matrix.tolist(),
            })

        # persist
        parquet_path = os.path.join(fold_dir, "sample_rollouts.parquet")
        save_sample_rollouts(sample_rows, parquet_path)
        print(f"  Parquet  -> {parquet_path}")

        # aggregate bar chart (CLS row)
        plot_path = os.path.join(fold_dir, "mean_rollout.png")
        save_aggregate_plot(sample_rows, plot_path, fold, experiment_tag)

        # per-sample heatmap PDFs
        if args.heatmap:
            hm_dir = os.path.join(fold_dir, "heatmaps")
            n = save_heatmap_pdfs(sample_rows, hm_dir, label_names, args.max_tokens)
            print(f"  Heatmaps ({n} PDFs) -> {hm_dir}/")

    print(f"\nDone. Outputs: {out_root}")


if __name__ == "__main__":
    main()
