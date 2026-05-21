#!/usr/bin/env bash
# Attention rollout heatmap generation for all configurations.
#
# Runs mbert_attention_flow.py for:
#   alpha3: p3, p5
#   alpha5: p3, p5
# Both finetuned and not_finetuned checkpoints.
#
# Usage:
#   bash run-attention-flow.sh              # full inference (GPU required)
#   bash run-attention-flow.sh --offline    # regenerate PDFs from saved parquet
#
# Outputs per fold (reports/attention_flow/<tag>/fold_XX/):
#   heatmaps.pdf               - layer x token heatmap (paper Fig. 2/3 style)
#   mean_rollout.png / .csv    - aggregate rollout bar chart
#   sample_rollouts.parquet    - saved data for offline regeneration

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export HF_HOME="/mnt/raid5/neemias/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# ── Flag parsing ──────────────────────────────────────────────────────────────
OFFLINE=false
for arg in "$@"; do
    case "$arg" in
        --offline) OFFLINE=true ;;
        *) echo "[WARN] Unknown flag: $arg" ;;
    esac
done

if $OFFLINE; then
    MODE_FLAG="--heatmap_only"
    echo "Mode: offline (regenerating PDFs from saved parquet, no GPU)"
else
    MODE_FLAG="--heatmap"
    echo "Mode: full inference (GPU)"
fi

COMMON="--max_tokens 40"
CKPT_DIR="checkpoints"
DATASET_TYPE="gpt4-openai-classify"

run_config() {
    local alpha=$1
    local pv=$2
    local ft=$3

    # Skip if checkpoint doesn't exist (e.g. alpha5 finetuned not yet trained)
    local ckpt="${CKPT_DIR}/best_checkpoint_${DATASET_TYPE}_modernbert_${pv}_sigma${alpha}_${ft}.pt"
    if [[ ! -f "${ckpt}" ]]; then
        echo ""
        echo "-- alpha${alpha} / ${pv} / ${ft} -- [SKIP: checkpoint not found: ${ckpt}]"
        return 0
    fi

    echo ""
    echo "-- alpha${alpha} / ${pv} / ${ft} --"
    uv run python scripts/mbert_attention_flow.py \
        --alpha_version "${alpha}" \
        --p_version "${pv}" \
        --fine_tuning "${ft}" \
        ${MODE_FLAG} \
        ${COMMON}
}

# alpha3
run_config 3 p3 finetuned
run_config 3 p3 not_finetuned
run_config 3 p5 finetuned
run_config 3 p5 not_finetuned

# alpha5
run_config 5 p3 finetuned
run_config 5 p3 not_finetuned
run_config 5 p5 finetuned
run_config 5 p5 not_finetuned

echo ""
echo "All done. Outputs: reports/attention_flow/"
