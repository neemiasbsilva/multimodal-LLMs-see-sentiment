#!/usr/bin/env bash
# Unified pipeline for Google Gemma-4-E4B-it experiments.
#
# Runs all four stages in sequence:
#   Stage 1 – Task 1: direct image sentiment classification (Gemma 4)
#   Stage 2 – Task 2 captions: image-to-text caption generation (Gemma 4)
#   Stage 3 – Task 2a: ModernBERT with frozen backbone on Gemma 4 captions
#   Stage 4 – Task 2b: ModernBERT fully fine-tuned on Gemma 4 captions
#
# Covers all PerceptSent sigma/problem-version combinations:
#   sigma3 (alpha3): p3, p5
#   sigma5 (alpha5): p3, p5
#
# Usage:
#   bash run-gemma4.sh [--4bit]
#
# Flags:
#   --4bit   Load Gemma 4 in 4-bit quantisation (~4 GB VRAM) for Stages 1 and 2.
#            Required when GPU 1 (RTX A6000) is occupied by another process.
#
# Hardware:
#   Stage 1 & 2: GPU determined by --gpu_ids (default 0,1). With --4bit,
#                model is placed on GPU 0 only (RTX 4000 Ada, 20 GB).
#   Stage 3 & 4: train_gpu0.py always uses CUDA_VISIBLE_DEVICES=0.
#
# Results:
#   Stage 1: data/gemma4-only/alpha{3,5}p{3,5}.csv
#   Stage 2: data/gemma4-classify/percept_dataset_alpha{3,5}_{p3,p5}.csv
#   Stage 3: experiments-not-finetuning/gemma4-modernbert-experiment-{p}-alpha{a}/logs/
#   Stage 4: experiments-finetuning/gemma4-modernbert-experiment-{p}-alpha{a}/logs/
#   Checkpoints: checkpoints/

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
export HF_HOME="/mnt/raid5/neemias/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export USE_HUB_KERNELS=0          # prevent Gemma4 import from hanging on kernel download
export PYTHONUNBUFFERED=1         # show Python output in real time
mkdir -p "${HF_HOME}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data/minigpt4-classify"
IMAGE_DIR="/mnt/raid5/neemias/perceptdata"
SAVE_TASK1="${SCRIPT_DIR}/data/gemma4-only"
SAVE_TASK2="${SCRIPT_DIR}/data/gemma4-classify"
GPU_IDS="0,1"

# ── Flag parsing ──────────────────────────────────────────────────────────────
EXTRA_FLAGS=""
for arg in "$@"; do
    case "$arg" in
        --4bit) EXTRA_FLAGS="--4bit" ;;
        *) echo "[WARN] Unknown flag: $arg" ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Task 1: direct sentiment classification (Gemma 4)
# ─────────────────────────────────────────────────────────────────────────────
# echo ""
# echo "========================================================"
# echo " Stage 1: Task 1 – Direct image sentiment classification"
# echo "========================================================"

# for alpha in 3 5; do
#     for pv in p3 p5; do
#         echo ""
#         echo "  -> sigma${alpha} / ${pv}"
#         uv run python gemma4_task1_classify.py \
#             --dataset_csv "${DATA_DIR}/percept_dataset_alpha${alpha}_${pv}.csv" \
#             --image_dir "${IMAGE_DIR}" \
#             --save_path "${SAVE_TASK1}" \
#             --alpha_version "${alpha}" \
#             --p_version "${pv}" \
#             --gpu_ids "${GPU_IDS}" \
#             --resume \
#             ${EXTRA_FLAGS}
#     done
# done

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Task 2 captions: image description generation (Gemma 4)
#
# Captions are image-only and do not depend on the sentiment scheme (p3/p5/…).
# We therefore run captioning ONCE per alpha version.  After captioning, the
# script automatically merges the generated text with every p-version
# ground-truth CSV found in gt_dir and writes the final training CSVs.
# ─────────────────────────────────────────────────────────────────────────────
# echo ""
# echo "========================================================"
# echo " Stage 2: Task 2 – Caption generation (Gemma 4)"
# echo "========================================================"

# for alpha in 3 5; do
#     echo ""
#     echo "  -> alpha${alpha}  (will auto-merge all p-versions)"
#     uv run python gemma4_task2_caption.py \
#         --alpha_version "${alpha}" \
#         --gt_dir "${DATA_DIR}" \
#         --image_dir "${IMAGE_DIR}" \
#         --save_path "${SAVE_TASK2}" \
#         --gpu_ids "${GPU_IDS}" \
#         --resume \
#         ${EXTRA_FLAGS}
# done

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – Task 2a: ModernBERT frozen backbone on Gemma 4 captions
# ─────────────────────────────────────────────────────────────────────────────
# echo ""
# echo "========================================================"
# echo " Stage 3: Task 2a – ModernBERT (frozen) on Gemma 4 captions"
# echo "========================================================"

# for alpha in 3 5; do
#     for pv in p3 p5; do
#         CONFIG="${SCRIPT_DIR}/experiments-not-finetuning/gemma4-modernbert-experiment-${pv}-alpha${alpha}/config.yaml"
#         echo ""
#         echo "  -> sigma${alpha} / ${pv}  [${CONFIG}]"
#         uv run python scripts/train_gpu0.py --config "${CONFIG}"
#     done
# done

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – Task 2b: ModernBERT fully fine-tuned on Gemma 4 captions
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Stage 4: Task 2b – ModernBERT (fine-tuned) on Gemma 4 captions"
echo "========================================================"

for alpha in 3 5; do
    for pv in p3 p5; do
        CONFIG="${SCRIPT_DIR}/experiments-finetuning/gemma4-modernbert-experiment-${pv}-alpha${alpha}/config.yaml"
        echo ""
        echo "  -> sigma${alpha} / ${pv}  [${CONFIG}]"
        uv run python scripts/train_gpu0.py --config "${CONFIG}"
    done
done

echo ""
echo "========================================================"
echo " All Gemma 4 experiments complete."
echo "========================================================"
