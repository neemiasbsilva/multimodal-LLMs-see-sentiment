#!/usr/bin/env bash
# Run Task 1 (direct sentiment classification) for Google Gemma-4-E4B-it
# across all PerceptSent sigma/problem-version combinations used in the paper.
#
# Hardware: GPU 0 = RTX 4000 Ada (20 GB), GPU 1 = RTX A6000 (48 GB)
# GPU 1 is typically occupied by another process, so we use --4bit to fit in GPU 0 (~4 GB).
#
# Configurations:
#   - sigma3 (alpha3): p3, p5
#   - sigma5 (alpha5): p3, p5
#
# Results are saved to data/gemma4-only/

set -e

export HF_HOME="/mnt/raid5/neemias/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${HF_HOME}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data/minigpt4-classify"
IMAGE_DIR="/mnt/raid5/neemias/perceptdata"
SAVE_PATH="${SCRIPT_DIR}/data/gemma4-only"
GPU_IDS="0,1"

# sigma3 – P3
uv run python gemma4_task1_classify.py \
    --dataset_csv "${DATA_DIR}/percept_dataset_alpha3_p3.csv" \
    --image_dir "${IMAGE_DIR}" \
    --save_path "${SAVE_PATH}" \
    --alpha_version 3 \
    --p_version p3 \
    --gpu_ids ${GPU_IDS} \
    --4bit \
    --resume

# sigma3 – P5
uv run python gemma4_task1_classify.py \
    --dataset_csv "${DATA_DIR}/percept_dataset_alpha3_p5.csv" \
    --image_dir "${IMAGE_DIR}" \
    --save_path "${SAVE_PATH}" \
    --alpha_version 3 \
    --p_version p5 \
    --gpu_ids ${GPU_IDS} \
    --4bit \
    --resume

# sigma5 – P3
uv run python gemma4_task1_classify.py \
    --dataset_csv "${DATA_DIR}/percept_dataset_alpha5_p3.csv" \
    --image_dir "${IMAGE_DIR}" \
    --save_path "${SAVE_PATH}" \
    --alpha_version 5 \
    --p_version p3 \
    --gpu_ids ${GPU_IDS} \
    --4bit \
    --resume

# sigma5 – P5
uv run python gemma4_task1_classify.py \
    --dataset_csv "${DATA_DIR}/percept_dataset_alpha5_p5.csv" \
    --image_dir "${IMAGE_DIR}" \
    --save_path "${SAVE_PATH}" \
    --alpha_version 5 \
    --p_version p5 \
    --gpu_ids ${GPU_IDS} \
    --4bit \
    --resume
