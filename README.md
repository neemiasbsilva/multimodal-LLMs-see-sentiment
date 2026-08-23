# Multimodal LLMs See Sentiment

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-5.7-FFD21E?logo=huggingface&logoColor=black)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)
![uv](https://img.shields.io/badge/uv-locked-DE5FE9?logo=uv&logoColor=white)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-multimodal--LLMs--See--Sentiment-yellow)](https://huggingface.co/Neemias/multimodal-LLMs-See-Sentiment)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-multimodal--LLMs--See--Sentiment-yellow)](https://huggingface.co/datasets/Neemias/multimodal-LLMs-See-Sentiment)
[![arXiv](https://img.shields.io/badge/arXiv-2508.16873-b31b1b.svg)](https://arxiv.org/abs/2508.16873)

**MLLMsent** ([arXiv:2508.16873](https://arxiv.org/abs/2508.16873)) asks a simple question:
do multimodal LLMs understand the *sentiment* of an
image, or only its contents? It turns out that asking an MLLM to **describe** an image and
then classifying that description beats asking the MLLM for the sentiment directly — by a
wide margin, and it beats CNN/Transformer vision baselines by up to 15 points.

```
                    ┌─ Task 1 ─────────────────────────────────┐
                    │  image ──▶ MLLM ──▶ sentiment            │  direct
  image ────────────┤                                          │
                    │  image ──▶ MLLM ──▶ description ──▶ LLM ─┤  MLLMsent  ◀── best
                    └─ Task 2 ─────────────────────────────────┘
```

The best configuration is **GPT-4o mini descriptions + fine-tuned ModernBERT**, at
**95.8 % mean 5-fold F1** on PerceptSent P3/σ5.

---

## Contents

- [Quickstart](#quickstart)
- [Pretrained checkpoints](#pretrained-checkpoints)
- [Datasets](#datasets)
- [The experiment matrix](#the-experiment-matrix)
- [The CLI](#the-cli)
- [Reproducing the paper](#reproducing-the-paper)
- [Project layout](#project-layout)
- [Docker](#docker)
- [Citation](#citation)

---

## Quickstart

```bash
git clone https://github.com/neemiasbsilva/multimodal-LLMs-See-Sentiment.git
cd multimodal-LLMs-See-Sentiment

uv sync              # or: pip install -e .
cp .env.example .env # add OPENAI_API_KEY / GEMINI_API_KEY if you will run Task 1/2
```

Classify a CSV of image descriptions with the paper's best checkpoint — no training, no
dataset download:

```bash
mllmsent hub pull-checkpoint openai-modernbert-p3-sigma5
mllmsent predict -- \
    --model_name modern-bert \
    --checkpoint_path checkpoints/best_checkpoint_gpt4-openai-classify_modernbert_p3_sigma5_finetuned.pt \
    --model_path answerdotai/ModernBERT-large \
    --input_file your_descriptions.csv \
    --output_file predictions.csv \
    --num_classes 3
```

---

## Pretrained checkpoints

All 73 trained classifiers are on the Hub as fp16 safetensors:
**[Neemias/multimodal-LLMs-See-Sentiment](https://huggingface.co/Neemias/multimodal-LLMs-See-Sentiment)**

```
{caption_mllm}/{backbone}/{problem}/sigma{n}/{finetuned|not_finetuned}/
    model.safetensors
    config.json     # base model, id2label, source SHA-256, 5-fold scores
```

`mllmsent hub pull-checkpoint <experiment-id>` downloads one under the filename the rest of
the tooling expects. Every `config.json` records the 5-fold F1 that checkpoint achieved, so
the model repo and the result CSVs in the dataset repo cross-reference each other.

Not published, because the weights were never retained: the LLaMA-3 qLoRA adapters, the Swin
baseline, and a few BART σ5 cells. Their result CSVs are in the dataset repo regardless.

---

## Datasets

**[Neemias/multimodal-LLMs-See-Sentiment](https://huggingface.co/datasets/Neemias/multimodal-LLMs-See-Sentiment)**
(`--repo-type=dataset`) carries both the inputs and every result:

| folder | contents |
|---|---|
| `inputs/` | MLLM-generated descriptions paired with sentiment labels, per MLLM |
| `captions/` | raw Task-1 direct-classification outputs |
| `splits/` | the legacy fixed train/validation/test split |
| `results/` | per-fold predictions, training curves and metrics for all 141 experiments |

```python
from datasets import load_dataset

data = load_dataset(
    "Neemias/multimodal-LLMs-See-Sentiment",
    data_files="inputs/gpt4-openai-classify/percept_dataset_alpha5_p3.csv",
)
```

Two axes run through every filename:

- **σ (sigma / alpha)** ∈ {3, 4, 5} — how many annotators had to agree for a sample to
  survive. Higher σ means a smaller, cleaner set.
- **P (problem)** ∈ {`p5`, `p3`, `p2plus`, `p2neg`} — label granularity. `p2plus` folds
  Neutral into Positive; `p2neg` folds it into Negative.

The PerceptSent source images are not redistributed. Point `IMAGE_DIR` at a directory
holding `part1/`…`part6/` to run Task 1 or Task 2 yourself.

---

## The experiment matrix

Every experiment is one row of a cartesian product declared in
[`experiments.yaml`](experiments.yaml) — 141 in total, replacing what used to be 141
near-identical `config.yaml` files whose *directory names* encoded the experiment's identity.

```yaml
runs:
  - id: core-finetuned
    track: finetuning
    mllm: [openai, deepseek, minigpt4]
    backbone: [modernbert, bart, llama3-qlora]
    problem: [p3, p5]
    sigma: [3, 5]
```

Inspect it without running anything:

```bash
mllmsent matrix list --track finetuning --backbone modernbert
mllmsent matrix show openai-modernbert-p3-sigma5
mllmsent matrix check          # every log dir, dataset and checkpoint accounted for
```

---

## The CLI

```
mllmsent matrix    list | resolve | show | check
mllmsent classify  --mllm {openai|gemini|deepseek|phi4|gemma4} --sigma N --p P   # Task 1
mllmsent caption   --mllm ... --sigma N                                          # Task 2
mllmsent train     [ids...] [--track ...] [--gpu 0] [--dry-run]
mllmsent sweep     [--track ...] [--skip-existing] [--continue-on-error]
mllmsent predict   -- --checkpoint_path ... --input_file ... --output_file ...
mllmsent evaluate  kfold | stats | posthoc | paired-ttest | vader | zero-shot
mllmsent attention-flow
mllmsent hub       status | convert | push-models | push-datasets | pull-checkpoint
```

`--gpu` is applied before torch is imported, so one process can target any device:

```bash
mllmsent --gpu 1 sweep --track not-finetuning --backbone modernbert
```

`--dry-run` prints every resolved path and hyperparameter for the selected experiments
without training, which is the fastest way to check a change did what you meant.

---

## Reproducing the paper

```bash
# 1. Descriptions from the MLLM (Task 2)
mllmsent caption --mllm openai --sigma 5 --resume

# 2. Train the text classifier on those descriptions
mllmsent train openai-modernbert-p3-sigma5 --gpu 0

# 3. Metrics and significance tests
mllmsent evaluate kfold -- --model openai --tasks 1 2a 2b
mllmsent evaluate stats
```

For the direct baseline (Task 1), swap step 1 for
`mllmsent classify --mllm openai --sigma 5 --p p3`.

Everything is written under [`output/`](output/), which is gitignored except for its
`.gitkeep` — the result CSVs live on the Hub instead of in git history.

---

## Project layout

```
multimodal-LLMs-See-Sentiment/
├── experiments.yaml          # the 141-experiment matrix — the single source of truth
├── src/mllmsent/
│   ├── cli.py                # the mllmsent entry point
│   ├── labels.py             # the one sentiment encoding everything derives from
│   ├── experiments/          # ExperimentSpec + matrix expansion
│   ├── inference/backends/   # one module per MLLM, on a shared common.py
│   ├── training/             # one trainer for every backbone and track
│   ├── evaluation/           # k-fold metrics, Holm-corrected t-tests, baselines
│   ├── analysis/             # attention rollout
│   └── hub/                  # convert, push and pull Hub artifacts
├── scripts/                  # env.sh + sweep.sh, and nothing else
├── data/                     # gitignored; mirrored on the Hub
├── checkpoints/              # gitignored; mirrored on the Hub
├── output/                   # gitignored; experiment results and staging
├── third_party/minigpt4/     # upstream MiniGPT-4, kept out of the package
└── notebooks/                # analysis and visualisation only
```

---

## Docker

An image with the environment — and no weights, keys or data — is published to GHCR on every
release:

```bash
docker pull ghcr.io/neemiasbsilva/multimodal-llms-see-sentiment:latest
docker run --rm -it --gpus all --env-file .env \
    ghcr.io/neemiasbsilva/multimodal-llms-see-sentiment:latest
```

---

## Citation

```bibtex
@misc{dasilva2026multimodalllmssentiment,
      title={Multimodal LLMs See Sentiment},
      author={Neemias B. da Silva and John Harrison and Rodrigo Minetto and Myriam R. Delgado and Bogdan T. Nassu and Thiago H. Silva},
      year={2026},
      eprint={2508.16873},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.16873},
}
```

---

## License

Code released under the terms in [LICENSE](LICENSE). The Hub artifacts are CC-BY-4.0; the
base models keep their own licenses.
