"""Model and dataset cards for the two Hub repositories."""

from __future__ import annotations

import json
from pathlib import Path

GITHUB_URL = "https://github.com/neemiasbsilva/multimodal-LLMs-see-sentiment"
PAPER_URL = "https://arxiv.org/abs/2508.16873"
ARXIV_ID = "2508.16873"

CITATION = """```bibtex
@misc{dasilva2026multimodalllmssentiment,
      title={Multimodal LLMs See Sentiment},
      author={Neemias B. da Silva and John Harrison and Rodrigo Minetto and Myriam R. Delgado and Bogdan T. Nassu and Thiago H. Silva},
      year={2026},
      eprint={2508.16873},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.16873},
}
```"""
MODEL_URL = "https://huggingface.co/Neemias/multimodal-LLMs-See-Sentiment"
DATASET_URL = "https://huggingface.co/datasets/Neemias/multimodal-LLMs-See-Sentiment"

MODEL_FRONT_MATTER = """---
license: cc-by-4.0
arxiv: 2508.16873
library_name: pytorch
pipeline_tag: text-classification
tags:
  - sentiment-analysis
  - multimodal
  - image-sentiment
  - perceptsent
  - mllm
language:
  - en
base_model:
  - answerdotai/ModernBERT-large
  - facebook/bart-large-mnli
---
"""

DATASET_FRONT_MATTER = """---
license: cc-by-4.0
arxiv: 2508.16873
task_categories:
  - text-classification
  - image-classification
language:
  - en
tags:
  - sentiment-analysis
  - multimodal
  - image-captioning
  - perceptsent
size_categories:
  - 10K<n<100K
---
"""


def _table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def model_card(manifest: dict) -> str:
    by_group: dict[tuple[str, str], list[dict]] = {}
    for entry in manifest.values():
        spec_id = entry["spec_id"]
        mllm, backbone = spec_id.split("-")[0], entry["base_model"]
        by_group.setdefault((mllm, backbone), []).append(entry)

    best = sorted(
        (entry for entry in manifest.values() if entry.get("mean_f1")),
        key=lambda entry: entry["mean_f1"],
        reverse=True,
    )[:10]

    leaderboard = _table(
        [
            (
                entry["spec_id"],
                entry["track"],
                f"{entry['mean_f1']:.4f}",
                str(entry["num_classes"]),
            )
            for entry in best
        ],
        ("checkpoint", "track", "mean 5-fold F1", "classes"),
    )

    coverage = _table(
        sorted(
            (
                (f"`{mllm}`", f"`{backbone}`", str(len(entries)))
                for (mllm, backbone), entries in by_group.items()
            )
        ),
        ("caption MLLM", "base model", "checkpoints"),
    )

    return f"""{MODEL_FRONT_MATTER}
# MLLMsent — sentiment classifiers over multimodal-LLM image descriptions

Fine-tuned text classifiers from **"Multimodal LLMs See Sentiment"**
([arXiv:{ARXIV_ID}]({PAPER_URL})). Each checkpoint scores the sentiment of an image
*description* produced by a multimodal LLM, which is the second stage of the MLLMsent
pipeline.

- **Paper:** [arXiv:{ARXIV_ID}]({PAPER_URL})
- **Code, training and inference:** {GITHUB_URL}
- **Datasets (inputs, captions and every result CSV):** {DATASET_URL}

## Pipeline

```
image ──▶ multimodal LLM ──▶ description ──▶ text classifier ──▶ sentiment
          (GPT-4o mini, Gemini,             (these checkpoints:
           DeepSeek-VL2, Phi-4,              ModernBERT-large,
           Gemma-4, MiniGPT-4)               BART-large-MNLI)
```

The paper's best configuration is **GPT-4o mini captions + fine-tuned ModernBERT**.

## Layout

```
{{caption_mllm}}/{{backbone}}/{{problem}}/sigma{{n}}/{{finetuned|not_finetuned}}/
    model.safetensors
    config.json
MANIFEST.json
```

- **problem** — label granularity: `p5` (5 classes), `p3` (3), `p2plus`/`p2neg` (2).
- **sigma** — annotator-agreement threshold used to filter the training set (3 or 5).
- **finetuned** — whole backbone trained. **not_finetuned** — backbone frozen, head only.

Every `config.json` carries the base model id, `id2label`/`label2id`, the source
checkpoint's SHA-256 and the 5-fold scores that checkpoint achieved.

## Coverage

{coverage}

Weights are **fp16 safetensors** converted from the original fp32 training checkpoints.

## Best checkpoints

{leaderboard}

## Usage

```python
import json, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

repo = "Neemias/multimodal-LLMs-See-Sentiment"
folder = "gpt4-openai-classify/modernbert/p3/sigma5/finetuned"

weights = load_file(hf_hub_download(repo, f"{{folder}}/model.safetensors"))
config = json.load(open(hf_hub_download(repo, f"{{folder}}/config.json")))

for alias, owner in config["tied_weights"].items():
    weights[alias] = weights[owner]


class SentimentClassifier(torch.nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.model = AutoModel.from_pretrained(base_model)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, num_classes),
        )

    def forward(self, ids, mask):
        return self.classifier(self.model(ids, attention_mask=mask).last_hidden_state[:, 0])


model = SentimentClassifier(config["base_model"], config["num_classes"])
model.load_state_dict({{k: v.float() for k, v in weights.items()}})
model.eval()

tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
batch = tokenizer(["A bright park full of children playing."], return_tensors="pt",
                  padding="max_length", truncation=True, max_length=config["max_len"])
prediction = model(batch["input_ids"], batch["attention_mask"]).argmax(-1).item()
print(config["id2label"][str(prediction)])
```

Or through the project CLI:

```bash
mllmsent hub pull-checkpoint openai-modernbert-p3-sigma5
mllmsent predict --spec openai-modernbert-p3-sigma5 --input captions.csv --output predictions.csv
```

## Not published here

- **LLaMA-3 qLoRA adapters** — the adapter weights were never retained; only the
  training logs and `adapter_config.json` survive.
- **Swin Transformer baseline** — its checkpoint-saving path was broken, so no
  weights were ever written. Results for it are in the dataset repo.
- A few BART sigma-5 fine-tuned cells, for the same reason.

## Citation

{CITATION}

## License

CC-BY-4.0. The base models keep their own licenses.
"""


def dataset_card(counts: dict) -> str:
    inventory = _table(
        [
            ("`inputs/`", "captions paired with sentiment labels, per MLLM", str(counts.get("inputs", 0))),
            ("`captions/`", "raw Task-1 direct-classification outputs", str(counts.get("captions", 0))),
            ("`splits/`", "legacy fixed train/validation/test split", str(counts.get("splits", 0))),
            ("`results/`", "per-fold predictions, training curves and metrics", str(counts.get("results", 0))),
        ],
        ("folder", "contents", "files"),
    )

    return f"""{DATASET_FRONT_MATTER}
# MLLMsent — datasets and experiment results

Every input and every output of **"Multimodal LLMs See Sentiment"**
([arXiv:{ARXIV_ID}]({PAPER_URL})): the image descriptions generated by six multimodal
LLMs, the sentiment labels derived from the PerceptSent annotations, and the complete
per-fold results of all 141 experiments.

- **Paper:** [arXiv:{ARXIV_ID}]({PAPER_URL})
- **Code, training and inference:** {GITHUB_URL}
- **Model checkpoints:** {MODEL_URL}

## Contents

{inventory}

## Naming

Files follow `percept_dataset_alpha{{sigma}}_{{problem}}.csv`:

- **sigma** ∈ {{3, 4, 5}} — how many annotators had to agree for a sample to be kept.
  Higher sigma means a smaller, cleaner set.
- **problem** ∈ {{`p5`, `p3`, `p2plus`, `p2neg`}} — label granularity. `p2plus` folds
  Neutral into Positive; `p2neg` folds it into Negative.

Columns are `id, text, sentiment`, where `text` is the MLLM-generated description and
`sentiment` is an integer whose meaning depends on the problem:

| problem | encoding |
|---|---|
| `p5` | 0 Negative, 1 SlightlyNegative, 2 Neutral, 3 SlightlyPositive, 4 Positive |
| `p3` | 0 Neutral, 1 Negative, 2 Positive |
| `p2plus` | 0 Positive (incl. Neutral), 1 Negative |
| `p2neg` | 0 Negative (incl. Neutral), 1 Positive |

## Caption sources

`inputs/` holds one folder per multimodal LLM that produced the descriptions:
`gpt4-openai-classify` (GPT-4o mini), `gemini-classify` (Gemini 2.5 Flash),
`deepseek` (DeepSeek-VL2), `phi4-classify` (Phi-4 vision), `gemma4-classify`
(Gemma-4), `minigpt4-classify` (MiniGPT-4). `percept_dataset` holds the labels alone.

## Results

`results/` mirrors the experiment trees. Each experiment directory contains:

- `test_logs.csv` — one row per fold: `kfold, accuracy, f1_score, time`
- `test_logs_0N.csv` — per-fold predictions: `id, text, target, prediction`
- `training_logs_0N.csv` — per-epoch training curves

## Source images

The PerceptSent images themselves are not redistributed here. See {GITHUB_URL} for
how to obtain them.

## Usage

```python
from datasets import load_dataset

data = load_dataset(
    "Neemias/multimodal-LLMs-See-Sentiment",
    data_files="inputs/gpt4-openai-classify/percept_dataset_alpha5_p3.csv",
)
```

## Citation

{CITATION}

## License

CC-BY-4.0.
"""


def write_model_card(staging: Path) -> Path:
    manifest_path = staging / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    path = staging / "README.md"
    path.write_text(model_card(manifest))
    return path


def write_dataset_card(staging: Path, counts: dict) -> Path:
    path = staging / "README.md"
    path.write_text(dataset_card(counts))
    return path
