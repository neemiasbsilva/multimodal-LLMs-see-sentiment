"""LLaMA-3 qLoRA track.

Fine-tuned runs train a LoRA adapter and then score the held-out fold;
not-finetuned runs skip training and prompt the base model few-shot. The label
map in the prompt is derived from mllmsent.labels, which corrects the hardcoded
maps the previous implementation used (p3 claimed Negative=0/Neutral=1 against a
ground truth of Neutral=0/Negative=1, and the two-class prompts named a Neutral
class that no longer exists after folding).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from tqdm import tqdm

from mllmsent.data.experiment_data import load_experiment_frame, load_twitter_validation
from mllmsent.experiments.spec import ExperimentSpec
from mllmsent.labels import label2id

KFOLD_SEED = 42
FEW_SHOT_EXAMPLES = 15

LORA = {"lora_alpha": 8, "lora_dropout": 0.1, "r": 32, "bias": "none"}


def build_prompt(problem: str) -> str:
    mapping = label2id(problem)
    rendered = ", ".join(f'"{name}": {index}' for name, index in mapping.items())
    return (
        "What is the sentiment of this description? "
        f"Please choose an answer from {{{rendered}}}\n"
    )


def build_training_arguments(spec: ExperimentSpec, output_dir: Path):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=spec.hyperparameters.epochs,
        per_device_train_batch_size=spec.hyperparameters.batch_size,
        per_device_eval_batch_size=1,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        eval_strategy="epoch",
        gradient_checkpointing=True,
        eval_accumulation_steps=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


def train_adapter(spec, model, tokenizer, train_data, eval_data, output_dir: Path):
    from peft import LoraConfig
    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=LoraConfig(task_type="CAUSAL_LM", **LORA),
        args=build_training_arguments(spec, output_dir),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return model, tokenizer


def predict(frame: pd.DataFrame, model, tokenizer) -> list:
    from transformers import pipeline

    generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1,
        temperature=0.01,
        do_sample=True,
    )
    predictions = []
    for index in tqdm(range(len(frame))):
        result = generator(frame.iloc[index]["text"])
        answer = result[0]["generated_text"].split("=")[-1].strip().strip(".,!?;'\"")
        predictions.append(int(answer) if answer.isdigit() else answer.lower())
    return predictions


def coerce_predictions(raw, truth, num_classes: int):
    """Non-numeric generations count as a miss rather than being dropped."""
    predictions, targets = [], []
    for value, target in zip(raw, truth):
        try:
            predictions.append(int(value))
        except (ValueError, TypeError):
            predictions.append((target + 1) % num_classes)
        targets.append(target)
    return predictions, targets


def train_qlora(spec: ExperimentSpec, matrix, folds: int = 5, max_epochs=None) -> dict:
    from datasets import Dataset

    from mllmsent.models.heads import Llama3

    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

    log_dir = matrix.paths.results_root / spec.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    frame = load_experiment_frame(spec, matrix.paths.data_root)
    twitter_val = (
        load_twitter_validation(spec, matrix.paths.twitter_captions_root)
        if spec.uses_twitter_validation
        else None
    )

    prompt = build_prompt(spec.problem)
    adapter_root = spec.checkpoint_dir(matrix.paths.checkpoint_root) / "llama"
    kfold = KFold(n_splits=folds, shuffle=True, random_state=KFOLD_SEED)
    metrics = pd.DataFrame([])

    for fold, (train_idx, val_idx) in enumerate(kfold.split(frame)):
        print(f"fold {fold + 1}/{folds}")
        started = time.time()

        llama = Llama3(spec.backbone_profile.model_path)
        model, tokenizer = llama.get_model()

        train_df = frame.iloc[train_idx].copy()
        val_df = (twitter_val.copy() if twitter_val is not None else frame.iloc[val_idx].copy())

        train_df["input"] = train_df["text"]
        val_df["input"] = val_df["text"]
        train_df["text"] = train_df.apply(
            lambda row: f"{prompt}{row['input']}={row['sentiment']}", axis=1
        )

        if spec.trains_qlora_adapter:
            val_df["text"] = val_df.apply(lambda row: f"{prompt}{row['input']}=", axis=1)
        else:
            header = "\n\n".join(train_df["text"].head(FEW_SHOT_EXAMPLES).tolist())
            val_df["text"] = val_df.apply(
                lambda row: f"{header}\n\n{prompt}{row['input']}=", axis=1
            )

        if spec.trains_qlora_adapter:
            model, tokenizer = train_adapter(
                spec,
                model,
                tokenizer,
                Dataset.from_pandas(train_df),
                Dataset.from_pandas(val_df),
                adapter_root / spec.qlora_adapter_name,
            )

        raw = predict(val_df, model, tokenizer)
        predictions, targets = coerce_predictions(
            raw, val_df["sentiment"].tolist(), spec.num_classes
        )

        metrics = pd.concat(
            [
                metrics,
                pd.DataFrame(
                    {
                        "kfold": [fold + 1],
                        "accuracy": [accuracy_score(targets, predictions)],
                        "f1_score": [f1_score(targets, predictions, average="weighted")],
                        "time": [int(time.time() - started)],
                    }
                ),
            ],
            axis=0,
        )
        metrics.to_csv(log_dir / "test_logs.csv", index=False)

        if twitter_val is None:
            results = pd.DataFrame(
                {
                    "id": frame["id"].iloc[val_idx].to_list(),
                    "text": frame["input"].iloc[val_idx].to_list()
                    if "input" in frame
                    else val_df["input"].to_list(),
                    "target": targets,
                    "prediction": predictions,
                }
            )
        else:
            results = pd.DataFrame(
                {
                    "text": val_df["input"].to_list(),
                    "target": targets,
                    "prediction": predictions,
                }
            )
        results.to_csv(log_dir / f"test_logs_{fold + 1:02d}.csv", index=False)

    scores = metrics["f1_score"].to_numpy()
    mean_f1 = float(np.mean(scores))
    interval = (
        stats.t.interval(0.95, len(scores) - 1, loc=mean_f1, scale=stats.sem(scores))
        if len(scores) > 1
        else (mean_f1, mean_f1)
    )
    print(f"mean F1 {mean_f1 * 100:.2f}%  95% CI {interval}")
    return {"mean_f1": mean_f1, "confidence_interval": interval, "folds": len(scores)}
