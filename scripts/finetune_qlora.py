#!/usr/bin/env python3
"""QLoRA fine-tuning for LLM sentiment classification.

Replaces: notebooks/fine-tuning-llm-qlora.ipynb

Loads a 4-bit quantised causal LLM, fine-tunes it with LoRA adapters via SFT,
then runs inference on the test split and reports weighted F1.

Example
-------
python scripts/finetune_qlora.py \\
    --data-file data/gpt4-openai-classify/percept_dataset_alpha5_p5.csv \\
    --model-name nvidia/Llama3-ChatQA-1.5-8B \\
    --n-classes 5 \\
    --output-dir logs/results/trained_model \\
    --log-dir logs/log \\
    --n-epochs 10
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import SFTTrainer

SENTIMENT_PROMPT = (
    "What is the sentiment of this description? "
    "Please choose an answer from "
    "{Positive/SlightlyPositive/Neutral/SlightlyNegative/Negative}.\n"
)

# Integer label → string (for reference; not used during SFT)
LABEL_STR: dict = {
    5: {4: "Positive", 3: "SlightlyPositive", 2: "Neutral", 1: "SlightlyNegative", 0: "Negative"},
    3: {2: "Positive", 0: "Neutral", 1: "Negative"},
    2: {1: "Positive", 0: "Negative"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tuning for LLM sentiment classification")
    p.add_argument("--data-file", required=True, help="Path to percept dataset CSV")
    p.add_argument(
        "--model-name",
        default="nvidia/Llama3-ChatQA-1.5-8B",
        help="HuggingFace model name or local path",
    )
    p.add_argument(
        "--output-dir",
        default="logs/results/trained_model",
        help="Where to save the fine-tuned model",
    )
    p.add_argument("--log-dir", default="logs/log", help="TrainingArguments output_dir")
    p.add_argument("--n-classes", type=int, default=5, choices=[2, 3, 5])
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=1064)
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()


# ── Model loading ──────────────────────────────────────────────────────────────


def load_model(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


# ── Data preparation ───────────────────────────────────────────────────────────


def prepare_data(
    df: pd.DataFrame, n_classes: int, test_size: float
) -> tuple:
    """Stratified split, build prompt strings, return HF Datasets + test DataFrame."""
    classes = sorted(df["sentiment"].unique())
    train_parts, test_parts = [], []
    for label in classes:
        sub = df[df["sentiment"] == label]
        if len(sub) < 2:
            continue
        tr, te = train_test_split(sub, test_size=test_size, random_state=42)
        train_parts.append(tr)
        test_parts.append(te)

    X_train = pd.concat(train_parts).copy()
    X_test = pd.concat(test_parts).copy()

    # Training text: prompt + caption + "=" + label
    X_train["text"] = X_train[["text", "sentiment"]].apply(
        lambda r: SENTIMENT_PROMPT + r["text"] + "=" + str(r["sentiment"]), axis=1
    )
    # Test text: prompt + caption + "=" (model must predict the next token)
    X_test_texts = X_test.copy()
    X_test_texts["text"] = X_test_texts["text"].apply(
        lambda t: SENTIMENT_PROMPT + t + "="
    )

    return (
        Dataset.from_pandas(X_train),
        Dataset.from_pandas(X_test_texts),
        X_test_texts,
        X_test["sentiment"].tolist(),
    )


# ── Training ───────────────────────────────────────────────────────────────────


def train(model, tokenizer, train_data: Dataset, eval_data: Dataset, args: argparse.Namespace) -> None:
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_args = TrainingArguments(
        output_dir=args.log_dir,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=25,
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        evaluation_strategy="epoch",
        gradient_checkpointing=True,
        eval_accumulation_steps=2,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        max_seq_length=args.max_seq_len,
    )
    trainer.train()
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")


# ── Inference ──────────────────────────────────────────────────────────────────


def run_inference(model, tokenizer, X_test: pd.DataFrame) -> list:
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1,
        temperature=0.01,
        do_sample=True,
    )
    preds = []
    for i in tqdm(range(len(X_test)), desc="Inference"):
        result = pipe(X_test.iloc[i]["text"])
        raw = result[0]["generated_text"].split("=")[-1].strip()
        try:
            preds.append(int(raw))
        except ValueError:
            preds.append(-1)
    return preds


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data_file)
    print(f"Loaded {len(df)} samples from {args.data_file}")
    print(f"Class distribution:\n{df['sentiment'].value_counts().to_string()}\n")

    train_data, eval_data, X_test, y_test = prepare_data(df, args.n_classes, args.test_size)
    model, tokenizer = load_model(args.model_name)
    train(model, tokenizer, train_data, eval_data, args)

    y_pred = run_inference(model, tokenizer, X_test)
    valid = [(yt, yp) for yt, yp in zip(y_test, y_pred) if yp != -1]
    if valid:
        yt_v, yp_v = zip(*valid)
        f1 = f1_score(list(yt_v), list(yp_v), average="weighted")
        print(f"\nTest Weighted F1: {f1:.4f}  ({len(valid)}/{len(y_test)} valid predictions)")
    else:
        print("No valid predictions produced.")


if __name__ == "__main__":
    main()
