"""
Task 1: Direct sentiment classification from images using Google Gemma-4-E4B-it.

This script loads the PerceptSent dataset images and uses Gemma-4-E4B-it
to directly classify the sentiment of each image. Results are saved as CSV files
following the same format used by Phi-4 (id, sentiment columns).

Usage:
    python gemma4_task1_classify.py \
        --dataset_csv data/minigpt4-classify/percept_dataset_alpha3_p3.csv \
        --image_dir /mnt/raid5/neemias/perceptdata \
        --save_path data/gemma4-only \
        --alpha_version 3 \
        --p_version p3 \
        --gpu_ids 0,1

Output format: CSV with columns [id, sentiment] where sentiment is the raw
string response from the model (e.g. "Positive", "Neutral", "Negative").
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

HF_CACHE_DIR = "/mnt/raid5/neemias/.cache/huggingface"


SENTIMENT_LABELS = {
    "p5": ["Positive", "Slightly positive", "Neutral", "Slightly negative", "Negative"],
    "p3": ["Positive", "Neutral", "Negative"],
    "p2": ["Positive", "Negative"],
    "p2plus": ["Positive", "Negative"],
    "p2neg": ["Positive", "Negative"],
}


def build_prompt(p_version: str) -> str:
    labels = SENTIMENT_LABELS.get(p_version, SENTIMENT_LABELS["p3"])
    label_str = ", ".join(labels)
    return (
        f"Analyze this image, and classify it as {label_str} sentiments, "
        "do not describe the image, and select only one class."
    )


def find_image(image_id: str, image_dir: str) -> str | None:
    """Search for image in part1..part6 subdirectories."""
    for part in [f"part{i}" for i in range(1, 7)]:
        path = os.path.join(image_dir, part, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None


def extract_sentiment(response: str, p_version: str) -> str:
    """Return the first matching sentiment label found in the model response."""
    labels = SENTIMENT_LABELS.get(p_version, SENTIMENT_LABELS["p3"])
    response_lower = response.lower()
    for label in sorted(labels, key=len, reverse=True):
        if label.lower() in response_lower:
            return label
    return response.strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 1 – direct sentiment classification with Google Gemma-4-E4B-it"
    )
    parser.add_argument(
        "--dataset_csv",
        required=True,
        help="Path to CSV with columns [id, text, sentiment]",
    )
    parser.add_argument(
        "--image_dir",
        default="/mnt/raid5/neemias/perceptdata",
        help="Root directory containing part1..part6 subdirectories with images",
    )
    parser.add_argument(
        "--save_path",
        default="data/gemma4-only",
        help="Directory where output CSV files will be written",
    )
    parser.add_argument(
        "--alpha_version",
        required=True,
        help="Alpha (sigma) version: 3 or 5",
    )
    parser.add_argument(
        "--p_version",
        required=True,
        choices=["p2", "p2plus", "p2neg", "p3", "p5"],
        help="Problem version: p2, p2plus, p2neg, p3 or p5",
    )
    parser.add_argument(
        "--model_id",
        default="google/gemma-4-E4B-it",
        help="HuggingFace model ID for Gemma 4 vision model",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0,1",
        help="Comma-separated GPU indices to use, e.g. '0,1'",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate for each classification response",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already present in the output CSV",
    )
    parser.add_argument(
        "--4bit",
        dest="load_4bit",
        action="store_true",
        help="Load model in 4-bit via bitsandbytes (~4 GB VRAM)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu_ids)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    os.makedirs(args.save_path, exist_ok=True)
    output_file = os.path.join(
        args.save_path, f"alpha{args.alpha_version}{args.p_version}.csv"
    )

    df = pd.read_csv(args.dataset_csv)
    print(f"Loaded {len(df)} samples from {args.dataset_csv}")

    already_done = set()
    if args.resume and os.path.exists(output_file):
        df_done = pd.read_csv(output_file)
        already_done = set(df_done["id"].tolist())
        print(f"Resuming – {len(already_done)} samples already processed")

    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    print(f"Loading {args.model_id} …")
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=HF_CACHE_DIR)

    model_kwargs = {"cache_dir": HF_CACHE_DIR}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        gpu_id = int(args.gpu_ids.split(",")[0])
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        model_kwargs["device_map"] = {"": gpu_id}
        print(f"  Using 4-bit quantization on GPU {gpu_id}")
    else:
        max_memory = {}
        for i in range(torch.cuda.device_count()):
            free_mb = torch.cuda.mem_get_info(i)[0] // (1024 ** 2)
            max_memory[i] = f"{max(0, free_mb - 1500)}MiB"
        max_memory["cpu"] = "48GiB"
        print(f"  bfloat16 max_memory: {max_memory}")
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"
        model_kwargs["max_memory"] = max_memory

    model = Gemma4ForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    model.eval()

    first_param = next(p for p in model.parameters() if p.dtype not in (torch.uint8, torch.int8))
    first_device = first_param.device
    model_dtype = first_param.dtype
    print(f"Model loaded. Device: {first_device}, dtype: {model_dtype}")

    prompt_text = build_prompt(args.p_version)
    print(f"Prompt: {prompt_text}")

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying images"):
        image_id = str(row["id"])

        if image_id in already_done:
            continue

        image_path = find_image(image_id, args.image_dir)
        if image_path is None:
            print(f"  [WARNING] Image not found: {image_id} – skipping")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  [ERROR] Could not open {image_path}: {e} – skipping")
            continue

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            # Step 1: render the chat template (text only, tokenize=False)
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            # Step 2: tokenize text + encode image together
            inputs = processor(text=text, images=[image], return_tensors="pt").to(first_device)
            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad(), torch.autocast("cuda", dtype=model_dtype):
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            generated_ids = output_ids[:, input_len:]
            response = processor.decode(generated_ids[0], skip_special_tokens=True).strip()

        except Exception as e:
            print(f"  [ERROR] Inference failed for {image_id}: {e} – skipping")
            torch.cuda.empty_cache()
            continue

        torch.cuda.empty_cache()

        sentiment = extract_sentiment(response, args.p_version)
        results.append({"id": image_id, "sentiment": sentiment, "raw_response": response})

        if len(results) % 50 == 0:
            df_partial = pd.DataFrame(results)
            if args.resume and os.path.exists(output_file):
                df_partial.to_csv(output_file, mode="a", header=False, index=False)
            else:
                df_partial.to_csv(output_file, index=False)
            results = []

    if results:
        df_partial = pd.DataFrame(results)
        if args.resume and os.path.exists(output_file):
            df_partial.to_csv(output_file, mode="a", header=False, index=False)
        else:
            if os.path.exists(output_file):
                df_partial.to_csv(output_file, mode="a", header=False, index=False)
            else:
                df_partial.to_csv(output_file, index=False)

    print(f"Done. Results saved to {output_file}")


if __name__ == "__main__":
    main()
