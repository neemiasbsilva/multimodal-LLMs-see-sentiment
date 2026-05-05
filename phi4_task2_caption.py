"""
Task 2 – Image caption generation with Microsoft Phi-4-reasoning-vision-15B.

Captions are purely image-derived and independent of the sentiment labelling
scheme (p3, p5, p2plus, p2neg).  This script therefore generates captions
ONCE per alpha version and then builds all p-version training CSVs by merging
the generated text with each ground-truth file found in --gt_dir.

Intermediate output (id + text only):
    data/phi4-classify/percept_dataset_alpha{alpha}_captions.csv

Final outputs (id, text, sentiment — drop-in for ModernBERT pipeline):
    data/phi4-classify/percept_dataset_alpha{alpha}_p3.csv
    data/phi4-classify/percept_dataset_alpha{alpha}_p5.csv
    data/phi4-classify/percept_dataset_alpha{alpha}_p2plus.csv
    data/phi4-classify/percept_dataset_alpha{alpha}_p2neg.csv

Usage:
    python phi4_task2_caption.py \\
        --alpha_version 3 \\
        --gt_dir data/minigpt4-classify \\
        --image_dir /mnt/raid5/neemias/perceptdata \\
        --save_path data/phi4-classify \\
        --gpu_ids 0,1 --4bit --resume
"""

import argparse
import glob
import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


CAPTION_PROMPT = "Describe this image in details."


def find_image(image_id: str, image_dir: str) -> str | None:
    for part in [f"part{i}" for i in range(1, 7)]:
        path = os.path.join(image_dir, part, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None


def merge_and_save(caption_file: str, gt_dir: str, alpha: str, save_path: str) -> None:
    """
    For every ground-truth CSV matching alpha in gt_dir, join with the
    caption file on 'id' and write the merged (id, text, sentiment) CSV.
    """
    caps = pd.read_csv(caption_file)
    pattern = os.path.join(gt_dir, f"percept_dataset_alpha{alpha}_*.csv")
    gt_files = sorted(glob.glob(pattern))

    if not gt_files:
        print(f"  [WARN] No ground-truth files found matching {pattern}")
        return

    for gt_path in gt_files:
        pv = os.path.basename(gt_path).replace(
            f"percept_dataset_alpha{alpha}_", ""
        ).replace(".csv", "")
        gt = pd.read_csv(gt_path)[["id", "sentiment"]]
        merged = caps.merge(gt, on="id", how="inner")[["id", "text", "sentiment"]]
        out = os.path.join(save_path, f"percept_dataset_alpha{alpha}_{pv}.csv")
        merged.to_csv(out, index=False)
        print(f"  Merged {pv}: {len(merged)} rows → {out}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 2 caption generation with Phi-4-reasoning-vision-15B"
    )
    parser.add_argument("--alpha_version", required=True,
        help="Alpha version: 3 or 5")
    parser.add_argument("--gt_dir", default="data/minigpt4-classify",
        help="Directory containing ground-truth CSVs used to get image IDs and sentiments")
    parser.add_argument("--image_dir", default="/mnt/raid5/neemias/perceptdata")
    parser.add_argument("--save_path", default="data/phi4-classify")
    parser.add_argument("--model_id", default="microsoft/Phi-4-reasoning-vision-15B")
    parser.add_argument("--revision",
        default="7df902e2fec305ff57c2eddf519485a74bb2daaa",
        help="Pinned commit to avoid re-downloading and overwriting local patches")
    parser.add_argument("--gpu_ids", type=str, default="0,1")
    parser.add_argument("--max_new_tokens", type=int, default=512,
        help="Max tokens for caption generation")
    parser.add_argument("--resume", action="store_true",
        help="Skip images already present in the caption CSV")
    parser.add_argument("--4bit", dest="load_4bit", action="store_true",
        help="Load model in 4-bit quantisation (~8 GB VRAM)")
    parser.add_argument("--merge_only", action="store_true",
        help="Skip captioning; only (re-)merge an existing caption CSV with gt_dir files")
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu_ids)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    os.makedirs(args.save_path, exist_ok=True)

    # The intermediate file contains only id + text (no sentiment).
    caption_file = os.path.join(
        args.save_path,
        f"percept_dataset_alpha{args.alpha_version}_captions.csv"
    )

    # ── Collect image IDs from *all* matching ground-truth files ─────────────
    # Use p3 as the canonical reference (all p-versions share the same images).
    ref_csv = os.path.join(
        args.gt_dir,
        f"percept_dataset_alpha{args.alpha_version}_p3.csv"
    )
    if not os.path.exists(ref_csv):
        # Fallback: first available file for this alpha
        candidates = sorted(glob.glob(
            os.path.join(args.gt_dir, f"percept_dataset_alpha{args.alpha_version}_*.csv")
        ))
        if not candidates:
            raise FileNotFoundError(
                f"No ground-truth CSV found in {args.gt_dir} for alpha{args.alpha_version}"
            )
        ref_csv = candidates[0]

    df = pd.read_csv(ref_csv)[["id"]].drop_duplicates()
    print(f"Reference CSV : {ref_csv} ({len(df)} images)")

    if not args.merge_only:
        # ── Resume support ───────────────────────────────────────────────────
        already_done: set[str] = set()
        if args.resume and os.path.exists(caption_file):
            df_done = pd.read_csv(caption_file)
            already_done = set(df_done["id"].astype(str).tolist())
            print(f"Resuming – {len(already_done)} images already captioned")

        # ── Model loading ────────────────────────────────────────────────────
        print(f"Loading {args.model_id} …")
        load_kw = dict(trust_remote_code=True, revision=args.revision)
        processor = AutoProcessor.from_pretrained(args.model_id, **load_kw)

        model_kw = dict(trust_remote_code=True, revision=args.revision)
        if args.load_4bit:
            from transformers import BitsAndBytesConfig
            gpu_id = int(args.gpu_ids.split(",")[0])
            model_kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            model_kw["device_map"] = {"": gpu_id}
            print(f"  4-bit quantisation on GPU {gpu_id}")
        else:
            max_memory = {}
            for i in range(torch.cuda.device_count()):
                free_mb = torch.cuda.mem_get_info(i)[0] // (1024 ** 2)
                max_memory[i] = f"{max(0, free_mb - 1500)}MiB"
            max_memory["cpu"] = "48GiB"
            model_kw["torch_dtype"] = torch.bfloat16
            model_kw["device_map"] = "auto"
            model_kw["max_memory"] = max_memory
            print(f"  bfloat16, max_memory: {max_memory}")

        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kw)
        model.eval()

        first_param = next(p for p in model.parameters() if p.dtype != torch.uint8)
        first_device = first_param.device
        model_dtype  = first_param.dtype
        print(f"Model loaded. device={first_device}  dtype={model_dtype}")

        # ── Inference loop ───────────────────────────────────────────────────
        results = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating captions"):
            image_id = str(row["id"])

            if image_id in already_done:
                continue

            image_path = find_image(image_id, args.image_dir)
            if image_path is None:
                print(f"  [WARN] Image not found: {image_id} – skipping")
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as exc:
                print(f"  [ERROR] Cannot open {image_path}: {exc} – skipping")
                continue

            try:
                prompt_with_image = f"<image>\n{CAPTION_PROMPT}"
                messages = [{"role": "user", "content": prompt_with_image}]
                text = processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(text=text, images=[image], return_tensors="pt")
                inputs = {k: v.to(first_device) for k, v in inputs.items()}

                with torch.no_grad(), torch.autocast("cuda", dtype=model_dtype):
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )

                input_len = inputs["input_ids"].shape[1]
                caption = processor.batch_decode(
                    output_ids[:, input_len:], skip_special_tokens=True
                )[0].strip()

            except Exception as exc:
                print(f"  [ERROR] Inference failed for {image_id}: {exc} – skipping")
                torch.cuda.empty_cache()
                continue

            results.append({"id": image_id, "text": caption})
            torch.cuda.empty_cache()

            if len(results) % 50 == 0:
                _flush(results, caption_file, args.resume, already_done)
                results = []

        if results:
            _flush(results, caption_file, args.resume, already_done)

        print(f"Captions saved to {caption_file}")

    # ── Merge captions with all p-version ground-truth files ─────────────────
    print("\nMerging captions with ground-truth sentiment labels …")
    merge_and_save(caption_file, args.gt_dir, args.alpha_version, args.save_path)
    print("Done.")


def _flush(results: list, caption_file: str, resume: bool, already_done: set) -> None:
    df_chunk = pd.DataFrame(results)
    if resume and os.path.exists(caption_file):
        df_chunk.to_csv(caption_file, mode="a", header=False, index=False)
    else:
        mode   = "a" if os.path.exists(caption_file) else "w"
        header = not os.path.exists(caption_file)
        df_chunk.to_csv(caption_file, mode=mode, header=header, index=False)
    already_done.update(df_chunk["id"].tolist())


if __name__ == "__main__":
    main()
