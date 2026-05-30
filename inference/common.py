"""Shared helpers for the MLLM inference scripts (Task 1 and Task 2).

Reused by the OpenAI, Gemini and DeepSeek scripts to keep prompts, image lookup,
label extraction and incremental CSV writing identical across model families.

Task 1 is direct image-to-sentiment classification; Task 2 is image-to-description
for the description-mediated pipeline (the description is later classified by a
text-only LLM such as the fine-tuned ModernBERT in ``scripts/inference.py``).
"""

from __future__ import annotations

import base64
import os
from typing import Callable

import pandas as pd
from tqdm import tqdm

# Sentiment label sets per problem version (P_C in the paper).
SENTIMENT_LABELS = {
    "p5": ["Positive", "Slightly positive", "Neutral", "Slightly negative", "Negative"],
    "p3": ["Positive", "Neutral", "Negative"],
    "p2": ["Positive", "Negative"],
    "p2plus": ["Positive", "Negative"],
    "p2neg": ["Positive", "Negative"],
}

# Kept identical across all MLLMs so Task 2 descriptions stay comparable.
DESCRIPTION_PROMPT = "Describe this image in details."

_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def build_classification_prompt(p_version: str) -> str:
    """Returns the Task 1 prompt listing the labels for ``p_version``."""
    labels = SENTIMENT_LABELS.get(p_version, SENTIMENT_LABELS["p3"])
    label_str = ", ".join(labels)
    return (
        f"Analyze this image, and classify it as {label_str} sentiments, "
        "do not describe the image, and select only one class."
    )


def find_image(image_id: str, image_dir: str) -> str | None:
    """Returns the path to ``{image_id}.jpg`` under part1..part6, or None."""
    for part in [f"part{i}" for i in range(1, 7)]:
        path = os.path.join(image_dir, part, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None


def extract_sentiment(response: str, p_version: str) -> str:
    """Maps a raw model response to a sentiment label.

    Longer labels are matched first so "slightly positive" wins over the
    substring "positive". Falls back to the stripped raw text when no label
    matches, keeping it available for manual inspection.
    """
    labels = SENTIMENT_LABELS.get(p_version, SENTIMENT_LABELS["p3"])
    response_lower = response.lower()
    for label in sorted(labels, key=len, reverse=True):
        if label.lower() in response_lower:
            return label
    return response.strip()


def image_mime(image_path: str) -> str:
    """Returns the MIME type inferred from the file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    return _MIME_BY_EXT.get(ext, "image/jpeg")


def read_image_bytes(image_path: str) -> tuple[bytes, str]:
    """Returns ``(raw_bytes, mime_type)`` for an image file."""
    with open(image_path, "rb") as fh:
        return fh.read(), image_mime(image_path)


def encode_image_base64(image_path: str) -> tuple[str, str]:
    """Returns ``(base64_string, mime_type)`` for an image file."""
    raw, mime = read_image_bytes(image_path)
    return base64.b64encode(raw).decode("utf-8"), mime


def build_output_path(save_path: str, alpha_version, p_version, kind: str) -> str:
    """Resolves the output CSV path following the repo's data layout.

    ``kind="classify"`` yields ``alpha{a}_{p}.csv`` (direct labels);
    ``kind="caption"`` yields ``descriptions.csv`` (descriptions are p-independent).
    """
    os.makedirs(save_path, exist_ok=True)
    if kind == "caption":
        return os.path.join(save_path, "descriptions.csv")
    name = f"alpha{alpha_version}_{p_version}.csv" if alpha_version else f"{p_version}.csv"
    return os.path.join(save_path, name)


def _load_done_ids(output_file: str) -> set[str]:
    if os.path.exists(output_file):
        try:
            return set(pd.read_csv(output_file)["id"].astype(str).tolist())
        except Exception:
            return set()
    return set()


def _append_csv(records: list[dict], output_file: str) -> None:
    """Appends records, writing the header only for a new file."""
    frame = pd.DataFrame(records)
    frame.to_csv(output_file, mode="a", header=not os.path.exists(output_file), index=False)


def run_over_dataset(
    df: pd.DataFrame,
    image_dir: str,
    output_file: str,
    infer_fn: Callable[[str], str],
    build_record: Callable[[str, str], dict],
    *,
    resume: bool = False,
    limit: int | None = None,
    flush_every: int = 20,
    desc: str = "Processing images",
) -> None:
    """Runs ``infer_fn`` over each image in ``df`` and writes a CSV incrementally.

    Args:
        df: DataFrame with an ``id`` column of image identifiers.
        image_dir: root directory holding part1..part6 image subdirectories.
        output_file: destination CSV path.
        infer_fn: maps an image path to the raw model output (the model call).
        build_record: maps ``(image_id, raw_output)`` to a CSV row dict.
        resume: skip ids already in ``output_file`` and append to it.
        limit: process at most this many not-yet-done images (smoke testing).
        flush_every: write to disk every N records to survive interruptions.
        desc: progress-bar label.
    """
    # A non-resume run starts clean so records never append onto stale data.
    if not resume and os.path.exists(output_file):
        os.remove(output_file)

    already_done = _load_done_ids(output_file) if resume else set()
    if already_done:
        print(f"Resuming — {len(already_done)} images already processed")

    results: list[dict] = []
    processed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        image_id = str(row["id"])
        if image_id in already_done:
            continue
        if limit is not None and processed >= limit:
            break

        image_path = find_image(image_id, image_dir)
        if image_path is None:
            print(f"  [WARNING] Image not found: {image_id} — skipping")
            continue

        try:
            raw = infer_fn(image_path)
        except Exception as exc:
            # Keep going on transient API/decoding errors; --resume recovers the rest.
            print(f"  [ERROR] Inference failed for {image_id}: {exc} — skipping")
            continue

        results.append(build_record(image_id, raw))
        processed += 1

        if len(results) >= flush_every:
            _append_csv(results, output_file)
            results = []

    if results:
        _append_csv(results, output_file)

    print(f"Done. Processed {processed} images. Results saved to {output_file}")


def add_io_args(parser, *, require_p_version: bool) -> None:
    """Registers the CLI arguments shared by every inference script."""
    parser.add_argument(
        "--dataset_csv",
        required=True,
        help="CSV with an 'id' column (source of image IDs; e.g. a percept_dataset_* file)",
    )
    parser.add_argument(
        "--image_dir",
        default=os.environ.get("IMAGE_DIR", "/mnt/raid5/neemias/perceptdata"),
        help="Root directory containing part1..part6 image subdirectories",
    )
    parser.add_argument(
        "--save_path",
        required=True,
        help="Directory where the output CSV will be written",
    )
    parser.add_argument(
        "--alpha_version",
        default=None,
        help="Alpha (sigma) version used only for output naming, e.g. 3 or 5",
    )
    parser.add_argument(
        "--p_version",
        required=require_p_version,
        choices=["p2", "p2plus", "p2neg", "p3", "p5"],
        help="Problem version: p2, p2plus, p2neg, p3 or p5 (drives Task 1 labels)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N images (useful for a quick smoke test)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already present in the output CSV (resume an interrupted run)",
    )
