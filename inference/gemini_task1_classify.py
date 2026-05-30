"""Task 1 — direct image-to-sentiment classification with Google Gemini.

Uses the google-genai SDK (default model gemini-2.5-flash), mirroring the OpenAI
Task 1 script. Requires GEMINI_API_KEY in ``.env``. Output CSV: [id, sentiment, raw_response].

    python inference/gemini_task1_classify.py --dataset_csv <csv> \\
        --save_path data/gemini-only --alpha_version 3 --p_version p3
"""

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    add_io_args,
    build_classification_prompt,
    build_output_path,
    extract_sentiment,
    read_image_bytes,
    run_over_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 1 — direct sentiment classification with Google Gemini"
    )
    add_io_args(parser, require_p_version=True)
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model id (default: env GEMINI_MODEL or gemini-2.5-flash)",
    )
    return parser.parse_args()


def load_env():
    """Loads ``.env`` if python-dotenv is available; env vars may also be preset."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ModuleNotFoundError:
        pass


def main():
    load_env()
    args = parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    prompt = build_classification_prompt(args.p_version)
    print(f"Model: {args.model}\nPrompt: {prompt}")

    df = pd.read_csv(args.dataset_csv)
    print(f"Loaded {len(df)} samples from {args.dataset_csv}")
    output_file = build_output_path(
        args.save_path, args.alpha_version, args.p_version, kind="classify"
    )

    def infer(image_path: str) -> str:
        raw_bytes, mime = read_image_bytes(image_path)
        response = client.models.generate_content(
            model=args.model,
            contents=[
                types.Part.from_bytes(data=raw_bytes, mime_type=mime),
                prompt,
            ],
        )
        return (response.text or "").strip()

    def build_record(image_id: str, raw: str) -> dict:
        return {
            "id": image_id,
            "sentiment": extract_sentiment(raw, args.p_version),
            "raw_response": raw,
        }

    run_over_dataset(
        df,
        args.image_dir,
        output_file,
        infer,
        build_record,
        resume=args.resume,
        limit=args.limit,
        desc="Gemini Task 1",
    )


if __name__ == "__main__":
    main()
