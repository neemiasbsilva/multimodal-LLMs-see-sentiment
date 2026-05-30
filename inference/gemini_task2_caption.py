"""Task 2 — image description generation with Google Gemini.

First stage of the description-mediated pipeline using the google-genai SDK
(default model gemini-2.5-flash); descriptions feed the text-only classifier.
Requires GEMINI_API_KEY in ``.env``. Output CSV (descriptions.csv): [id, text].

    python inference/gemini_task2_caption.py --dataset_csv <csv> \\
        --save_path data/gemini-classify
"""

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    DESCRIPTION_PROMPT,
    add_io_args,
    build_output_path,
    read_image_bytes,
    run_over_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 2 — image description generation with Google Gemini"
    )
    add_io_args(parser, require_p_version=False)
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
    print(f"Model: {args.model}\nPrompt: {DESCRIPTION_PROMPT}")

    df = pd.read_csv(args.dataset_csv)
    print(f"Loaded {len(df)} samples from {args.dataset_csv}")
    output_file = build_output_path(
        args.save_path, args.alpha_version, args.p_version, kind="caption"
    )

    def infer(image_path: str) -> str:
        raw_bytes, mime = read_image_bytes(image_path)
        response = client.models.generate_content(
            model=args.model,
            contents=[
                types.Part.from_bytes(data=raw_bytes, mime_type=mime),
                DESCRIPTION_PROMPT,
            ],
        )
        return (response.text or "").strip()

    def build_record(image_id: str, raw: str) -> dict:
        return {"id": image_id, "text": raw}

    run_over_dataset(
        df,
        args.image_dir,
        output_file,
        infer,
        build_record,
        resume=args.resume,
        limit=args.limit,
        desc="Gemini Task 2",
    )


if __name__ == "__main__":
    main()
