"""Task 1 — direct image-to-sentiment classification with OpenAI GPT-4o mini.

The MLLM stage of the paper's best configuration (GPT-4o mini + ModernBERT =
MLLMsent). Requires OPENAI_API_KEY in ``.env``. Output CSV: [id, sentiment, raw_response].

    python inference/openai_task1_classify.py --dataset_csv <csv> \\
        --save_path data/gpt4-openai-only --alpha_version 3 --p_version p3
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
    encode_image_base64,
    extract_sentiment,
    run_over_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 1 — direct sentiment classification with OpenAI GPT-4o mini"
    )
    add_io_args(parser, require_p_version=True)
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model id (default: env OPENAI_MODEL or gpt-4o-mini)",
    )
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (paper used the API default of 1.0)",
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

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt = build_classification_prompt(args.p_version)
    print(f"Model: {args.model}\nPrompt: {prompt}")

    df = pd.read_csv(args.dataset_csv)
    print(f"Loaded {len(df)} samples from {args.dataset_csv}")
    output_file = build_output_path(
        args.save_path, args.alpha_version, args.p_version, kind="classify"
    )

    def infer(image_path: str) -> str:
        b64, mime = encode_image_base64(image_path)
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        return (response.choices[0].message.content or "").strip()

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
        desc="GPT-4o mini Task 1",
    )


if __name__ == "__main__":
    main()
