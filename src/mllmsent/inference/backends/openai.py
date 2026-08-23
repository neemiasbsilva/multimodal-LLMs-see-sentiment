"""OpenAI GPT-4o mini backend (Task 1 and Task 2)."""

from __future__ import annotations

import os

from mllmsent.inference.backends.base import (
    CAPTION,
    Backend,
    load_env,
    require_env,
    resolve_model,
)
from mllmsent.inference.common import encode_image_base64


def create_infer(args, task: str, prompt: str):
    load_env()
    api_key = require_env("OPENAI_API_KEY")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = resolve_model(args, BACKEND)
    max_tokens = args.max_new_tokens or (512 if task == CAPTION else 300)

    def infer(image_path: str) -> str:
        encoded, mime = encode_image_base64(image_path)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=args.temperature,
        )
        return (response.choices[0].message.content or "").strip()

    return infer


BACKEND = Backend(
    name="openai",
    label="GPT-4o mini",
    create_infer=create_infer,
    default_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    default_caption_dir="gpt4-openai-only",
)
