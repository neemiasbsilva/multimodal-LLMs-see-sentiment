"""Google Gemini backend (Task 1 and Task 2)."""

from __future__ import annotations

import os

from mllmsent.inference.backends.base import (
    Backend,
    load_env,
    require_env,
    resolve_model,
)
from mllmsent.inference.common import read_image_bytes


def create_infer(args, task: str, prompt: str):
    load_env()
    api_key = require_env("GEMINI_API_KEY")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model = resolve_model(args, BACKEND)

    def infer(image_path: str) -> str:
        raw_bytes, mime = read_image_bytes(image_path)
        response = client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=raw_bytes, mime_type=mime), prompt],
        )
        return (response.text or "").strip()

    return infer


BACKEND = Backend(
    name="gemini",
    label="Gemini",
    create_infer=create_infer,
    default_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    default_caption_dir="gemini-only",
)
