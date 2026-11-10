"""GPT-6.1 Terra backend (Task 1 and Task 2).

The GPT-4o mini request, unchanged, sent to gpt-6.1-terra: same prompt, same
image encoding, same 300-token cap and temperature. Reasoning defaults to
`none`, Terra's non-reasoning mode, so the two description sets differ only in
the model; --reasoning-effort low|medium|high|xhigh|max lets it think first.
"""

from __future__ import annotations

import os

from mllmsent.inference.backends import openai as openai_backend
from mllmsent.inference.backends.base import Backend


def create_infer(args, task: str, prompt: str):
    return openai_backend.create_infer(args, task, prompt, backend=BACKEND)


BACKEND = Backend(
    name="terra",
    label="GPT-6.1 Terra",
    create_infer=create_infer,
    default_model=os.environ.get("OPENAI_TERRA_MODEL", "gpt-6.1-terra"),
    default_caption_dir="gpt61-terra-only",
    default_reasoning_effort="none",
)
