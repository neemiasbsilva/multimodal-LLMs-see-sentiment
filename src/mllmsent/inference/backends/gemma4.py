"""Google Gemma-4 backend (Task 1 and Task 2), running locally on a GPU."""

from __future__ import annotations

from mllmsent.inference.backends.base import Backend
from mllmsent.inference.backends.local_vision import (
    LocalVisionModel,
    create_local_vision_infer,
)


def _load(model_id: str, kwargs: dict):
    from transformers import Gemma4ForConditionalGeneration

    return Gemma4ForConditionalGeneration.from_pretrained(model_id, **kwargs)


MODEL = LocalVisionModel(
    load_model=_load,
    caption_max_new_tokens=300,
    classify_max_new_tokens=64,
    supports_thinking_flag=True,
)


def create_infer(args, task: str, prompt: str):
    return create_local_vision_infer(args, task, prompt, BACKEND, MODEL)


BACKEND = Backend(
    name="gemma4",
    label="Gemma-4",
    create_infer=create_infer,
    default_model="google/gemma-4-E4B-it",
    default_caption_dir="gemma4-only",
)
