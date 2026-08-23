"""Microsoft Phi-4 vision backend (Task 1 and Task 2), running locally on a GPU."""

from __future__ import annotations

from mllmsent.inference.backends.base import Backend
from mllmsent.inference.backends.local_vision import (
    LocalVisionModel,
    create_local_vision_infer,
)


def _load(model_id: str, kwargs: dict):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, **kwargs)


MODEL = LocalVisionModel(
    load_model=_load,
    caption_max_new_tokens=512,
    classify_max_new_tokens=64,
)


def create_infer(args, task: str, prompt: str):
    return create_local_vision_infer(args, task, prompt, BACKEND, MODEL)


BACKEND = Backend(
    name="phi4",
    label="Phi-4 vision",
    create_infer=create_infer,
    default_model="microsoft/Phi-4-reasoning-vision-15B",
    default_revision="7df902e2fec305ff57c2eddf519485a74bb2daaa",
    default_caption_dir="phi4-only",
)
