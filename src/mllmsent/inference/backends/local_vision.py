"""Shared loader for the locally-hosted vision LLMs (Phi-4 and Gemma-4).

Both are Hugging Face vision models driven through a chat template, differing
only in the model class, the default id, and whether a revision is pinned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from mllmsent.inference.backends.base import CAPTION, resolve_model, resolve_revision


@dataclass(frozen=True, slots=True)
class LocalVisionModel:
    load_model: Callable
    caption_max_new_tokens: int = 512
    classify_max_new_tokens: int = 64
    supports_thinking_flag: bool = False


def build_model_kwargs(args, cache_dir, revision):
    import torch

    kwargs: dict = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if revision:
        kwargs["revision"] = revision

    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs["device_map"] = {"": int(str(args.gpu_ids).split(",")[0])}
        return kwargs

    max_memory = {}
    for index in range(torch.cuda.device_count()):
        free_mb = torch.cuda.mem_get_info(index)[0] // (1024**2)
        max_memory[index] = f"{max(0, free_mb - 1500)}MiB"
    max_memory["cpu"] = "48GiB"
    kwargs["torch_dtype"] = torch.bfloat16
    kwargs["device_map"] = "auto"
    kwargs["max_memory"] = max_memory
    return kwargs


def create_local_vision_infer(args, task, prompt, backend, model: LocalVisionModel):
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_ids))

    model_id = resolve_model(args, backend)
    revision = resolve_revision(args, backend)
    cache_dir = args.cache_dir
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    processor_kwargs = {}
    if cache_dir:
        processor_kwargs["cache_dir"] = cache_dir
    if revision:
        processor_kwargs["revision"] = revision
    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)

    loaded = model.load_model(model_id, build_model_kwargs(args, cache_dir, revision))
    loaded.eval()

    first_param = next(
        parameter
        for parameter in loaded.parameters()
        if parameter.dtype not in (torch.uint8, torch.int8)
    )
    device, dtype = first_param.device, first_param.dtype
    print(f"  loaded {model_id} on {device} ({dtype})")

    max_new_tokens = args.max_new_tokens or (
        model.caption_max_new_tokens if task == CAPTION else model.classify_max_new_tokens
    )

    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if model.supports_thinking_flag:
        template_kwargs["enable_thinking"] = False

    def infer(image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = processor.apply_chat_template(messages, **template_kwargs)
        inputs = processor(text=text, images=[image], return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            output_ids = loaded.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        response = processor.decode(
            output_ids[:, input_len:][0], skip_special_tokens=True
        ).strip()
        torch.cuda.empty_cache()
        return response

    return infer
