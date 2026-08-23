"""DeepSeek-VL2 backend (Task 1 and Task 2), running locally on a GPU."""

from __future__ import annotations

from mllmsent.inference.backends.base import CAPTION, Backend, resolve_model


def create_infer(args, task: str, prompt: str):
    from mllmsent.inference.backends.deepseek_runtime import DeepSeekVL2

    runtime = DeepSeekVL2(
        model_path=resolve_model(args, BACKEND),
        cache_dir=args.cache_dir,
        gpu_id=int(str(args.gpu_ids).split(",")[0]),
    )
    max_new_tokens = args.max_new_tokens or (512 if task == CAPTION else 64)

    def infer(image_path: str) -> str:
        return runtime.generate(image_path, prompt, max_new_tokens=max_new_tokens)

    return infer


BACKEND = Backend(
    name="deepseek",
    label="DeepSeek-VL2",
    create_infer=create_infer,
    default_model="deepseek-ai/deepseek-vl2-small",
    default_caption_dir="deepseek-only",
)
