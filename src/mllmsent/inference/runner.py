"""Dispatch of Task 1 (classify) and Task 2 (caption) across MLLM backends."""

from __future__ import annotations

import importlib

import pandas as pd

from mllmsent.inference.backends.base import CAPTION, CLASSIFY, Backend
from mllmsent.inference.common import (
    DESCRIPTION_PROMPT,
    build_classification_prompt,
    build_output_path,
    extract_sentiment,
    run_over_dataset,
)

BACKEND_MODULES = {
    "openai": "mllmsent.inference.backends.openai",
    "gemini": "mllmsent.inference.backends.gemini",
    "deepseek": "mllmsent.inference.backends.deepseek",
    "phi4": "mllmsent.inference.backends.phi4",
    "gemma4": "mllmsent.inference.backends.gemma4",
}

BACKEND_NAMES = tuple(BACKEND_MODULES)


def load_backend(name: str) -> Backend:
    if name not in BACKEND_MODULES:
        raise SystemExit(
            f"unknown MLLM '{name}'; choose from {', '.join(BACKEND_NAMES)}"
        )
    return importlib.import_module(BACKEND_MODULES[name]).BACKEND


def build_record_factory(task: str, problem: str):
    if task == CAPTION:
        return lambda image_id, raw: {"id": image_id, "text": raw}
    return lambda image_id, raw: {
        "id": image_id,
        "sentiment": extract_sentiment(raw, problem),
        "raw_response": raw,
    }


def run(backend: Backend, task: str, args) -> str:
    problem = getattr(args, "p_version", None)
    prompt = DESCRIPTION_PROMPT if task == CAPTION else build_classification_prompt(problem)

    frame = pd.read_csv(args.dataset_csv)
    output_file = build_output_path(
        args.save_path, args.alpha_version, problem, kind=task
    )

    print(f"{backend.label} — Task {'2' if task == CAPTION else '1'}")
    print(f"  dataset : {args.dataset_csv} ({len(frame)} rows)")
    print(f"  output  : {output_file}")
    print(f"  prompt  : {prompt}")

    infer = backend.create_infer(args, task, prompt)
    run_over_dataset(
        frame,
        args.image_dir,
        output_file,
        infer,
        build_record_factory(task, problem),
        resume=args.resume,
        limit=args.limit,
        desc=f"{backend.label} Task {'2' if task == CAPTION else '1'}",
    )
    return output_file
