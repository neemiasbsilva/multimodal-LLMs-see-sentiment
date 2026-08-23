"""Backend contract for the MLLM stage.

A backend turns parsed options plus a task into an `infer(image_path) -> str`
callable. Everything else — prompts, image lookup, label extraction, resume and
incremental CSV writing — lives in mllmsent.inference.common and is identical
across model families.

Flags are a single union shared by every backend rather than one set per model,
so switching --mllm never changes the rest of the command line.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

CLASSIFY = "classify"
CAPTION = "caption"


@dataclass(frozen=True, slots=True)
class Backend:
    name: str
    label: str
    create_infer: Callable
    default_model: str
    default_caption_dir: str | None = None
    default_revision: str | None = None


def add_backend_args(parser) -> None:
    group = parser.add_argument_group("model options")
    group.add_argument(
        "--model",
        default=None,
        help="model id; defaults to the backend's paper configuration",
    )
    group.add_argument(
        "--revision",
        default=None,
        help="pinned model revision (local backends only)",
    )
    group.add_argument(
        "--cache-dir",
        dest="cache_dir",
        default=os.environ.get("HF_HOME"),
        help="Hugging Face cache directory for locally-hosted weights",
    )
    group.add_argument(
        "--gpu-ids",
        dest="gpu_ids",
        default="0",
        help="comma-separated GPU indices for locally-hosted models",
    )
    group.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=None)
    group.add_argument("--temperature", type=float, default=1.0)
    group.add_argument(
        "--4bit",
        dest="load_4bit",
        action="store_true",
        help="load a local model in 4-bit to fit a smaller GPU",
    )


def resolve_model(args, backend: Backend) -> str:
    return args.model or backend.default_model


def resolve_revision(args, backend: Backend) -> str | None:
    return args.revision if args.revision is not None else backend.default_revision


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ModuleNotFoundError:
        pass


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value
