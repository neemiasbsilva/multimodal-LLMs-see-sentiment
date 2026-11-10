"""OpenAI Chat Completions backend (Task 1 and Task 2).

Serves GPT-4o mini, the paper's configuration, and GPT-6.1 Terra through
mllmsent.inference.backends.terra. Both get the request the original
openai_task{1,2} scripts sent: the prompt plus the image as a base64 data URL
at the API's default detail, a 300-token answer cap and temperature 1.0.

Reasoning models add `reasoning_effort`. At `none` the cap and the temperature
keep their meaning, so a Terra run differs from the GPT-4o mini run only in the
model. Any other effort drops `temperature` (the API rejects it while the model
reasons) and lifts the cap, which would otherwise count the hidden reasoning
tokens and cut the answer short.
"""

from __future__ import annotations

import atexit
import os

from mllmsent.inference.backends.base import (
    Backend,
    load_env,
    require_env,
    resolve_model,
)
from mllmsent.inference.common import encode_image_base64

# The cap the original scripts applied to both tasks (openai_task*.py: --max_tokens 300).
PAPER_MAX_TOKENS = 300


def resolve_reasoning_effort(args, backend: Backend) -> str | None:
    """The effort to send, or None for models without reasoning (GPT-4o mini)."""
    requested = getattr(args, "reasoning_effort", None)
    if requested is None:
        return backend.default_reasoning_effort
    if backend.default_reasoning_effort is None:
        raise SystemExit(f"--reasoning-effort does not apply to {backend.label}")
    return requested


def resolve_max_tokens(args, reasoning_effort: str | None) -> int | None:
    if args.max_new_tokens is not None:
        return args.max_new_tokens
    if reasoning_effort in (None, "none"):
        return PAPER_MAX_TOKENS
    return None


def build_request(
    model: str,
    prompt: str,
    image_url: str,
    *,
    max_tokens: int | None,
    temperature: float,
    reasoning_effort: str | None,
) -> dict:
    """The chat.completions.create keyword arguments for one image."""
    request: dict = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }
    if reasoning_effort is None:
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        request["temperature"] = temperature
        return request

    request["reasoning_effort"] = reasoning_effort
    if max_tokens is not None:
        # max_tokens is deprecated for reasoning models; this one also counts
        # the reasoning tokens.
        request["max_completion_tokens"] = max_tokens
    if reasoning_effort == "none":
        request["temperature"] = temperature
    return request


class UsageTally:
    """Token totals for the run, printed at exit so a pilot prices the full run."""

    def __init__(self, label: str, model: str):
        self.label = label
        self.model = model
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.truncated = 0

    def add(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.calls += 1
        self.input_tokens += usage.prompt_tokens or 0
        self.output_tokens += usage.completion_tokens or 0
        details = getattr(usage, "completion_tokens_details", None)
        self.reasoning_tokens += getattr(details, "reasoning_tokens", None) or 0
        if response.choices and response.choices[0].finish_reason == "length":
            self.truncated += 1

    def report(self) -> None:
        if not self.calls:
            return
        print(
            f"{self.label} ({self.model}) usage: {self.calls} calls, "
            f"{self.input_tokens:,} input tokens, {self.output_tokens:,} output tokens "
            f"({self.reasoning_tokens:,} reasoning), {self.truncated} answers cut by the cap"
        )


def create_infer(args, task: str, prompt: str, backend: Backend | None = None):
    backend = backend or BACKEND
    load_env()
    api_key = require_env("OPENAI_API_KEY")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = resolve_model(args, backend)
    reasoning_effort = resolve_reasoning_effort(args, backend)
    max_tokens = resolve_max_tokens(args, reasoning_effort)

    settings = f"max_tokens={max_tokens if max_tokens is not None else 'uncapped'}"
    if reasoning_effort is not None:
        settings = f"reasoning_effort={reasoning_effort}  {settings}"
    if reasoning_effort in (None, "none"):
        settings += f"  temperature={args.temperature}"
    print(f"  model   : {model}  {settings}")
    if reasoning_effort not in (None, "none") and args.max_new_tokens is None:
        print(
            "  note    : with reasoning on, a token cap also counts the reasoning tokens,"
            " so none is applied; pass --max-new-tokens to set one"
        )

    tally = UsageTally(backend.label, model)
    atexit.register(tally.report)

    def infer(image_path: str) -> str:
        encoded, mime = encode_image_base64(image_path)
        response = client.chat.completions.create(
            **build_request(
                model,
                prompt,
                f"data:{mime};base64,{encoded}",
                max_tokens=max_tokens,
                temperature=args.temperature,
                reasoning_effort=reasoning_effort,
            )
        )
        tally.add(response)
        return (response.choices[0].message.content or "").strip()

    return infer


BACKEND = Backend(
    name="openai",
    label="GPT-4o mini",
    create_infer=create_infer,
    default_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    default_caption_dir="gpt4-openai-only",
)
