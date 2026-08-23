"""Entry point for the `mllmsent` command.

CUDA_VISIBLE_DEVICES has to be set before torch is imported anywhere, so --gpu
is applied here and every subcommand imports its own modules lazily inside its
handler. That also keeps `mllmsent matrix ...` fast enough to use in a loop.
"""

from __future__ import annotations

import argparse
import os
import sys

from mllmsent import __version__
from mllmsent.commands import (
    hub_cmd,
    inference_cmd,
    matrix_cmd,
    tools_cmd,
    train_cmd,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mllmsent",
        description="MLLMsent: multimodal LLM captioning and sentiment classification.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--matrix",
        default=None,
        help="path to experiments.yaml (default: nearest one above the cwd)",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="value for CUDA_VISIBLE_DEVICES, applied before torch is imported",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix_cmd.register(subparsers)
    inference_cmd.register(subparsers)
    train_cmd.register(subparsers)
    tools_cmd.register(subparsers)
    hub_cmd.register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
