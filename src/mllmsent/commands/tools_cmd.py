"""Subcommands that wrap the analysis and evaluation modules.

Each of these modules keeps its own argparse entry point; the CLI forwards the
remaining arguments so `mllmsent evaluate stats --alpha 0.01` behaves exactly
like calling the module directly, without the sys.path dance the old scripts
needed.
"""

from __future__ import annotations

import runpy
import sys

WRAPPED = {
    "predict": ("mllmsent.inference.checkpoint", "score a CSV with a trained checkpoint"),
    "attention-flow": ("mllmsent.analysis.attention_flow", "attention rollout heatmaps"),
    "build-data": ("mllmsent.data.build", "rebuild the per-sigma/problem CSVs"),
}

EVALUATIONS = {
    "kfold": ("mllmsent.evaluation.kfold", "k-fold metrics per model"),
    "stats": ("mllmsent.evaluation.stats_tests", "paired t-tests with Holm correction"),
    "posthoc": ("mllmsent.evaluation.posthoc", "post-hoc tests against the baselines"),
    "paired-ttest": ("mllmsent.evaluation.paired_ttest", "Task 1 paired t-test"),
    "vader": ("mllmsent.evaluation.vader", "VADER lexicon baseline"),
    "zero-shot": ("mllmsent.evaluation.zero_shot", "zero-shot baseline"),
}


def forward(module: str, extra: list[str]) -> int:
    sys.argv = [module.rsplit(".", 1)[-1], *extra]
    runpy.run_module(module, run_name="__main__")
    return 0


def make_handler(module: str):
    def handler(args) -> int:
        return forward(module, args.extra)

    return handler


def add_passthrough(parser, module: str) -> None:
    parser.add_argument(
        "extra",
        nargs="*",
        help="arguments forwarded verbatim; use -- before flags the CLI also defines",
    )
    parser.set_defaults(handler=make_handler(module))


def register(subparsers) -> None:
    for name, (module, help_text) in WRAPPED.items():
        parser = subparsers.add_parser(name, help=help_text)
        add_passthrough(parser, module)

    evaluate = subparsers.add_parser("evaluate", help="metrics and statistical tests")
    actions = evaluate.add_subparsers(dest="analysis", required=True)
    for name, (module, help_text) in EVALUATIONS.items():
        parser = actions.add_parser(name, help=help_text)
        add_passthrough(parser, module)
