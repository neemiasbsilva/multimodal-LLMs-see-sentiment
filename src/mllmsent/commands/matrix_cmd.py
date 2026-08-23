"""`mllmsent matrix` — inspect and validate the experiment matrix."""

from __future__ import annotations

import argparse
import json
import sys

from mllmsent.experiments.matrix import load_matrix
from mllmsent.experiments.resolve import COLUMNS, project


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--track", nargs="*")
    parser.add_argument("--mllm", nargs="*")
    parser.add_argument("--backbone", nargs="*")
    parser.add_argument("--problem", nargs="*")
    parser.add_argument("--sigma", nargs="*", type=int)


def selected_specs(matrix, args):
    return matrix.select(
        track=getattr(args, "track", None),
        mllm=getattr(args, "mllm", None),
        backbone=getattr(args, "backbone", None),
        problem=getattr(args, "problem", None),
        sigma=getattr(args, "sigma", None),
    )


def cmd_list(args) -> int:
    matrix = load_matrix(args.matrix)
    specs = selected_specs(matrix, args)
    if args.format == "json":
        print(json.dumps([project(spec, matrix) for spec in specs], indent=2))
        return 0
    for spec in specs:
        print(f"{spec.qualified_id}\t{spec.dir_name}")
    print(f"{len(specs)} experiments", file=sys.stderr)
    return 0


def cmd_resolve(args) -> int:
    matrix = load_matrix(args.matrix)
    specs = selected_specs(matrix, args)
    print("\t".join(COLUMNS))
    for spec in specs:
        row = project(spec, matrix)
        print("\t".join(row[column] for column in COLUMNS))
    print(f"resolved {len(specs)} experiments", file=sys.stderr)
    return 0


def cmd_show(args) -> int:
    matrix = load_matrix(args.matrix)
    row = project(matrix.get(args.experiment_id), matrix)
    width = max(len(column) for column in COLUMNS)
    for column in COLUMNS:
        print(f"{column:<{width}}  {row[column]}")
    return 0


def cmd_check(args) -> int:
    matrix = load_matrix(args.matrix)
    results_root = matrix.paths.results_root
    checkpoint_root = matrix.paths.checkpoint_root
    data_root = matrix.paths.data_root

    missing_logs, missing_data, resolved, unresolved = [], [], [], []
    for spec in matrix.specs:
        if not (results_root / spec.log_dir).is_dir():
            missing_logs.append(spec.qualified_id)
        if not spec.dataset_csv(data_root).is_file():
            missing_data.append((spec.qualified_id, str(spec.dataset_csv(data_root))))
        if spec.writes_per_fold_checkpoints or spec.backbone == "swin":
            continue
        path = spec.resolve_checkpoint(checkpoint_root)
        (resolved if path.is_file() else unresolved).append(spec.qualified_id)

    print(f"experiments        : {len(matrix.specs)}")
    print(f"log dirs present   : {len(matrix.specs) - len(missing_logs)}")
    print(f"datasets present   : {len(matrix.specs) - len(missing_data)}")
    print(f"checkpoints on disk: {len(resolved)}")
    print(f"checkpoints absent : {len(unresolved)}")

    for label, entries in (
        ("missing log dir", missing_logs),
        ("missing dataset", [f"{name}  ->  {path}" for name, path in missing_data]),
    ):
        for entry in entries:
            print(f"  [{label}] {entry}", file=sys.stderr)

    if args.strict and (missing_logs or missing_data):
        return 1
    return 0


def register(subparsers) -> None:
    parser = subparsers.add_parser("matrix", help="inspect the experiment matrix")
    actions = parser.add_subparsers(dest="action", required=True)

    listing = actions.add_parser("list", help="list experiment ids")
    listing.add_argument("--format", choices=("table", "json"), default="table")
    add_filter_args(listing)
    listing.set_defaults(handler=cmd_list)

    resolving = actions.add_parser("resolve", help="emit every derived field as TSV")
    add_filter_args(resolving)
    resolving.set_defaults(handler=cmd_resolve)

    showing = actions.add_parser("show", help="show one experiment")
    showing.add_argument("experiment_id")
    showing.set_defaults(handler=cmd_show)

    checking = actions.add_parser("check", help="verify the matrix against disk")
    checking.add_argument("--strict", action="store_true")
    checking.set_defaults(handler=cmd_check)
