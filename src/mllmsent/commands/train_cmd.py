"""`mllmsent train` and `mllmsent sweep` — replaces the 13 config-list shell scripts."""

from __future__ import annotations

import sys
import time

from mllmsent.commands.matrix_cmd import add_filter_args, selected_specs
from mllmsent.experiments.matrix import load_matrix
from mllmsent.experiments.resolve import checkpoint_path


def resolve_targets(matrix, args):
    if getattr(args, "experiment_id", None):
        return [matrix.get(spec_id) for spec_id in args.experiment_id]
    return list(selected_specs(matrix, args))


def has_results(matrix, spec) -> bool:
    return (matrix.paths.results_root / spec.log_dir / "test_logs.csv").is_file()


def describe(matrix, spec) -> str:
    checkpoint = checkpoint_path(spec, matrix.paths.checkpoint_root)
    return (
        f"{spec.qualified_id}\n"
        f"    dataset    {spec.dataset_csv(matrix.paths.data_root)}\n"
        f"    log dir    {spec.log_dir}\n"
        f"    checkpoint {checkpoint}\n"
        f"    model      {spec.backbone_profile.model_path}\n"
        f"    lr={spec.hyperparameters.learning_rate} "
        f"bs={spec.hyperparameters.batch_size} "
        f"epochs={spec.hyperparameters.epochs} "
        f"patience={spec.hyperparameters.patience} "
        f"frozen={spec.freeze_backbone}"
    )


def run_specs(matrix, specs, args) -> int:
    if args.dry_run:
        for spec in specs:
            print(describe(matrix, spec))
        print(f"\n{len(specs)} experiments (dry run, nothing trained)", file=sys.stderr)
        return 0

    from mllmsent.training.trainer import train

    failures = []
    for index, spec in enumerate(specs, start=1):
        if getattr(args, "skip_existing", False) and has_results(matrix, spec):
            print(f"[{index}/{len(specs)}] skip {spec.qualified_id} (results present)")
            continue

        print(f"\n[{index}/{len(specs)}] {spec.qualified_id}")
        started = time.time()
        try:
            train(spec, matrix, folds=args.folds, max_epochs=args.max_epochs)
        except Exception as error:
            failures.append((spec.qualified_id, error))
            print(f"  FAILED: {type(error).__name__}: {error}", file=sys.stderr)
            if not getattr(args, "continue_on_error", False):
                raise
        else:
            print(f"  done in {time.time() - started:.0f}s")

    if failures:
        print(f"\n{len(failures)} of {len(specs)} failed:", file=sys.stderr)
        for name, error in failures:
            print(f"  {name}: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


def cmd_train(args) -> int:
    matrix = load_matrix(args.matrix)
    specs = resolve_targets(matrix, args)
    if not specs:
        raise SystemExit("no experiments matched")
    return run_specs(matrix, specs, args)


def cmd_sweep(args) -> int:
    matrix = load_matrix(args.matrix)
    specs = resolve_targets(matrix, args)
    if not specs:
        raise SystemExit("no experiments matched")
    print(f"{len(specs)} experiments selected", file=sys.stderr)
    return run_specs(matrix, specs, args)


def add_run_args(parser) -> None:
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--max-epochs",
        dest="max_epochs",
        type=int,
        default=None,
        help="cap epochs regardless of the matrix (smoke tests)",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="print every resolved path and hyperparameter, train nothing",
    )


def register(subparsers) -> None:
    train = subparsers.add_parser("train", help="train one or more experiments")
    train.add_argument("experiment_id", nargs="*")
    add_filter_args(train)
    add_run_args(train)
    train.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    train.add_argument(
        "--continue-on-error", dest="continue_on_error", action="store_true"
    )
    train.set_defaults(handler=cmd_train)

    sweep = subparsers.add_parser(
        "sweep", help="train every experiment matching the filters"
    )
    add_filter_args(sweep)
    add_run_args(sweep)
    sweep.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    sweep.add_argument(
        "--continue-on-error", dest="continue_on_error", action="store_true"
    )
    sweep.set_defaults(handler=cmd_sweep, experiment_id=[])
