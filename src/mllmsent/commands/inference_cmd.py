"""`mllmsent classify` and `mllmsent caption` — the MLLM stage of the pipeline."""

from __future__ import annotations

from mllmsent.experiments.matrix import load_matrix
from mllmsent.inference.backends.base import CAPTION, CLASSIFY, add_backend_args
from mllmsent.inference.runner import BACKEND_NAMES, load_backend, run


def add_common_args(parser, task: str) -> None:
    parser.add_argument("--mllm", required=True, choices=BACKEND_NAMES)
    parser.add_argument(
        "--sigma",
        dest="alpha_version",
        help="annotator-agreement threshold; picks the source CSV and names the output",
    )
    parser.add_argument(
        "--p",
        dest="p_version",
        choices=["p2", "p2plus", "p2neg", "p3", "p5"],
        required=task == CLASSIFY,
        help="problem version; drives the Task 1 label set",
    )
    parser.add_argument("--dataset-csv", dest="dataset_csv")
    parser.add_argument("--image-dir", dest="image_dir")
    parser.add_argument("--save-path", dest="save_path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    add_backend_args(parser)


def resolve_paths(args, task: str) -> None:
    matrix = load_matrix(args.matrix)

    if args.image_dir is None:
        args.image_dir = str(matrix.paths.image_root)

    candidates = matrix.select(mllm=args.mllm)
    if not candidates:
        raise SystemExit(f"no experiments declared for mllm={args.mllm}")

    if args.dataset_csv is None:
        if args.alpha_version is None:
            raise SystemExit("pass --dataset-csv, or --sigma (and --p for classify)")
        # Task 2 descriptions are p-independent, so any problem at this sigma
        # supplies the same image ids.
        source = next(
            (
                spec
                for spec in candidates
                if str(spec.sigma) == str(args.alpha_version)
                and (args.p_version is None or spec.problem == args.p_version)
            ),
            None,
        )
        if source is None:
            raise SystemExit(
                f"no experiment for mllm={args.mllm} p={args.p_version} "
                f"sigma={args.alpha_version}"
            )
        args.dataset_csv = str(source.dataset_csv(matrix.paths.data_root))

    if args.save_path is None:
        backend = load_backend(args.mllm)
        subdir = (
            candidates[0].mllm_profile.dataset_dir
            if task == CAPTION
            else backend.default_caption_dir
        )
        args.save_path = str(matrix.paths.data_root / subdir)


def cmd_classify(args) -> int:
    resolve_paths(args, CLASSIFY)
    run(load_backend(args.mllm), CLASSIFY, args)
    return 0


def cmd_caption(args) -> int:
    resolve_paths(args, CAPTION)
    run(load_backend(args.mllm), CAPTION, args)
    return 0


def register(subparsers) -> None:
    classify = subparsers.add_parser(
        "classify", help="Task 1: classify sentiment directly from the image"
    )
    add_common_args(classify, CLASSIFY)
    classify.set_defaults(handler=cmd_classify)

    caption = subparsers.add_parser(
        "caption", help="Task 2: generate the image description"
    )
    add_common_args(caption, CAPTION)
    caption.set_defaults(handler=cmd_caption)
