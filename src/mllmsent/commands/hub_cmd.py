"""`mllmsent hub` — convert, publish and retrieve both Hub repositories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mllmsent.experiments.matrix import load_matrix
from mllmsent.hub.fetch import GROUPS

MODEL_STAGING = "hub/models"
DATASET_STAGING = "hub/datasets"
STATE_FILENAME = "hub/upload_state.json"


def staging_path(matrix, override: str | None, default: str) -> Path:
    if override:
        return Path(override).resolve()
    return matrix.paths.output_root / default


def cmd_status(args) -> int:
    from mllmsent.hub.auth import describe_identity, read_token, resolve_api

    matrix = load_matrix(args.matrix)
    model_repo = matrix.hub.get("model_repo")
    dataset_repo = matrix.hub.get("dataset_repo")

    print(f"model repo   : {model_repo}")
    print(f"dataset repo : {dataset_repo}")

    if read_token() is None:
        print("token        : not found (run `hf auth login`)")
        return 0

    api = resolve_api()
    print(f"logged in as : {describe_identity(api)}")

    for repo_id, repo_type in ((model_repo, "model"), (dataset_repo, "dataset")):
        try:
            info = api.repo_info(repo_id, repo_type=repo_type)
            print(f"{repo_type:<12} : exists, {len(info.siblings or [])} files")
        except Exception as error:
            print(f"{repo_type:<12} : unavailable ({type(error).__name__})")

    models = staging_path(matrix, args.model_staging, MODEL_STAGING)
    datasets = staging_path(matrix, args.dataset_staging, DATASET_STAGING)
    for label, path in (("model staging", models), ("dataset staging", datasets)):
        if path.is_dir():
            files = sum(1 for entry in path.rglob("*") if entry.is_file())
            print(f"{label:<15}: {path} ({files} files)")
        else:
            print(f"{label:<15}: {path} (absent)")
    return 0


def cmd_convert(args) -> int:
    from mllmsent.hub.cards import write_model_card
    from mllmsent.hub.convert import convert_all

    matrix = load_matrix(args.matrix)
    staging = staging_path(matrix, args.staging, MODEL_STAGING)
    specs = None
    if args.only:
        specs = [matrix.get(spec_id) for spec_id in args.only]

    summary = convert_all(
        matrix, staging, dtype_name=args.dtype, force=args.force, specs=specs
    )
    write_model_card(staging)
    print(
        f"\nconverted={summary['converted']} skipped={summary['skipped']} "
        f"missing={summary['missing']} manifest_entries={summary['entries']}"
    )
    print(f"staging: {staging}")
    return 0


def cmd_push_models(args) -> int:
    from mllmsent.hub.auth import resolve_api
    from mllmsent.hub.cards import write_model_card
    from mllmsent.hub.push import UploadState, model_chunks, upload_chunks, upload_file

    matrix = load_matrix(args.matrix)
    staging = staging_path(matrix, args.staging, MODEL_STAGING)
    if not staging.is_dir():
        raise SystemExit(f"nothing staged at {staging}; run `mllmsent hub convert` first")

    repo_id = args.repo or matrix.hub["model_repo"]
    api = resolve_api(write=not args.dry_run)
    state = UploadState(matrix.paths.output_root / STATE_FILENAME)

    write_model_card(staging)
    summary = upload_chunks(
        api, repo_id, "model", staging, model_chunks(staging), state,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        for name in ("README.md", "MANIFEST.json"):
            path = staging / name
            if path.is_file():
                upload_file(api, repo_id, "model", path, name)

    print(f"\nuploaded={summary['uploaded']} skipped={summary['skipped']} of {summary['chunks']} chunks")
    print(f"https://huggingface.co/{repo_id}")
    return 0


def cmd_push_datasets(args) -> int:
    from mllmsent.hub.auth import resolve_api
    from mllmsent.hub.cards import write_dataset_card
    from mllmsent.hub.datasets import build_staging, dataset_chunks
    from mllmsent.hub.push import UploadState, upload_chunks, upload_file

    matrix = load_matrix(args.matrix)
    staging = staging_path(matrix, args.staging, DATASET_STAGING)
    counts = build_staging(matrix, staging)

    repo_id = args.repo or matrix.hub["dataset_repo"]
    api = resolve_api(write=not args.dry_run)
    state = UploadState(matrix.paths.output_root / STATE_FILENAME)

    write_dataset_card(staging, counts)
    summary = upload_chunks(
        api, repo_id, "dataset", staging, dataset_chunks(staging), state,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        upload_file(api, repo_id, "dataset", staging / "README.md", "README.md")

    print(f"\nuploaded={summary['uploaded']} skipped={summary['skipped']} of {summary['chunks']} chunks")
    print(f"https://huggingface.co/datasets/{repo_id}")
    return 0


def cmd_pull_datasets(args) -> int:
    from mllmsent.hub.fetch import download_dataset

    matrix = load_matrix(args.matrix)
    repo_id = args.repo or matrix.hub["dataset_repo"]
    summary = download_dataset(
        repo_id,
        matrix,
        groups=tuple(args.groups) if args.groups else GROUPS,
        revision=args.revision,
        force=args.force,
    )

    print(f"\nwritten={summary['written']} already present={summary['kept']}")
    print(f"https://huggingface.co/datasets/{repo_id}")
    return 0


def cmd_pull_checkpoint(args) -> int:
    from mllmsent.hub.pull import download_checkpoint, materialize_checkpoint

    matrix = load_matrix(args.matrix)
    spec = matrix.get(args.experiment_id)
    repo_id = args.repo or matrix.hub["model_repo"]
    destination = Path(args.dest) if args.dest else matrix.paths.output_root / "hub/pulled"
    folder = download_checkpoint(repo_id, spec, destination, revision=args.revision)

    checkpoint = spec.checkpoint_dir(matrix.paths.checkpoint_root) / spec.checkpoint_name
    if checkpoint.is_file() and not args.force:
        state = "already present"
    else:
        materialize_checkpoint(folder, checkpoint)
        state = "written"

    print(f"{spec.qualified_id}")
    print(f"  safetensors  {folder}")
    print(f"  checkpoint   {checkpoint} ({state})")
    return 0


def cmd_manifest(args) -> int:
    matrix = load_matrix(args.matrix)
    staging = staging_path(matrix, args.staging, MODEL_STAGING)
    path = staging / "MANIFEST.json"
    if not path.is_file():
        raise SystemExit(f"no manifest at {path}")
    manifest = json.loads(path.read_text())
    scored = [entry for entry in manifest.values() if entry.get("mean_f1")]
    scored.sort(key=lambda entry: entry["mean_f1"], reverse=True)
    print(f"{len(manifest)} checkpoints staged, {len(scored)} with scores\n")
    for entry in scored[: args.top]:
        print(f"  {entry['mean_f1']:.4f}  {entry['spec_id']:<40} {entry['track']}")
    return 0


def register(subparsers) -> None:
    parser = subparsers.add_parser("hub", help="publish to the Hugging Face Hub")
    actions = parser.add_subparsers(dest="action", required=True)

    status = actions.add_parser("status", help="auth, repos and staging state")
    status.add_argument("--model-staging")
    status.add_argument("--dataset-staging")
    status.set_defaults(handler=cmd_status)

    convert = actions.add_parser("convert", help="fp32 .pt -> safetensors")
    convert.add_argument("--staging")
    convert.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    convert.add_argument("--force", action="store_true")
    convert.add_argument("--only", nargs="*")
    convert.set_defaults(handler=cmd_convert)

    push_models = actions.add_parser("push-models", help="upload the model repo")
    push_models.add_argument("--staging")
    push_models.add_argument("--repo")
    push_models.add_argument("--dry-run", action="store_true")
    push_models.set_defaults(handler=cmd_push_models)

    push_datasets = actions.add_parser("push-datasets", help="upload the dataset repo")
    push_datasets.add_argument("--staging")
    push_datasets.add_argument("--repo")
    push_datasets.add_argument("--dry-run", action="store_true")
    push_datasets.set_defaults(handler=cmd_push_datasets)

    pull_datasets = actions.add_parser(
        "pull-datasets", help="download the inputs and results into data/ and output/"
    )
    pull_datasets.add_argument("--groups", nargs="*", choices=GROUPS)
    pull_datasets.add_argument("--repo")
    pull_datasets.add_argument("--revision")
    pull_datasets.add_argument("--force", action="store_true")
    pull_datasets.set_defaults(handler=cmd_pull_datasets)

    pull = actions.add_parser(
        "pull-checkpoint", help="download one checkpoint into checkpoints/"
    )
    pull.add_argument("experiment_id")
    pull.add_argument("--repo")
    pull.add_argument("--dest")
    pull.add_argument("--revision")
    pull.add_argument("--force", action="store_true")
    pull.set_defaults(handler=cmd_pull_checkpoint)

    manifest = actions.add_parser("manifest", help="summarise the staged manifest")
    manifest.add_argument("--staging")
    manifest.add_argument("--top", type=int, default=15)
    manifest.set_defaults(handler=cmd_manifest)
