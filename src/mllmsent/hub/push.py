"""Chunked, resumable uploads to the Hub.

upload_folder already streams, deduplicates through Xet and splits oversized
folders across commits. What it does not do is remember across process restarts
which chunks are already current, so a folder digest is recorded per chunk and
compared before re-uploading.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

IGNORE_PATTERNS = [".*", "**/__pycache__/**", "**/.DS_Store"]


def folder_digest(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(folder)).encode())
            digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()


class UploadState:
    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, str] = {}
        if path.is_file():
            try:
                self.entries = json.loads(path.read_text())
            except json.JSONDecodeError:
                self.entries = {}

    def is_current(self, key: str, digest: str) -> bool:
        return self.entries.get(key) == digest

    def record(self, key: str, digest: str) -> None:
        self.entries[key] = digest
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2, sort_keys=True) + "\n")


def model_chunks(staging: Path) -> list[Path]:
    """One chunk per {mllm}/{backbone}, keeping commits in the low-GB range."""
    chunks = []
    for mllm_dir in sorted(p for p in staging.iterdir() if p.is_dir()):
        for backbone_dir in sorted(p for p in mllm_dir.iterdir() if p.is_dir()):
            chunks.append(backbone_dir)
    return chunks


def upload_chunks(
    api,
    repo_id: str,
    repo_type: str,
    staging: Path,
    chunks: list[Path],
    state: UploadState,
    dry_run: bool = False,
    on_event=print,
) -> dict:
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=False)

    uploaded = skipped = 0
    for chunk in chunks:
        relative = chunk.relative_to(staging)
        key = f"{repo_type}:{relative}"
        digest = folder_digest(chunk)
        size_gb = sum(p.stat().st_size for p in chunk.rglob("*") if p.is_file()) / 2**30

        if state.is_current(key, digest):
            skipped += 1
            on_event(f"skip   {relative}  ({size_gb:.2f} GB)")
            continue

        on_event(f"upload {relative}  ({size_gb:.2f} GB)")
        if dry_run:
            continue

        api.upload_folder(
            folder_path=str(chunk),
            path_in_repo=str(relative),
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Add {relative}",
            ignore_patterns=IGNORE_PATTERNS,
        )
        state.record(key, digest)
        uploaded += 1

    return {"uploaded": uploaded, "skipped": skipped, "chunks": len(chunks)}


def upload_file(api, repo_id: str, repo_type: str, path: Path, path_in_repo: str) -> None:
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=f"Update {path_in_repo}",
    )
