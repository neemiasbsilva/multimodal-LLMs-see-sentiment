"""Loading converted checkpoints back, from the Hub or from a local folder.

The Hub carries fp16 safetensors; training, predict and evaluate all load a
pickled fp32 state_dict from checkpoints/. materialize_checkpoint closes that
gap, so a pulled checkpoint is indistinguishable from a locally trained one.
"""

from __future__ import annotations

import json
from pathlib import Path

from mllmsent.hub.convert import CONFIG_FILENAME, WEIGHTS_FILENAME, repo_subpath


def load_state_dict(folder: Path) -> tuple[dict, dict]:
    """Returns (state_dict, config), restoring tensors dropped as tied aliases."""
    from safetensors.torch import load_file

    config = json.loads((folder / CONFIG_FILENAME).read_text())
    state = load_file(str(folder / WEIGHTS_FILENAME))
    for alias, owner in config.get("tied_weights", {}).items():
        state[alias] = state[owner]
    return state, config


def download_checkpoint(
    repo_id: str,
    spec,
    destination: Path,
    revision: str | None = None,
) -> Path:
    from huggingface_hub import hf_hub_download

    subpath = repo_subpath(spec)
    destination.mkdir(parents=True, exist_ok=True)
    for filename in (WEIGHTS_FILENAME, CONFIG_FILENAME):
        hf_hub_download(
            repo_id=repo_id,
            filename=f"{subpath}/{filename}",
            revision=revision,
            local_dir=str(destination),
        )
    return destination / subpath


def materialize_checkpoint(folder: Path, destination: Path) -> Path:
    import torch

    state, _ = load_state_dict(folder)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            key: tensor.float() if tensor.is_floating_point() else tensor
            for key, tensor in state.items()
        },
        destination,
    )
    return destination
