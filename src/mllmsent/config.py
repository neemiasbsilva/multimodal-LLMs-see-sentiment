"""Repository paths and the `!env` YAML tag used by experiments.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

MATRIX_FILENAME = "experiments.yaml"
MATRIX_ENV_VAR = "MLLMSENT_MATRIX"


class MatrixLoader(yaml.SafeLoader):
    pass


def _env_constructor(loader: MatrixLoader, node: yaml.Node) -> str:
    values = loader.construct_sequence(node)
    name = values[0]
    default = values[1] if len(values) > 1 else ""
    return os.environ.get(name, default)


MatrixLoader.add_constructor("!env", _env_constructor)


def load_yaml(path: Path) -> dict:
    with open(path) as handle:
        return yaml.load(handle, Loader=MatrixLoader)


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / MATRIX_FILENAME).is_file():
            return candidate
    return current


def resolve_matrix_path(explicit: str | os.PathLike | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    from_env = os.environ.get(MATRIX_ENV_VAR)
    if from_env:
        return Path(from_env).resolve()
    return find_repo_root() / MATRIX_FILENAME


@dataclass(frozen=True, slots=True)
class Paths:
    root: Path
    data_root: Path
    output_root: Path
    checkpoint_root: Path
    results_root: Path
    image_root: Path
    twitter_captions_root: Path

    @classmethod
    def from_mapping(cls, mapping: dict, root: Path) -> "Paths":
        def resolve(key: str, default: str) -> Path:
            value = Path(mapping.get(key, default)).expanduser()
            return value if value.is_absolute() else root / value

        return cls(
            root=root,
            data_root=resolve("data_root", "data"),
            output_root=resolve("output_root", "output"),
            checkpoint_root=resolve("checkpoint_root", "checkpoints"),
            results_root=resolve("results_root", "output/experiments"),
            image_root=resolve("image_root", "perceptdata"),
            twitter_captions_root=resolve("twitter_captions_root", "data/twitter"),
        )
