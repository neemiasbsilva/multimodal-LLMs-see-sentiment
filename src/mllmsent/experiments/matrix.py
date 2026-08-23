"""Expansion of experiments.yaml into ExperimentSpec objects."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

from mllmsent.config import Paths, load_yaml, resolve_matrix_path
from mllmsent.experiments.spec import (
    BackboneProfile,
    ExperimentSpec,
    Hyperparameters,
    MllmProfile,
    resolve_fine_tuning,
)

IDENTITY_FIELDS = ("mllm", "backbone", "problem", "sigma", "track")


@dataclass(frozen=True, slots=True)
class Matrix:
    specs: tuple[ExperimentSpec, ...]
    paths: Paths
    hub: dict[str, str]
    source: Path

    def select(self, **filters) -> tuple[ExperimentSpec, ...]:
        selected = self.specs
        for field, wanted in filters.items():
            if wanted is None:
                continue
            allowed = set(wanted) if isinstance(wanted, (list, tuple, set)) else {wanted}
            selected = tuple(
                spec for spec in selected if getattr(spec, field) in allowed
            )
        return selected

    def get(self, spec_id: str) -> ExperimentSpec:
        for spec in self.specs:
            if spec.id == spec_id or spec.qualified_id == spec_id:
                return spec
        raise KeyError(f"unknown experiment: {spec_id}")


def _matches(spec_fields: dict, predicate: dict) -> bool:
    for field, wanted in predicate.items():
        value = spec_fields.get(field)
        allowed = set(wanted) if isinstance(wanted, (list, tuple, set)) else {wanted}
        if value not in allowed:
            return False
    return True


def _resolve_hyperparameters(
    track: dict,
    backbone: BackboneProfile,
    spec_fields: dict,
    spec_id: str,
) -> Hyperparameters:
    values = {
        "epochs": backbone.epochs,
        "max_len": backbone.max_len,
        **track.get("hyperparameters", {}),
    }
    for override in track.get("overrides", []):
        predicate = override.get("when")
        only = override.get("only")
        if predicate is not None and not _matches(spec_fields, predicate):
            continue
        if only is not None and spec_id not in only:
            continue
        values.update(override["set"])

    return Hyperparameters(
        learning_rate=float(values["learning_rate"]),
        batch_size=int(values["batch_size"]),
        epochs=int(values["epochs"]),
        max_len=values["max_len"],
        patience=int(values["patience"]),
    )


def _per_spec_override(block: dict, key: str, spec_id: str):
    mapping = block.get(key)
    if not isinstance(mapping, dict):
        return None
    return mapping.get(spec_id)



def _resolve_dir_name(block, spec_id, mllm_profile, backbone_profile, dir_kind_override, problem, sigma):
    override = _per_spec_override(block, "dir_override", spec_id)
    if override:
        return override
    kind = dir_kind_override or backbone_profile.dir_kind
    tokens = (
        mllm_profile.dir_prefix,
        backbone_profile.dir_token,
        kind,
        problem,
        None if sigma is None else f"alpha{sigma}",
    )
    return "-".join(token for token in tokens if token)


def _resolve_checkpoint_dir(track, block, spec_id, tree, dir_name):
    explicit = _per_spec_override(block, "checkpoint_dir_override", spec_id)
    if explicit:
        return explicit
    template = track.get("checkpoint_dir_template")
    if template:
        return template.format(tree=tree, dir_name=dir_name)
    return None

def load_matrix(path: str | Path | None = None) -> Matrix:
    source = resolve_matrix_path(path)
    document = load_yaml(source)
    root = source.parent

    mllm_profiles = {
        name: MllmProfile(name=name, **fields)
        for name, fields in document["mllms"].items()
    }
    backbone_profiles = {
        name: BackboneProfile(name=name, **fields)
        for name, fields in document["backbones"].items()
    }
    tracks = document["tracks"]
    aliases = document.get("checkpoint_aliases", {})

    specs: list[ExperimentSpec] = []
    for block in document["runs"]:
        track_name = block["track"]
        track = tracks[track_name]
        dir_kind_map = block.get("dir_kind") or {}

        combinations = itertools.product(
            block["mllm"], block["backbone"], block["problem"], block["sigma"]
        )
        for mllm, backbone, problem, sigma in combinations:
            backbone_profile = backbone_profiles[backbone]
            sigma_token = "none" if sigma is None else str(sigma)
            spec_id = f"{mllm}-{backbone}-{problem}-sigma{sigma_token}"
            spec_fields = {
                "mllm": mllm,
                "backbone": backbone,
                "problem": problem,
                "sigma": sigma,
                "track": track_name,
            }

            hyperparameters = _resolve_hyperparameters(
                track, backbone_profile, spec_fields, spec_id
            )
            dir_name = _resolve_dir_name(
                block,
                spec_id,
                mllm_profiles[mllm],
                backbone_profile,
                dir_kind_map.get(backbone),
                problem,
                sigma,
            )
            fine_tuning = resolve_fine_tuning(
                track["tree"], backbone_profile.finetuning_rule
            )
            specs.append(
                ExperimentSpec(
                    mllm=mllm,
                    backbone=backbone,
                    problem=problem,
                    sigma=sigma,
                    track=track_name,
                    tree=track["tree"],
                    objective=track["objective"],
                    validation=track["validation"],
                    freeze_backbone=track["freeze_backbone"],
                    hyperparameters=hyperparameters,
                    mllm_profile=mllm_profiles[mllm],
                    backbone_profile=backbone_profile,
                    dir_override=_per_spec_override(block, "dir_override", spec_id),
                    dir_kind_override=dir_kind_map.get(backbone),
                    checkpoint_alias=aliases.get(f"{spec_id}-{fine_tuning}"),
                    checkpoint_dir_override=_resolve_checkpoint_dir(
                        track, block, spec_id, track["tree"], dir_name
                    ),
                    dataset_override=_per_spec_override(
                        block, "dataset_override", spec_id
                    ),
                )
            )

    return Matrix(
        specs=tuple(specs),
        paths=Paths.from_mapping(document.get("paths", {}), root),
        hub=document.get("hub", {}),
        source=source,
    )
