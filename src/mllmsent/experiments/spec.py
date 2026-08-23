"""Experiment identity as data.

Before this module, an experiment's identity was recovered by string-splitting
the path of its config.yaml, which is why 141 near-identical config files had to
exist at exactly the right paths. Here identity is declared (mllm, backbone,
problem, sigma, track) and every legacy path is regenerated from it, so the
existing log directories and checkpoint filenames keep resolving unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mllmsent.labels import num_classes

FINETUNING_TREE = "experiments-finetuning"
NOT_FINETUNING_TREE = "experiments-not-finetuning"

RULE_TREE_IS_FINETUNING = "tree_is_finetuning"
RULE_TREE_NOT_FROZEN = "tree_not_frozen"


def resolve_fine_tuning(tree: str, rule: str) -> str:
    if rule == RULE_TREE_IS_FINETUNING:
        return "finetuned" if tree == FINETUNING_TREE else "not_finetuned"
    return "not_finetuned" if tree == NOT_FINETUNING_TREE else "finetuned"


@dataclass(frozen=True, slots=True)
class MllmProfile:
    name: str
    dataset_dir: str
    checkpoint_token: str
    dir_prefix: str
    caption_dir: str | None = None


@dataclass(frozen=True, slots=True)
class BackboneProfile:
    name: str
    model_name: str
    model_path: str
    dir_token: str
    dir_kind: str | None
    checkpoint_token: str
    finetuning_rule: str
    max_len: int | None = None
    epochs: int = 100
    unfrozen_patience: int = 10
    frozen_patience: int = 25
    modality: str = "text"


@dataclass(frozen=True, slots=True)
class Hyperparameters:
    learning_rate: float
    batch_size: int
    epochs: int
    max_len: int | None
    patience: int


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    mllm: str
    backbone: str
    problem: str
    sigma: int | None
    track: str
    tree: str
    objective: str
    validation: str
    freeze_backbone: bool
    hyperparameters: Hyperparameters
    mllm_profile: MllmProfile
    backbone_profile: BackboneProfile
    dir_override: str | None = None
    dir_kind_override: str | None = None
    checkpoint_alias: str | None = None
    checkpoint_dir_override: str | None = None
    dataset_override: str | None = None

    @property
    def sigma_token(self) -> str:
        return "none" if self.sigma is None else str(self.sigma)

    @property
    def id(self) -> str:
        return f"{self.mllm}-{self.backbone}-{self.problem}-sigma{self.sigma_token}"

    @property
    def qualified_id(self) -> str:
        return f"{self.track}/{self.id}"

    @property
    def dir_name(self) -> str:
        if self.dir_override:
            return self.dir_override
        kind = self.dir_kind_override or self.backbone_profile.dir_kind
        tokens = (
            self.mllm_profile.dir_prefix,
            self.backbone_profile.dir_token,
            kind,
            self.problem,
            None if self.sigma is None else f"alpha{self.sigma}",
        )
        return "-".join(token for token in tokens if token)

    @property
    def log_dir(self) -> str:
        return f"{self.tree}/{self.dir_name}/logs"

    @property
    def fine_tuning(self) -> str:
        return resolve_fine_tuning(self.tree, self.backbone_profile.finetuning_rule)

    @property
    def trains_qlora_adapter(self) -> bool:
        return self.backbone == "llama3-qlora" and self.fine_tuning == "finetuned"

    @property
    def num_classes(self) -> int:
        return num_classes(self.problem)

    @property
    def uses_twitter_validation(self) -> bool:
        return self.validation == "twitter_captions"

    @property
    def writes_per_fold_checkpoints(self) -> bool:
        return self.objective == "regression"

    @property
    def checkpoint_name(self) -> str:
        return (
            f"best_checkpoint_{self.mllm_profile.checkpoint_token}"
            f"_{self.backbone_profile.checkpoint_token}"
            f"_{self.problem}_sigma{self.sigma_token}_{self.fine_tuning}.pt"
        )

    def checkpoint_dir(self, root: Path) -> Path:
        if self.checkpoint_dir_override:
            return Path(self.checkpoint_dir_override)
        return Path(root)

    def fold_checkpoint(self, root: Path, fold: int) -> Path:
        return self.checkpoint_dir(root) / f"best_model_fold{fold}.pt"

    def resolve_checkpoint(self, root: Path) -> Path:
        directory = self.checkpoint_dir(root)
        canonical = directory / self.checkpoint_name
        if canonical.exists() or not self.checkpoint_alias:
            return canonical
        alias = directory / self.checkpoint_alias
        return alias if alias.exists() else canonical

    def dataset_csv(self, data_root: Path) -> Path:
        if self.dataset_override:
            return Path(data_root) / self.dataset_override
        return (
            Path(data_root)
            / self.mllm_profile.dataset_dir
            / f"percept_dataset_alpha{self.sigma}_{self.problem}.csv"
        )

    def twitter_validation_csv(self, captions_root: Path) -> Path:
        return Path(captions_root) / f"captions_alpha{self.sigma}_limited.csv"
