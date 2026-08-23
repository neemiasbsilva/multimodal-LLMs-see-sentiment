"""The one place an experiment's training frame is loaded.

Replaces the two near-identical copies of load_dataframe/load_experiment_data
that used to live in utils/data_loader.py and scripts/utils_dl.py, which
different call sites imported inconsistently.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mllmsent.experiments.spec import ExperimentSpec

TWITTER_P2PLUS_REMAP = {0: 1, 1: 0}


def load_experiment_frame(spec: ExperimentSpec, data_root: Path) -> pd.DataFrame:
    path = spec.dataset_csv(data_root)
    if not path.is_file():
        raise SystemExit(f"dataset not found for {spec.qualified_id}: {path}")
    frame = pd.read_csv(path)
    print(f"loaded {len(frame)} rows from {path}")
    return frame


def load_twitter_validation(spec: ExperimentSpec, captions_root: Path) -> pd.DataFrame:
    path = spec.twitter_validation_csv(captions_root)
    if not path.is_file():
        raise SystemExit(
            f"twitter validation set not found: {path}\n"
            "Set TWITTER_CAPTIONS_DIR or paths.twitter_captions_root in experiments.yaml."
        )
    frame = pd.read_csv(path)[["text", "sentiment"]]
    if spec.problem == "p2plus":
        frame["sentiment"] = frame["sentiment"].map(
            lambda value: TWITTER_P2PLUS_REMAP.get(value, value)
        )
    return frame


def fold_frame(frame: pd.DataFrame, index, modality: str = "text") -> pd.DataFrame:
    # ImageDataset reads row["id"] and row["sentiment"]; SentimentDataset reads
    # row["text"] and row["sentiment"].
    columns = ["id", "sentiment"] if modality == "image" else ["text", "sentiment"]
    return pd.DataFrame(
        {column: frame[column].iloc[index].to_list() for column in columns}
    )
