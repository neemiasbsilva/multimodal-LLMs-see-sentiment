from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

# Sentiment schema definitions for each problem variant.
# Each schema has:
#   simple_sentiment : maps raw 5-class labels to the problem's categories
#   sentiment_idx    : positional index used only for ia_calculation (legacy; kept for compat)
#   sentiment_values : integer label assigned to each category
SENTIMENT_SCHEMAS: dict = {
    "p5": {
        "simple_sentiment": {
            "Positive": "Positive",
            "SlightlyPositive": "SlightlyPositive",
            "Neutral": "Neutral",
            "SlightlyNegative": "SlightlyNegative",
            "Negative": "Negative",
        },
        "sentiment_idx": {
            "Positive": 4,
            "SlightlyPositive": 3,
            "Neutral": 2,
            "SlightlyNegative": 1,
            "Negative": 0,
        },
        "sentiment_values": {
            "Negative": 0,
            "SlightlyNegative": 1,
            "Neutral": 2,
            "SlightlyPositive": 3,
            "Positive": 4,
        },
    },
    "p3": {
        "simple_sentiment": {
            "Positive": "Positive",
            "SlightlyPositive": "Positive",
            "Neutral": "Neutral",
            "SlightlyNegative": "Negative",
            "Negative": "Negative",
        },
        "sentiment_idx": {
            "Positive": 2,
            "SlightlyPositive": 2,
            "Neutral": 0,
            "SlightlyNegative": 1,
            "Negative": 1,
        },
        "sentiment_values": {"Negative": 1, "Neutral": 0, "Positive": 2},
    },
    "p2plus": {
        # Neutral → Positive
        "simple_sentiment": {
            "Positive": "Positive",
            "SlightlyPositive": "Positive",
            "Neutral": "Positive",
            "SlightlyNegative": "Negative",
            "Negative": "Negative",
        },
        "sentiment_idx": {
            "Positive": 0,
            "SlightlyPositive": 0,
            "Neutral": 0,
            "SlightlyNegative": 1,
            "Negative": 1,
        },
        "sentiment_values": {"Negative": 1, "Positive": 0},
    },
    "p2neg": {
        # Neutral → Negative
        "simple_sentiment": {
            "Positive": "Positive",
            "SlightlyPositive": "Positive",
            "Neutral": "Negative",
            "SlightlyNegative": "Negative",
            "Negative": "Negative",
        },
        "sentiment_idx": {
            "Positive": 1,
            "SlightlyPositive": 1,
            "Neutral": 0,
            "SlightlyNegative": 0,
            "Negative": 0,
        },
        "sentiment_values": {"Negative": 0, "Positive": 1},
    },
}


def ia_calculation(sg: list) -> float:
    """Image Agreement: fraction of annotators who chose the majority label."""
    return max(sg) / len(sg)


def build_percept_dataset(
    df_captions: pd.DataFrame,
    data_json: dict,
    schema_name: str,
    freq_threshold: int,
    caption_col: str = "text",
) -> pd.DataFrame:
    """Build a classification dataset from captions + annotator vote data.

    Parameters
    ----------
    df_captions     : DataFrame with 'id' and caption_col columns.
    data_json       : Loaded dataset.json dict (keys: 'tasks').
    schema_name     : One of 'p5', 'p3', 'p2plus', 'p2neg'.
    freq_threshold  : Minimum majority-vote count to include a sample
                      (alpha3=3, alpha4=4, alpha5=5).
    caption_col     : Name of the caption column in df_captions.

    Returns
    -------
    DataFrame with columns: id, text, sentiment.
    """
    schema = SENTIMENT_SCHEMAS[schema_name]
    simple_sentiment = schema["simple_sentiment"]
    sentiment_values = schema["sentiment_values"]

    sentiment_data: dict = {}
    for id_, caption in tqdm(
        zip(df_captions["id"], df_captions[caption_col]),
        total=len(df_captions),
        desc=f"Init [{schema_name}]",
        leave=False,
    ):
        sentiment_data[id_] = {"cluster": [], "caption": caption}

    for samples in tqdm(data_json["tasks"], desc=f"Votes [{schema_name}]", leave=False):
        for sample in samples["images"]:
            id_ = sample["id"]
            if id_ in sentiment_data:
                mapped = simple_sentiment[sample["sentiment"]]
                sentiment_data[id_]["cluster"].append(mapped)

    rows: dict = {"id": [], "text": [], "sentiment": []}
    for id_, info in tqdm(
        sentiment_data.items(), desc=f"Filter α{freq_threshold}", leave=False
    ):
        counter = Counter(info["cluster"])
        if not counter:
            continue
        most_common, freq = counter.most_common(1)[0]
        if freq >= freq_threshold:
            rows["id"].append(id_)
            rows["text"].append(info["caption"])
            rows["sentiment"].append(sentiment_values[most_common])

    return pd.DataFrame(rows)


def build_regression_dataset(
    df_captions: pd.DataFrame,
    data_json: dict,
    schema_name: str,
    caption_col: str = "text",
) -> pd.DataFrame:
    """Build a regression target dataset (mean of numeric votes per image).

    Parameters
    ----------
    df_captions : DataFrame with 'id' and caption_col columns.
    data_json   : Loaded dataset.json dict.
    schema_name : One of 'p5', 'p3', 'p2plus', 'p2neg'.
    caption_col : Name of the caption column.

    Returns
    -------
    DataFrame with columns: id, text, sentiment_score.
    """
    schema = SENTIMENT_SCHEMAS[schema_name]
    simple_sentiment = schema["simple_sentiment"]
    sentiment_values = schema["sentiment_values"]

    sentiment_data: dict = {}
    for id_, caption in zip(df_captions["id"], df_captions[caption_col]):
        sentiment_data[id_] = {"votes": [], "caption": caption}

    for samples in tqdm(data_json["tasks"], desc=f"Votes [{schema_name}]", leave=False):
        for sample in samples["images"]:
            id_ = sample["id"]
            if id_ in sentiment_data:
                mapped = simple_sentiment[sample["sentiment"]]
                sentiment_data[id_]["votes"].append(sentiment_values[mapped])

    rows: dict = {"id": [], "text": [], "sentiment_score": []}
    for id_, info in sentiment_data.items():
        if info["votes"]:
            rows["id"].append(id_)
            rows["text"].append(info["caption"])
            rows["sentiment_score"].append(float(np.mean(info["votes"])))

    return pd.DataFrame(rows)
