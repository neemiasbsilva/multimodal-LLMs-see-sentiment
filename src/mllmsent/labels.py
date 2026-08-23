"""Single source of truth for the sentiment label encodings.

`SENTIMENT_SCHEMAS[problem]["sentiment_values"]` is the encoding the training
CSVs were built with, so every id2label, class count and plot legend in the
project derives from it rather than restating it.
"""

from __future__ import annotations

SENTIMENT_SCHEMAS: dict[str, dict[str, dict]] = {
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

PROBLEMS = tuple(SENTIMENT_SCHEMAS)

DISPLAY_NAMES = {
    "Negative": "Negative",
    "SlightlyNegative": "Slightly negative",
    "Neutral": "Neutral",
    "SlightlyPositive": "Slightly positive",
    "Positive": "Positive",
}

PROMPT_LABELS = {
    "p5": ["Positive", "Slightly positive", "Neutral", "Slightly negative", "Negative"],
    "p3": ["Positive", "Neutral", "Negative"],
    "p2": ["Positive", "Negative"],
    "p2plus": ["Positive", "Negative"],
    "p2neg": ["Positive", "Negative"],
}


def label2id(problem: str) -> dict[str, int]:
    return dict(SENTIMENT_SCHEMAS[problem]["sentiment_values"])


def id2label(problem: str) -> dict[int, str]:
    return {index: name for name, index in label2id(problem).items()}


def display_id2label(problem: str) -> dict[int, str]:
    return {index: DISPLAY_NAMES[name] for index, name in id2label(problem).items()}


def ordered_label_names(problem: str) -> list[str]:
    mapping = display_id2label(problem)
    return [mapping[index] for index in sorted(mapping)]


def num_classes(problem: str) -> int:
    return len(SENTIMENT_SCHEMAS[problem]["sentiment_values"])
