"""Backbone construction, keyed by the matrix's backbone name."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class BackboneBuilder:
    build_model: Callable
    build_tokenizer: Callable
    needs_logits: bool = False


def _distilbert(model_path: str, num_classes: int):
    from mllmsent.models.heads import DistilBERTModel

    return DistilBERTModel(model_path, num_classes)


def _distilbert_tokenizer(model_path: str):
    from transformers import DistilBertTokenizer

    return DistilBertTokenizer.from_pretrained(model_path, do_lower_case=True)


def _modernbert(model_path: str, num_classes: int):
    from mllmsent.models.heads import ModernBERTModel

    return ModernBERTModel(model_path, num_classes)


def _auto_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path)


def _bart(model_path: str, num_classes: int):
    from transformers import BartForSequenceClassification

    return BartForSequenceClassification.from_pretrained(
        model_path, num_labels=num_classes, ignore_mismatched_sizes=True
    )


def _bart_tokenizer(model_path: str):
    from transformers import BartTokenizerFast

    return BartTokenizerFast.from_pretrained(model_path)


def _swin(model_path: str, num_classes: int):
    from transformers import AutoModelForImageClassification

    return AutoModelForImageClassification.from_pretrained(
        model_path, num_labels=num_classes, ignore_mismatched_sizes=True
    )


def _swin_processor(model_path: str):
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(model_path)


BUILDERS = {
    "distilbert": BackboneBuilder(_distilbert, _distilbert_tokenizer),
    "modernbert": BackboneBuilder(_modernbert, _auto_tokenizer),
    "bart": BackboneBuilder(_bart, _bart_tokenizer, needs_logits=True),
    "swin": BackboneBuilder(_swin, _swin_processor, needs_logits=True),
}


def get_builder(backbone: str) -> BackboneBuilder:
    if backbone not in BUILDERS:
        raise SystemExit(
            f"no builder for backbone '{backbone}'; "
            f"known: {', '.join(sorted(BUILDERS))}"
        )
    return BUILDERS[backbone]
