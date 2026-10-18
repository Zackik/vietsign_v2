"""Deterministic text metrics for translation software validation."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from vietsign.translation.normalization import normalize_vietnamese_text


def exact_match(reference: str, prediction: str) -> float:
    return float(normalize_vietnamese_text(reference) == normalize_vietnamese_text(prediction))


def sentence_bleu(reference: str, prediction: str, *, max_order: int = 4) -> float:
    ref_tokens = normalize_vietnamese_text(reference).split()
    pred_tokens = normalize_vietnamese_text(prediction).split()
    if not ref_tokens or not pred_tokens:
        return 0.0
    precisions = []
    for order in range(1, max_order + 1):
        ref_counts = _ngram_counts(ref_tokens, order)
        pred_counts = _ngram_counts(pred_tokens, order)
        if not pred_counts:
            precisions.append(1.0)
            continue
        overlap = sum(min(count, ref_counts.get(ngram, 0)) for ngram, count in pred_counts.items())
        precisions.append((overlap + 1.0) / (sum(pred_counts.values()) + 1.0))
    brevity = 1.0 if len(pred_tokens) > len(ref_tokens) else math.exp(1.0 - len(ref_tokens) / max(len(pred_tokens), 1))
    return brevity * math.exp(sum(math.log(value) for value in precisions) / max_order)


def chrf(reference: str, prediction: str, *, max_order: int = 6, beta: float = 2.0) -> float:
    reference = normalize_vietnamese_text(reference)
    prediction = normalize_vietnamese_text(prediction)
    if not reference or not prediction:
        return 0.0
    precisions = []
    recalls = []
    for order in range(1, max_order + 1):
        ref_counts = _char_ngram_counts(reference, order)
        pred_counts = _char_ngram_counts(prediction, order)
        overlap = sum(min(count, ref_counts.get(ngram, 0)) for ngram, count in pred_counts.items())
        precisions.append(overlap / max(sum(pred_counts.values()), 1))
        recalls.append(overlap / max(sum(ref_counts.values()), 1))
    precision = sum(precisions) / max_order
    recall = sum(recalls) / max_order
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / max(beta_sq * precision + recall, 1e-12)


def aggregate_translation_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    if not pairs:
        return {"bleu": 0.0, "chrf": 0.0, "exact_match": 0.0}
    return {
        "bleu": sum(sentence_bleu(ref, pred) for ref, pred in pairs) / len(pairs),
        "chrf": sum(chrf(ref, pred) for ref, pred in pairs) / len(pairs),
        "exact_match": sum(exact_match(ref, pred) for ref, pred in pairs) / len(pairs),
    }


def _ngram_counts(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def _char_ngram_counts(text: str, order: int) -> Counter[str]:
    compact = text.replace(" ", "")
    return Counter(compact[index : index + order] for index in range(len(compact) - order + 1))
