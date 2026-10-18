"""Recognition metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MetricsResult:
    loss: float | None
    top1_accuracy: float
    top5_accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_accuracy: dict[str, float | None]
    confusion_matrix: list[list[int]]


def topk_predictions(
    logits: torch.Tensor,
    *,
    id_to_label: dict[int, str],
    k: int = 5,
) -> list[list[dict[str, float | int | str]]]:
    probabilities = torch.softmax(logits, dim=-1)
    top_k = min(k, logits.shape[-1])
    values, indices = probabilities.topk(top_k, dim=-1)
    output = []
    for row_values, row_indices in zip(values, indices):
        output.append(
            [
                {
                    "label": id_to_label[int(index)],
                    "label_id": int(index),
                    "confidence": float(value),
                }
                for value, index in zip(row_values, row_indices)
            ]
        )
    return output


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_names: list[str],
    loss: float | None = None,
    top_k: int = 5,
) -> MetricsResult:
    if logits.numel() == 0:
        raise ValueError("Cannot compute metrics on empty logits.")
    predictions = logits.argmax(dim=-1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, predicted in zip(labels_np, predictions):
        confusion[int(actual), int(predicted)] += 1

    top1 = float((predictions == labels_np).mean()) if len(labels_np) else 0.0
    topn = min(top_k, num_classes)
    top_indices = logits.topk(topn, dim=-1).indices.detach().cpu().numpy()
    top5 = float(np.mean([label in row for label, row in zip(labels_np, top_indices)]))
    per_class_accuracy = {}
    f1_scores = []
    weighted_sum = 0.0
    total_support = 0
    for class_id, class_name in enumerate(class_names):
        tp = int(confusion[class_id, class_id])
        support = int(confusion[class_id].sum())
        predicted_count = int(confusion[:, class_id].sum())
        accuracy = (tp / support) if support else None
        per_class_accuracy[class_name] = accuracy
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        f1_scores.append(f1)
        weighted_sum += f1 * support
        total_support += support
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    weighted_f1 = float(weighted_sum / total_support) if total_support else 0.0
    return MetricsResult(
        loss=loss,
        top1_accuracy=top1,
        top5_accuracy=top5,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_accuracy=per_class_accuracy,
        confusion_matrix=confusion.tolist(),
    )

