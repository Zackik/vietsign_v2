"""Domain-aware evaluation helpers for multi-camera sign recognition."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class DomainMetric:
    """Metrics for one signer/view/camera/domain bucket."""

    domain: str
    support: int
    top1_accuracy: float
    top5_accuracy: float


@dataclass(frozen=True)
class DomainMetricsReport:
    """Cross-domain recognition report."""

    group_name: str
    groups: list[DomainMetric] = field(default_factory=list)

    @property
    def worst_top1_accuracy(self) -> float | None:
        if not self.groups:
            return None
        return min(group.top1_accuracy for group in self.groups)

    @property
    def macro_top1_accuracy(self) -> float | None:
        if not self.groups:
            return None
        return sum(group.top1_accuracy for group in self.groups) / len(self.groups)


def compute_domain_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    domains: list[str | None],
    *,
    group_name: str,
    top_k: int = 5,
) -> DomainMetricsReport:
    """Compute top-k accuracy per domain bucket.

    `domains` must align with the batch dimension of `logits`/`labels`. Missing
    values are grouped under `"unknown"` so a report can still be produced while
    highlighting incomplete metadata.
    """

    if logits.shape[0] != labels.shape[0] or logits.shape[0] != len(domains):
        raise ValueError("logits, labels, and domains must have the same number of samples.")
    if logits.numel() == 0:
        raise ValueError("Cannot compute domain metrics on empty logits.")

    predictions = logits.argmax(dim=-1).detach().cpu()
    labels_cpu = labels.detach().cpu()
    topn = min(top_k, logits.shape[-1])
    top_indices = logits.topk(topn, dim=-1).indices.detach().cpu()
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, domain in enumerate(domains):
        grouped[(domain or "unknown").strip() or "unknown"].append(index)

    metrics = []
    for domain, indices in sorted(grouped.items()):
        correct1 = sum(int(predictions[index] == labels_cpu[index]) for index in indices)
        correctk = sum(int(labels_cpu[index].item() in top_indices[index].tolist()) for index in indices)
        support = len(indices)
        metrics.append(
            DomainMetric(
                domain=domain,
                support=support,
                top1_accuracy=correct1 / support,
                top5_accuracy=correctk / support,
            )
        )
    return DomainMetricsReport(group_name=group_name, groups=metrics)
