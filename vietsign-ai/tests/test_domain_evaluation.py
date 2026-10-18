from __future__ import annotations

import pytest
import torch

from vietsign.evaluation.domain import compute_domain_metrics


def test_domain_metrics_report_per_view_accuracy() -> None:
    logits = torch.tensor(
        [
            [4.0, 0.0],
            [0.0, 4.0],
            [4.0, 0.0],
            [4.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 1, 1, 0])

    report = compute_domain_metrics(
        logits,
        labels,
        ["front", "front", "side", "side"],
        group_name="view_id",
        top_k=1,
    )

    by_domain = {group.domain: group for group in report.groups}
    assert by_domain["front"].top1_accuracy == 1.0
    assert by_domain["side"].top1_accuracy == 0.5
    assert report.worst_top1_accuracy == 0.5
    assert report.macro_top1_accuracy == 0.75


def test_domain_metrics_requires_aligned_inputs() -> None:
    with pytest.raises(ValueError, match="same number of samples"):
        compute_domain_metrics(
            torch.zeros((2, 3)),
            torch.zeros((2,), dtype=torch.long),
            ["front"],
            group_name="view_id",
        )
