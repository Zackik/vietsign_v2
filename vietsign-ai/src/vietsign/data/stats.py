"""Dataset statistics for normalized samples."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from vietsign.data.schema import SignSample


class NumericSummary(BaseModel):
    """Simple numeric summary."""

    count: int = 0
    min: float | None = None
    max: float | None = None
    mean: float | None = None


class DatasetStatistics(BaseModel):
    """Machine-readable dataset statistics."""

    total_samples: int
    num_labels: int
    num_signers: int
    num_cameras: int
    num_views: int
    num_domains: int
    samples_per_split: dict[str, int] = Field(default_factory=dict)
    samples_per_class: dict[str, int] = Field(default_factory=dict)
    samples_per_signer: dict[str, int] = Field(default_factory=dict)
    samples_per_camera: dict[str, int] = Field(default_factory=dict)
    samples_per_view: dict[str, int] = Field(default_factory=dict)
    samples_per_domain: dict[str, int] = Field(default_factory=dict)
    duration: NumericSummary
    fps: NumericSummary
    resolutions: dict[str, int] = Field(default_factory=dict)
    class_imbalance: dict[str, Any] = Field(default_factory=dict)


def compute_dataset_statistics(samples: list[SignSample]) -> DatasetStatistics:
    """Compute summary statistics for normalized samples."""

    labels = Counter(sample.label for sample in samples)
    signer_ids = [sample.signer_id for sample in samples if sample.signer_id]
    signers = Counter(signer_ids)
    cameras = Counter(sample.camera for sample in samples if sample.camera)
    views = Counter(sample.view_id for sample in samples if sample.view_id)
    domains = Counter(sample.domain_id for sample in samples if sample.domain_id)

    return DatasetStatistics(
        total_samples=len(samples),
        num_labels=len(labels),
        num_signers=len(signers),
        num_cameras=len(cameras),
        num_views=len(views),
        num_domains=len(domains),
        samples_per_split=dict(sorted(Counter(sample.split for sample in samples).items())),
        samples_per_class=dict(sorted(labels.items())),
        samples_per_signer=dict(sorted(signers.items())),
        samples_per_camera=dict(sorted(cameras.items())),
        samples_per_view=dict(sorted(views.items())),
        samples_per_domain=dict(sorted(domains.items())),
        duration=_numeric_summary(sample.duration for sample in samples),
        fps=_numeric_summary(sample.fps for sample in samples),
        resolutions=dict(
            sorted(
                Counter(
                    f"{sample.width}x{sample.height}"
                    for sample in samples
                    if sample.width is not None and sample.height is not None
                ).items()
            )
        ),
        class_imbalance=_class_imbalance(labels),
    )


def _numeric_summary(values: object) -> NumericSummary:
    collected = [float(value) for value in values if value is not None]  # type: ignore[union-attr]
    if not collected:
        return NumericSummary()
    return NumericSummary(
        count=len(collected),
        min=min(collected),
        max=max(collected),
        mean=mean(collected),
    )


def _class_imbalance(labels: Counter[str]) -> dict[str, Any]:
    if not labels:
        return {"ratio": None, "minority_classes": [], "majority_classes": []}
    min_count = min(labels.values())
    max_count = max(labels.values())
    return {
        "ratio": (max_count / min_count) if min_count else None,
        "minority_classes": sorted(label for label, count in labels.items() if count == min_count),
        "majority_classes": sorted(label for label, count in labels.items() if count == max_count),
    }
