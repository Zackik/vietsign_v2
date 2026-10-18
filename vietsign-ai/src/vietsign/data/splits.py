"""Signer-independent split utilities."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Literal

from vietsign.data.schema import SignSample

GroupField = Literal["signer_id", "camera", "view_id", "domain_id"]


def make_signer_independent_splits(
    samples: list[SignSample],
    *,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> list[SignSample]:
    """Assign splits by signer so each signer belongs to exactly one split."""

    split_ratios = ratios or {"train": 0.70, "validation": 0.15, "test": 0.15}
    if any(sample.signer_id is None for sample in samples):
        raise ValueError("Signer-independent splitting requires signer_id for every sample.")
    if set(split_ratios) != {"train", "validation", "test"}:
        raise ValueError("ratios must contain exactly train, validation, and test.")

    total_ratio = sum(split_ratios.values())
    if total_ratio <= 0:
        raise ValueError("split ratios must sum to a positive value.")
    normalized_ratios = {name: value / total_ratio for name, value in split_ratios.items()}

    signer_to_samples: dict[str, list[SignSample]] = defaultdict(list)
    for sample in samples:
        signer_to_samples[sample.signer_id or ""].append(sample)

    signer_ids = sorted(signer_to_samples)
    rng = random.Random(seed)
    rng.shuffle(signer_ids)

    total_samples = len(samples)
    targets = {
        split: total_samples * normalized_ratios[split]
        for split in ("train", "validation", "test")
    }
    split_counts = {"train": 0, "validation": 0, "test": 0}
    signer_assignments: dict[str, str] = {}

    for signer_id in signer_ids:
        sample_count = len(signer_to_samples[signer_id])
        split = min(
            ("train", "validation", "test"),
            key=lambda candidate: split_counts[candidate] / max(targets[candidate], 1e-9),
        )
        signer_assignments[signer_id] = split
        split_counts[split] += sample_count

    result: list[SignSample] = []
    for sample in samples:
        result.append(sample.model_copy(update={"split": signer_assignments[sample.signer_id or ""]}))
    return result

def make_group_independent_splits(
    samples: list[SignSample],
    *,
    group_field: GroupField,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> list[SignSample]:
    """Assign splits so every group value appears in exactly one split."""

    split_ratios = ratios or {"train": 0.70, "validation": 0.15, "test": 0.15}
    if set(split_ratios) != {"train", "validation", "test"}:
        raise ValueError("ratios must contain exactly train, validation, and test.")
    if any(getattr(sample, group_field) in (None, "") for sample in samples):
        raise ValueError(f"Group-independent splitting requires {group_field} for every sample.")

    total_ratio = sum(split_ratios.values())
    if total_ratio <= 0:
        raise ValueError("split ratios must sum to a positive value.")
    normalized_ratios = {name: value / total_ratio for name, value in split_ratios.items()}

    group_to_samples: dict[str, list[SignSample]] = defaultdict(list)
    for sample in samples:
        group_to_samples[str(getattr(sample, group_field))].append(sample)

    group_ids = sorted(group_to_samples)
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    total_samples = len(samples)
    targets = {
        split: total_samples * normalized_ratios[split]
        for split in ("train", "validation", "test")
    }
    split_counts = {"train": 0, "validation": 0, "test": 0}
    group_assignments: dict[str, str] = {}

    for group_id in group_ids:
        sample_count = len(group_to_samples[group_id])
        split = min(
            ("train", "validation", "test"),
            key=lambda candidate: split_counts[candidate] / max(targets[candidate], 1e-9),
        )
        group_assignments[group_id] = split
        split_counts[split] += sample_count

    return [
        sample.model_copy(update={"split": group_assignments[str(getattr(sample, group_field))]})
        for sample in samples
    ]
