"""Extraction quality, stability, and smoothing utilities."""

from __future__ import annotations

from statistics import mean, median

import numpy as np

from vietsign.vision.processing import detection_coverage
from vietsign.vision.schema import KeypointTensors


def overall_valid_joint_ratio(tensors: KeypointTensors) -> float:
    return float(tensors.valid_mask.sum() / max(tensors.valid_mask.size, 1))


def empty_frame_ratio(tensors: KeypointTensors) -> float:
    if tensors.valid_mask.shape[0] == 0:
        return 1.0
    return float((~np.any(tensors.valid_mask, axis=1)).sum() / tensors.valid_mask.shape[0])


def interpolation_ratio(tensors: KeypointTensors) -> float:
    return float((tensors.source_mask == 2).sum() / max(tensors.source_mask.size, 1))


def sequence_quality(tensors: KeypointTensors) -> dict[str, float | bool]:
    coverage = detection_coverage(tensors)
    return {
        **coverage,
        "overall_valid_joint_ratio": overall_valid_joint_ratio(tensors),
        "empty_frame_ratio": empty_frame_ratio(tensors),
        "empty_sequence": bool(tensors.valid_mask.sum() == 0),
        "interpolation_ratio": interpolation_ratio(tensors),
        "large_coordinate_jump_count": float(count_large_coordinate_jumps(tensors)),
    }


def aggregate_quality(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    metrics = [
        "left_hand_coverage",
        "right_hand_coverage",
        "pose_coverage",
        "overall_valid_joint_ratio",
        "empty_frame_ratio",
        "interpolation_ratio",
    ]
    output: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = sorted(float(row.get(metric, 0.0)) for row in rows)
        if not values:
            output[metric] = {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
            continue
        output[metric] = {
            "mean": mean(values),
            "median": median(values),
            "p10": _percentile(values, 10),
            "p90": _percentile(values, 90),
        }
    return output


def count_large_coordinate_jumps(tensors: KeypointTensors, *, threshold: float = 0.35) -> int:
    if tensors.coordinates.shape[0] < 2:
        return 0
    valid_pairs = tensors.valid_mask[1:] & tensors.valid_mask[:-1]
    deltas = np.linalg.norm(tensors.coordinates[1:] - tensors.coordinates[:-1], axis=-1)
    return int(((deltas > threshold) & valid_pairs).sum())


def smooth_ema(tensors: KeypointTensors, *, alpha: float = 0.6) -> KeypointTensors:
    """Conservative EMA smoothing that respects missing masks."""

    coordinates = tensors.coordinates.copy()
    for joint_idx in range(coordinates.shape[1]):
        previous: np.ndarray | None = None
        for frame_idx in range(coordinates.shape[0]):
            if not tensors.valid_mask[frame_idx, joint_idx]:
                previous = None
                continue
            current = coordinates[frame_idx, joint_idx].copy()
            if previous is not None:
                coordinates[frame_idx, joint_idx] = alpha * current + (1.0 - alpha) * previous
            previous = coordinates[frame_idx, joint_idx].copy()
    return KeypointTensors(
        coordinates=coordinates,
        valid_mask=tensors.valid_mask,
        confidence=tensors.confidence,
        source_mask=tensors.source_mask,
        timestamps=tensors.timestamps,
        frame_indices=tensors.frame_indices,
        joint_names=tensors.joint_names,
    )


def _percentile(values: list[float], percentile: int) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight

