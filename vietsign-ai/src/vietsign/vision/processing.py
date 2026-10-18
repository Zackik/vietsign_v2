"""Keypoint normalization, interpolation, motion, and coverage."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from vietsign.vision.constants import (
    CANONICAL_JOINT_INDEX,
    LEFT_HAND_SLICE,
    POSE_SLICE,
    RIGHT_HAND_SLICE,
)
from vietsign.vision.schema import INTERPOLATED_SOURCE, KeypointTensors


def normalize_tensors(
    tensors: KeypointTensors,
    *,
    method: str = "image",
    image_size: tuple[int, int] | None = None,
    min_scale: float = 1e-6,
) -> KeypointTensors:
    """Normalize coordinates using image or body-relative strategy."""

    coordinates = tensors.coordinates.copy()
    if method == "none":
        return _copy_with_coordinates(tensors, coordinates)
    if method == "image":
        if image_size is not None:
            width, height = image_size
            if width > 0:
                coordinates[..., 0] = coordinates[..., 0] / width
            if height > 0:
                coordinates[..., 1] = coordinates[..., 1] / height
        return _copy_with_coordinates(tensors, coordinates)
    if method != "body_relative":
        raise ValueError("normalization method must be one of: none, image, body_relative")

    left_idx = CANONICAL_JOINT_INDEX["pose:left_shoulder"]
    right_idx = CANONICAL_JOINT_INDEX["pose:right_shoulder"]
    normalized = coordinates.copy()
    for frame_idx in range(coordinates.shape[0]):
        if not (tensors.valid_mask[frame_idx, left_idx] and tensors.valid_mask[frame_idx, right_idx]):
            continue
        left = coordinates[frame_idx, left_idx, :3]
        right = coordinates[frame_idx, right_idx, :3]
        origin = (left + right) / 2.0
        scale = float(np.linalg.norm(left[:2] - right[:2]))
        if scale <= min_scale:
            continue
        normalized[frame_idx] = (coordinates[frame_idx] - origin) / scale
    return _copy_with_coordinates(tensors, normalized)


def interpolate_short_gaps(tensors: KeypointTensors, *, max_gap: int) -> KeypointTensors:
    """Linearly interpolate short missing gaps per joint."""

    coordinates = tensors.coordinates.copy()
    valid_mask = tensors.valid_mask.copy()
    source_mask = tensors.source_mask.copy()
    if max_gap <= 0:
        return _copy_tensors(tensors, coordinates=coordinates, valid_mask=valid_mask, source_mask=source_mask)

    _, joint_count, _ = coordinates.shape
    for joint_idx in range(joint_count):
        valid_indices = np.flatnonzero(valid_mask[:, joint_idx])
        if len(valid_indices) < 2:
            continue
        for start, end in pairwise(valid_indices):
            gap = int(end - start - 1)
            if gap <= 0 or gap > max_gap:
                continue
            for offset, frame_idx in enumerate(range(start + 1, end), start=1):
                ratio = offset / (gap + 1)
                coordinates[frame_idx, joint_idx] = (
                    coordinates[start, joint_idx] * (1.0 - ratio)
                    + coordinates[end, joint_idx] * ratio
                )
                valid_mask[frame_idx, joint_idx] = True
                source_mask[frame_idx, joint_idx] = INTERPOLATED_SOURCE
    return _copy_tensors(tensors, coordinates=coordinates, valid_mask=valid_mask, source_mask=source_mask)


def compute_velocity(tensors: KeypointTensors) -> np.ndarray:
    """Compute per-joint velocity [T, J, 3] without crossing missing detections."""

    velocity = np.zeros_like(tensors.coordinates)
    for frame_idx in range(1, tensors.coordinates.shape[0]):
        both_valid = tensors.valid_mask[frame_idx] & tensors.valid_mask[frame_idx - 1]
        velocity[frame_idx, both_valid] = (
            tensors.coordinates[frame_idx, both_valid] - tensors.coordinates[frame_idx - 1, both_valid]
        )
    return velocity


def compute_acceleration(velocity: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Compute per-joint acceleration [T, J, 3] without crossing invalid velocity spans."""

    acceleration = np.zeros_like(velocity)
    for frame_idx in range(2, velocity.shape[0]):
        valid = valid_mask[frame_idx] & valid_mask[frame_idx - 1] & valid_mask[frame_idx - 2]
        acceleration[frame_idx, valid] = velocity[frame_idx, valid] - velocity[frame_idx - 1, valid]
    return acceleration


def detection_coverage(tensors: KeypointTensors) -> dict[str, float]:
    """Calculate region-level frame coverage from validity masks."""

    total = max(tensors.valid_mask.shape[0], 1)
    return {
        "left_hand_coverage": float(np.any(tensors.valid_mask[:, LEFT_HAND_SLICE], axis=1).sum() / total),
        "right_hand_coverage": float(np.any(tensors.valid_mask[:, RIGHT_HAND_SLICE], axis=1).sum() / total),
        "pose_coverage": float(np.any(tensors.valid_mask[:, POSE_SLICE], axis=1).sum() / total),
    }


def _copy_with_coordinates(tensors: KeypointTensors, coordinates: np.ndarray) -> KeypointTensors:
    return _copy_tensors(tensors, coordinates=coordinates)


def _copy_tensors(
    tensors: KeypointTensors,
    *,
    coordinates: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    source_mask: np.ndarray | None = None,
) -> KeypointTensors:
    return KeypointTensors(
        coordinates=coordinates if coordinates is not None else tensors.coordinates,
        valid_mask=valid_mask if valid_mask is not None else tensors.valid_mask,
        confidence=tensors.confidence,
        source_mask=source_mask if source_mask is not None else tensors.source_mask,
        timestamps=tensors.timestamps,
        frame_indices=tensors.frame_indices,
        joint_names=tensors.joint_names,
    )
