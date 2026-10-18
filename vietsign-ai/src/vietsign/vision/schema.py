"""Backend-independent keypoint schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vietsign.vision.constants import (
    CANONICAL_JOINT_NAMES,
    HAND_JOINT_NAMES,
    LEFT_HAND_SLICE,
    POSE_JOINT_NAMES,
    POSE_SLICE,
    RIGHT_HAND_SLICE,
    SCHEMA_VERSION,
)

MISSING_SOURCE = 0
ORIGINAL_SOURCE = 1
INTERPOLATED_SOURCE = 2


@dataclass(frozen=True)
class Keypoint:
    """One canonical keypoint.

    Confidence is backend-provided when available. If a backend detects a
    landmark but does not expose per-landmark confidence, confidence is `None`.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    confidence: float | None = None
    valid: bool = False

    @classmethod
    def missing(cls) -> Keypoint:
        return cls()


@dataclass(frozen=True)
class FrameKeypoints:
    """Canonical keypoints for one sampled video frame."""

    frame_index: int
    timestamp: float
    left_hand: tuple[Keypoint, ...] = field(default_factory=tuple)
    right_hand: tuple[Keypoint, ...] = field(default_factory=tuple)
    pose: tuple[Keypoint, ...] = field(default_factory=tuple)
    face: tuple[Keypoint, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "left_hand", _pad_keypoints(self.left_hand, len(HAND_JOINT_NAMES)))
        object.__setattr__(self, "right_hand", _pad_keypoints(self.right_hand, len(HAND_JOINT_NAMES)))
        object.__setattr__(self, "pose", _pad_keypoints(self.pose, len(POSE_JOINT_NAMES)))


@dataclass(frozen=True)
class KeypointSequence:
    """Temporal keypoint representation independent of extraction backend."""

    frames: tuple[FrameKeypoints, ...]
    fps: float | None
    source_video: Path
    extractor: str
    schema_version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_tensors(self) -> KeypointTensors:
        """Convert to model-ready tensors.

        Contract:
        - coordinates: [T, J, 3], float32
        - valid_mask: [T, J], bool
        - confidence: [T, J], float32, NaN when unknown or missing
        - source_mask: [T, J], uint8 where 0=missing, 1=original, 2=interpolated
        - J = 55 by default: 21 left hand + 21 right hand + 13 upper-body pose joints.
        """

        frame_count = len(self.frames)
        joint_count = len(CANONICAL_JOINT_NAMES)
        coordinates = np.zeros((frame_count, joint_count, 3), dtype=np.float32)
        valid_mask = np.zeros((frame_count, joint_count), dtype=np.bool_)
        confidence = np.full((frame_count, joint_count), np.nan, dtype=np.float32)
        source_mask = np.zeros((frame_count, joint_count), dtype=np.uint8)
        timestamps = np.zeros((frame_count,), dtype=np.float32)
        frame_indices = np.zeros((frame_count,), dtype=np.int64)

        for frame_idx, frame in enumerate(self.frames):
            timestamps[frame_idx] = frame.timestamp
            frame_indices[frame_idx] = frame.frame_index
            _write_region(frame.left_hand, LEFT_HAND_SLICE, frame_idx, coordinates, valid_mask, confidence)
            _write_region(
                frame.right_hand, RIGHT_HAND_SLICE, frame_idx, coordinates, valid_mask, confidence
            )
            _write_region(frame.pose, POSE_SLICE, frame_idx, coordinates, valid_mask, confidence)

        source_mask[valid_mask] = ORIGINAL_SOURCE
        return KeypointTensors(
            coordinates=coordinates,
            valid_mask=valid_mask,
            confidence=confidence,
            source_mask=source_mask,
            timestamps=timestamps,
            frame_indices=frame_indices,
            joint_names=CANONICAL_JOINT_NAMES,
        )


@dataclass(frozen=True)
class KeypointTensors:
    """Model-ready keypoint tensors."""

    coordinates: np.ndarray
    valid_mask: np.ndarray
    confidence: np.ndarray
    source_mask: np.ndarray
    timestamps: np.ndarray
    frame_indices: np.ndarray
    joint_names: tuple[str, ...]


def _pad_keypoints(keypoints: tuple[Keypoint, ...], expected: int) -> tuple[Keypoint, ...]:
    values = tuple(keypoints[:expected])
    if len(values) == expected:
        return values
    return values + tuple(Keypoint.missing() for _ in range(expected - len(values)))


def _write_region(
    keypoints: tuple[Keypoint, ...],
    region: slice,
    frame_idx: int,
    coordinates: np.ndarray,
    valid_mask: np.ndarray,
    confidence: np.ndarray,
) -> None:
    for offset, keypoint in enumerate(keypoints):
        joint_idx = region.start + offset
        if not keypoint.valid:
            continue
        coordinates[frame_idx, joint_idx] = (keypoint.x, keypoint.y, keypoint.z)
        valid_mask[frame_idx, joint_idx] = True
        if keypoint.confidence is not None:
            confidence[frame_idx, joint_idx] = keypoint.confidence
