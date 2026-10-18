"""Model input preparation for cached keypoint tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vietsign.training.config import InputFeatureConfig, SequenceConfig
from vietsign.vision.constants import LEFT_HAND_SLICE, POSE_SLICE, RIGHT_HAND_SLICE


class EmptySequenceError(ValueError):
    """Raised when a sequence has no usable landmark detections."""


@dataclass(frozen=True)
class PreparedSample:
    sample_id: str
    label: int
    label_name: str
    inputs: torch.Tensor
    valid_mask: torch.Tensor
    temporal_mask: torch.Tensor
    valid_joint_ratio: float
    metadata: dict[str, Any]


def selected_joint_indices(features: InputFeatureConfig) -> np.ndarray:
    indices: list[int] = []
    if features.left_hand:
        indices.extend(range(LEFT_HAND_SLICE.start, LEFT_HAND_SLICE.stop))
    if features.right_hand:
        indices.extend(range(RIGHT_HAND_SLICE.start, RIGHT_HAND_SLICE.stop))
    if features.pose:
        indices.extend(range(POSE_SLICE.start, POSE_SLICE.stop))
    if not indices:
        raise ValueError("At least one feature region must be enabled.")
    return np.asarray(indices, dtype=np.int64)


def prepare_keypoint_arrays(
    arrays: dict[str, np.ndarray],
    *,
    feature_config: InputFeatureConfig,
    sequence_config: SequenceConfig,
    minimum_valid_joint_ratio: float = 0.0,
    reject_empty: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Prepare one cached sequence for model input.

    Returns:
    - inputs: [T, J_selected, C]
    - valid_mask: [T, J_selected]
    - temporal_mask: [T]
    - valid_joint_ratio
    """

    coordinates = _required(arrays, "coordinates").astype(np.float32)
    valid_mask = _required(arrays, "valid_mask").astype(bool)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(f"coordinates must have shape [T,J,3], got {coordinates.shape}")
    if valid_mask.shape != coordinates.shape[:2]:
        raise ValueError("valid_mask shape must match coordinates [T,J]")

    joint_indices = selected_joint_indices(feature_config)
    coordinates = coordinates[:, joint_indices, :]
    valid_mask = valid_mask[:, joint_indices]
    valid_ratio = float(valid_mask.sum() / max(valid_mask.size, 1))
    if reject_empty and valid_ratio <= 0.0:
        raise EmptySequenceError("Sequence has no valid keypoints.")
    if valid_ratio < minimum_valid_joint_ratio:
        raise EmptySequenceError(
            f"Sequence valid joint ratio {valid_ratio:.4f} is below minimum {minimum_valid_joint_ratio:.4f}."
        )

    coordinates = coordinates * valid_mask[..., None].astype(np.float32)
    channels = [coordinates]
    if feature_config.velocity:
        velocity = arrays.get("velocity")
        if velocity is None:
            velocity = np.zeros_like(_required(arrays, "coordinates"), dtype=np.float32)
        velocity = velocity.astype(np.float32)[:, joint_indices, :] * valid_mask[..., None].astype(np.float32)
        channels.append(velocity)
    if feature_config.confidence:
        confidence = arrays.get("confidence")
        if confidence is None:
            confidence = np.full(_required(arrays, "coordinates").shape[:2], np.nan, dtype=np.float32)
        confidence = np.nan_to_num(confidence.astype(np.float32)[:, joint_indices], nan=0.0)
        channels.append(confidence[..., None])
    if feature_config.validity:
        channels.append(valid_mask.astype(np.float32)[..., None])
    if feature_config.source_mask:
        source_mask = arrays.get("source_mask")
        if source_mask is None:
            source_mask = np.zeros(_required(arrays, "coordinates").shape[:2], dtype=np.float32)
        channels.append((source_mask.astype(np.float32)[:, joint_indices] / 2.0)[..., None])

    inputs = np.concatenate(channels, axis=-1).astype(np.float32)
    inputs, valid_mask, temporal_mask = apply_sequence_mode(
        inputs, valid_mask, sequence_config=sequence_config
    )
    if not np.isfinite(inputs).all():
        raise ValueError("Prepared keypoint inputs contain NaN or infinity.")
    return inputs, valid_mask, temporal_mask, valid_ratio


def apply_sequence_mode(
    inputs: np.ndarray,
    valid_mask: np.ndarray,
    *,
    sequence_config: SequenceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = inputs.shape[0]
    temporal_mask = np.ones((length,), dtype=bool)
    if sequence_config.mode == "variable" or sequence_config.length is None:
        return inputs, valid_mask, temporal_mask
    target = sequence_config.length
    if length == target:
        return inputs, valid_mask, temporal_mask
    if length <= 0:
        raise EmptySequenceError("Sequence has no frames.")
    indices = np.linspace(0, length - 1, num=target, dtype=np.int64)
    return inputs[indices], valid_mask[indices], np.ones((target,), dtype=bool)


def collate_keypoint_batch(samples: list[PreparedSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    batch_size = len(samples)
    max_t = max(sample.inputs.shape[0] for sample in samples)
    joint_count = samples[0].inputs.shape[1]
    channel_count = samples[0].inputs.shape[2]

    coordinates = torch.zeros((batch_size, max_t, joint_count, channel_count), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, max_t, joint_count), dtype=torch.bool)
    temporal_mask = torch.zeros((batch_size, max_t), dtype=torch.bool)
    labels = torch.zeros((batch_size,), dtype=torch.long)
    sample_ids: list[str] = []

    for index, sample in enumerate(samples):
        frames = sample.inputs.shape[0]
        coordinates[index, :frames] = sample.inputs
        valid_mask[index, :frames] = sample.valid_mask
        temporal_mask[index, :frames] = sample.temporal_mask
        labels[index] = sample.label
        sample_ids.append(sample.sample_id)

    return {
        "coordinates": coordinates,
        "valid_mask": valid_mask,
        "temporal_mask": temporal_mask,
        "labels": labels,
        "sample_ids": sample_ids,
    }


def load_npz_safely(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _required(arrays: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"Cached feature file missing required array: {name}")
    return arrays[name]

