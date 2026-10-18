"""Multimodal RGB + keypoint dataset and prediction helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from vietsign.data.schema import SignSample
from vietsign.preprocessing.cache import cache_paths
from vietsign.training.config import RecognitionTrainingConfig
from vietsign.training.input import load_npz_safely, prepare_keypoint_arrays
from vietsign.training.rgb import load_rgb_clip


class MultimodalDatasetError(RuntimeError):
    """Raised when RGB and keypoint modalities cannot be safely aligned."""


@dataclass(frozen=True)
class MultimodalSample:
    sample_id: str
    signer_id: str | None
    label: int
    label_name: str
    pixel_values: torch.Tensor
    coordinates: torch.Tensor
    valid_mask: torch.Tensor
    temporal_mask: torch.Tensor
    modality_mask: dict[str, torch.Tensor]
    metadata: dict[str, Any]


class MultimodalVSLDataset(Dataset[MultimodalSample]):
    """Dataset that aligns RGB video and cached keypoints by sample_id."""

    def __init__(
        self,
        samples: list[SignSample],
        *,
        cache_root: str | Path,
        label_to_id: dict[str, int],
        config: RecognitionTrainingConfig,
        training: bool,
        require_rgb: bool = True,
        require_pose: bool = True,
    ) -> None:
        self.samples_by_id = {sample.sample_id: sample for sample in samples}
        if len(self.samples_by_id) != len(samples):
            raise MultimodalDatasetError("Duplicate sample_id detected in multimodal dataset.")
        self.samples = list(samples)
        self.cache_root = Path(cache_root)
        self.label_to_id = dict(label_to_id)
        self.config = config
        self.training = training
        self.require_rgb = require_rgb
        self.require_pose = require_pose

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> MultimodalSample:
        sample = self.samples[index]
        pixel_values, rgb_metadata = self._load_rgb(sample)
        coordinates, valid_mask, temporal_mask, valid_ratio, pose_metadata = self._load_pose(sample)
        _validate_alignment(sample, pose_metadata)
        return MultimodalSample(
            sample_id=sample.sample_id,
            signer_id=sample.signer_id,
            label=self.label_to_id[sample.label],
            label_name=sample.label,
            pixel_values=torch.from_numpy(pixel_values),
            coordinates=torch.from_numpy(coordinates),
            valid_mask=torch.from_numpy(valid_mask),
            temporal_mask=torch.from_numpy(temporal_mask),
            modality_mask={
                "rgb": torch.tensor(True),
                "pose": torch.tensor(valid_ratio > 0.0),
            },
            metadata={"rgb": rgb_metadata, "pose": pose_metadata},
        )

    def _load_rgb(self, sample: SignSample):
        try:
            return load_rgb_clip(
                sample.video_path,
                video_config=self.config.video,
                preprocessing=self.config.rgb,
                training=self.training,
            )
        except Exception as error:
            if self.require_rgb:
                raise MultimodalDatasetError(f"RGB load failed for {sample.sample_id}: {error}") from error
            raise

    def _load_pose(self, sample: SignSample):
        npz_path, metadata_path = cache_paths(self.cache_root, sample)
        if (not npz_path.exists() or not metadata_path.exists()) and self.require_pose:
            raise MultimodalDatasetError(f"Missing keypoint cache for {sample.sample_id}: {npz_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        arrays = load_npz_safely(npz_path)
        inputs, valid_mask, temporal_mask, valid_ratio = prepare_keypoint_arrays(
            arrays,
            feature_config=self.config.features,
            sequence_config=self.config.sequence,
            minimum_valid_joint_ratio=self.config.data_quality.minimum_valid_joint_ratio,
            reject_empty=self.config.data_quality.reject_empty,
        )
        return inputs, valid_mask, temporal_mask, valid_ratio, metadata


def collate_multimodal_batch(samples: list[MultimodalSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty multimodal batch.")
    max_pose_t = max(sample.coordinates.shape[0] for sample in samples)
    joint_count = samples[0].coordinates.shape[1]
    channels = samples[0].coordinates.shape[2]
    batch_size = len(samples)
    coordinates = torch.zeros((batch_size, max_pose_t, joint_count, channels), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, max_pose_t, joint_count), dtype=torch.bool)
    temporal_mask = torch.zeros((batch_size, max_pose_t), dtype=torch.bool)
    for index, sample in enumerate(samples):
        frames = sample.coordinates.shape[0]
        coordinates[index, :frames] = sample.coordinates
        valid_mask[index, :frames] = sample.valid_mask
        temporal_mask[index, :frames] = sample.temporal_mask
    return {
        "pixel_values": torch.stack([sample.pixel_values for sample in samples], dim=0),
        "coordinates": coordinates,
        "valid_mask": valid_mask,
        "temporal_mask": temporal_mask,
        "labels": torch.tensor([sample.label for sample in samples], dtype=torch.long),
        "sample_ids": [sample.sample_id for sample in samples],
        "signer_ids": [sample.signer_id for sample in samples],
        "modality_mask": {
            "rgb": torch.stack([sample.modality_mask["rgb"] for sample in samples]),
            "pose": torch.stack([sample.modality_mask["pose"] for sample in samples]),
        },
        "metadata": [sample.metadata for sample in samples],
    }


def load_keypoint_arrays_for_prediction(
    keypoints_path: str | Path,
    metadata_path: str | Path,
    config: RecognitionTrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    arrays = load_npz_safely(keypoints_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    inputs, valid_mask, temporal_mask, _ratio = prepare_keypoint_arrays(
        arrays,
        feature_config=config.features,
        sequence_config=config.sequence,
        minimum_valid_joint_ratio=config.data_quality.minimum_valid_joint_ratio,
        reject_empty=config.data_quality.reject_empty,
    )
    return (
        torch.from_numpy(inputs).unsqueeze(0),
        torch.from_numpy(valid_mask).unsqueeze(0),
        torch.from_numpy(temporal_mask).unsqueeze(0),
        metadata,
    )


def _validate_alignment(sample: SignSample, pose_metadata: dict[str, Any]) -> None:
    if pose_metadata.get("sample_id") and pose_metadata["sample_id"] != sample.sample_id:
        raise MultimodalDatasetError(
            f"sample_id mismatch: RGB/sample={sample.sample_id}, pose={pose_metadata['sample_id']}"
        )
    if pose_metadata.get("label") and pose_metadata["label"] != sample.label:
        raise MultimodalDatasetError(
            f"label mismatch for {sample.sample_id}: RGB/sample={sample.label}, pose={pose_metadata['label']}"
        )
    source_video = pose_metadata.get("source_video")
    if source_video and Path(source_video).name != Path(sample.video_path).name:
        raise MultimodalDatasetError(
            f"source video mismatch for {sample.sample_id}: {sample.video_path} vs {source_video}"
        )
