"""RGB video dataset, preprocessing, and cache helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import numpy as np
import torch
from torch.utils.data import Dataset

from vietsign.data.schema import SignSample
from vietsign.data.video import inspect_video
from vietsign.training.config import (
    RecognitionTrainingConfig,
    RGBPreprocessingConfig,
    RGBVideoConfig,
)

RGB_CACHE_SCHEMA_VERSION = "rgb_embeddings.v1"


@dataclass(frozen=True)
class RGBSample:
    sample_id: str
    signer_id: str | None
    label: int
    label_name: str
    pixel_values: torch.Tensor
    temporal_mask: torch.Tensor
    metadata: dict[str, Any]


class RGBVideoDataset(Dataset[RGBSample]):
    """Lazy RGB video dataset over existing SignSample metadata."""

    def __init__(
        self,
        samples: list[SignSample],
        *,
        label_to_id: dict[str, int],
        config: RecognitionTrainingConfig,
        training: bool,
    ) -> None:
        self.samples = list(samples)
        self.label_to_id = dict(label_to_id)
        self.config = config
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> RGBSample:
        sample = self.samples[index]
        frames, metadata = load_rgb_clip(
            sample.video_path,
            video_config=self.config.video,
            preprocessing=self.config.rgb,
            training=self.training,
        )
        return RGBSample(
            sample_id=sample.sample_id,
            signer_id=sample.signer_id,
            label=self.label_to_id[sample.label],
            label_name=sample.label,
            pixel_values=torch.from_numpy(frames),
            temporal_mask=torch.ones((frames.shape[0],), dtype=torch.bool),
            metadata=metadata,
        )


def collate_rgb_batch(samples: list[RGBSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty RGB batch.")
    return {
        "pixel_values": torch.stack([sample.pixel_values for sample in samples], dim=0),
        "temporal_mask": torch.stack([sample.temporal_mask for sample in samples], dim=0),
        "labels": torch.tensor([sample.label for sample in samples], dtype=torch.long),
        "sample_ids": [sample.sample_id for sample in samples],
        "signer_ids": [sample.signer_id for sample in samples],
        "metadata": [sample.metadata for sample in samples],
    }


def load_rgb_clip(
    video_path: str | Path,
    *,
    video_config: RGBVideoConfig,
    preprocessing: RGBPreprocessingConfig,
    training: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata = inspect_video(video_path)
    if not metadata.readable or metadata.frame_count is None:
        raise ValueError(f"Unable to read video {video_path}: {metadata.error}")
    indices = select_frame_indices(
        metadata.frame_count,
        num_frames=video_config.num_frames,
        sampling=video_config.sampling,
        training=training,
    )
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame_bgr = capture.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(preprocess_rgb_frame(frame_rgb, preprocessing=preprocessing, training=training))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames could be decoded from {video_path}")
    while len(frames) < video_config.num_frames:
        frames.append(frames[-1].copy())
    tensor = np.stack(frames[: video_config.num_frames], axis=0).astype(np.float32)
    timestamps = [
        (float(index) / metadata.fps) if metadata.fps and metadata.fps > 0 else float(offset)
        for offset, index in enumerate(indices.tolist())
    ]
    return tensor, {
        "video": metadata.model_dump(mode="json"),
        "selected_frame_indices": [int(index) for index in indices.tolist()],
        "sampled_timestamps": timestamps,
        "input_resolution": [preprocessing.crop_size, preprocessing.crop_size],
        "tensor_shape": list(tensor.shape),
        "horizontal_flip_enabled": preprocessing.horizontal_flip,
    }


def select_frame_indices(
    frame_count: int,
    *,
    num_frames: int,
    sampling: str,
    training: bool,
) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if sampling == "random" and training and frame_count > num_frames:
        start = int(np.random.randint(0, frame_count - num_frames + 1))
        return np.arange(start, start + num_frames, dtype=np.int64)
    if sampling not in {"uniform", "random"}:
        raise ValueError(f"Unsupported RGB sampling strategy: {sampling}")
    return np.linspace(0, frame_count - 1, num=num_frames, dtype=np.int64)


def preprocess_rgb_frame(
    frame_rgb: np.ndarray,
    *,
    preprocessing: RGBPreprocessingConfig,
    training: bool,
) -> np.ndarray:
    resized = cv2.resize(frame_rgb, (preprocessing.resize, preprocessing.resize), interpolation=cv2.INTER_AREA)
    cropped = _crop(resized, crop_size=preprocessing.crop_size, random_crop=training and preprocessing.random_crop)
    if training and preprocessing.horizontal_flip:
        cropped = cv2.flip(cropped, 1)
    values = cropped.astype(np.float32) / 255.0
    mean = np.asarray(preprocessing.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(preprocessing.std, dtype=np.float32).reshape(1, 1, 3)
    normalized = (values - mean) / std
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32)


def rgb_cache_fingerprint(config: RecognitionTrainingConfig) -> str:
    payload = {
        "schema_version": RGB_CACHE_SCHEMA_VERSION,
        "model": {
            "type": config.model.type,
            "pretrained_model": config.model.pretrained_model,
            "pretrained": config.model.pretrained,
            "freeze_backbone": config.model.freeze_backbone,
            "hidden_dim": config.model.hidden_dim,
        },
        "video": config.video.model_dump(mode="json"),
        "rgb": config.rgb.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rgb_cache_paths(root: str | Path, sample: SignSample) -> tuple[Path, Path]:
    base = Path(root) / sample.dataset / sample.sample_id
    return base.with_suffix(".npz"), base.with_suffix(".json")


def save_rgb_embedding_cache(
    npz_path: Path,
    metadata_path: Path,
    *,
    embedding: np.ndarray,
    sample: SignSample,
    metadata: dict[str, Any],
) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, embedding=embedding.astype(np.float32))
    payload = {
        "schema_version": RGB_CACHE_SCHEMA_VERSION,
        "sample_id": sample.sample_id,
        "label": sample.label,
        "label_id": sample.label_id,
        "signer_id": sample.signer_id,
        "dataset": sample.dataset,
        **metadata,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def load_rgb_embedding_cache(
    npz_path: Path,
    metadata_path: Path,
    *,
    expected_fingerprint: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != RGB_CACHE_SCHEMA_VERSION:
        raise ValueError(f"RGB cache schema mismatch: {metadata.get('schema_version')}")
    if expected_fingerprint is not None and metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError("RGB cache fingerprint mismatch.")
    with np.load(npz_path, allow_pickle=False) as data:
        return data["embedding"], metadata


def _crop(frame: np.ndarray, *, crop_size: int, random_crop: bool) -> np.ndarray:
    height, width = frame.shape[:2]
    if height < crop_size or width < crop_size:
        frame = cv2.resize(frame, (max(width, crop_size), max(height, crop_size)))
        height, width = frame.shape[:2]
    if random_crop:
        top = int(np.random.randint(0, height - crop_size + 1))
        left = int(np.random.randint(0, width - crop_size + 1))
    else:
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
    return frame[top : top + crop_size, left : left + crop_size]
