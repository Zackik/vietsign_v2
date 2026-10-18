"""Continuous sign recognition data and CTC training helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from vietsign.continuous.vocabulary import GlossVocabulary
from vietsign.data.schema import ContinuousSignSample
from vietsign.preprocessing.cache import cache_paths
from vietsign.training.config import RecognitionTrainingConfig
from vietsign.training.input import load_npz_safely, prepare_keypoint_arrays


class ContinuousDatasetError(RuntimeError):
    """Raised when a continuous sample cannot be loaded safely."""


def load_continuous_samples(metadata_csv: str | Path, *, dataset: str = "continuous") -> list[ContinuousSignSample]:
    path = Path(metadata_csv)
    if not path.exists():
        raise ContinuousDatasetError(f"Continuous metadata CSV not found: {path}")
    samples = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            glosses = [token for token in row["glosses"].replace("|", " ").split() if token]
            samples.append(
                ContinuousSignSample(
                    sample_id=row["sample_id"],
                    video_path=Path(row["video_path"]),
                    glosses=glosses,
                    split=row["split"],
                    dataset=row.get("dataset") or dataset,
                    signer_id=row.get("signer_id") or None,
                    sentence_id=row.get("sentence_id") or None,
                    translation=row.get("translation") or None,
                )
            )
    return samples


class ContinuousKeypointDataset(Dataset):
    def __init__(
        self,
        samples: list[ContinuousSignSample],
        *,
        cache_root: str | Path,
        vocabulary: GlossVocabulary,
        config: RecognitionTrainingConfig,
    ) -> None:
        self.samples = list(samples)
        self.cache_root = Path(cache_root)
        self.vocabulary = vocabulary
        self.config = config

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        npz_path, metadata_path = cache_paths(self.cache_root, sample)  # type: ignore[arg-type]
        if not npz_path.exists() or not metadata_path.exists():
            raise ContinuousDatasetError(f"Missing keypoint cache for continuous sample {sample.sample_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("sample_id") and metadata["sample_id"] != sample.sample_id:
            raise ContinuousDatasetError(f"Cache sample mismatch for {sample.sample_id}")
        arrays = load_npz_safely(npz_path)
        inputs, valid_mask, temporal_mask, valid_ratio = prepare_keypoint_arrays(
            arrays,
            feature_config=self.config.features,
            sequence_config=self.config.sequence,
            minimum_valid_joint_ratio=self.config.data_quality.minimum_valid_joint_ratio,
            reject_empty=self.config.data_quality.reject_empty,
        )
        target = torch.tensor(self.vocabulary.encode(sample.glosses), dtype=torch.long)
        if target.numel() == 0:
            raise ContinuousDatasetError(f"Empty gloss target for {sample.sample_id}")
        if int(temporal_mask.sum()) < target.numel():
            raise ContinuousDatasetError(
                f"CTC impossible length for {sample.sample_id}: input={int(temporal_mask.sum())}, target={target.numel()}"
            )
        return {
            "sample_id": sample.sample_id,
            "signer_id": sample.signer_id,
            "coordinates": torch.from_numpy(inputs),
            "valid_mask": torch.from_numpy(valid_mask),
            "temporal_mask": torch.from_numpy(temporal_mask),
            "target": target,
            "glosses": sample.glosses,
            "valid_joint_ratio": valid_ratio,
        }


def collate_continuous_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate empty continuous batch.")
    batch_size = len(samples)
    max_t = max(sample["coordinates"].shape[0] for sample in samples)
    joints = samples[0]["coordinates"].shape[1]
    channels = samples[0]["coordinates"].shape[2]
    coordinates = torch.zeros((batch_size, max_t, joints, channels), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, max_t, joints), dtype=torch.bool)
    temporal_mask = torch.zeros((batch_size, max_t), dtype=torch.bool)
    targets = []
    target_lengths = []
    for index, sample in enumerate(samples):
        frames = sample["coordinates"].shape[0]
        coordinates[index, :frames] = sample["coordinates"]
        valid_mask[index, :frames] = sample["valid_mask"]
        temporal_mask[index, :frames] = sample["temporal_mask"]
        targets.append(sample["target"])
        target_lengths.append(int(sample["target"].numel()))
    return {
        "coordinates": coordinates,
        "valid_mask": valid_mask,
        "temporal_mask": temporal_mask,
        "input_lengths": temporal_mask.long().sum(dim=1),
        "targets": torch.cat(targets),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "reference_glosses": [sample["glosses"] for sample in samples],
        "sample_ids": [sample["sample_id"] for sample in samples],
        "signer_ids": [sample["signer_id"] for sample in samples],
    }
