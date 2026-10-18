"""Training dataset over Phase 3 cached keypoint features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from vietsign.data.adapters import create_adapter
from vietsign.data.config import load_dataset_config
from vietsign.data.schema import SignSample
from vietsign.preprocessing.cache import CacheMismatchError, cache_paths
from vietsign.training.config import RecognitionTrainingConfig
from vietsign.training.input import PreparedSample, load_npz_safely, prepare_keypoint_arrays
from vietsign.vision.constants import SCHEMA_VERSION


class CachedKeypointDatasetError(RuntimeError):
    """Raised when cached keypoint features cannot be used for training."""


class CachedKeypointDataset(Dataset[PreparedSample]):
    """Dataset that reads SignSample metadata plus cached keypoint NPZ files."""

    def __init__(
        self,
        samples: list[SignSample],
        *,
        cache_root: str | Path,
        label_to_id: dict[str, int],
        config: RecognitionTrainingConfig,
    ) -> None:
        self.samples = list(samples)
        self.cache_root = Path(cache_root)
        self.label_to_id = dict(label_to_id)
        self.config = config

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PreparedSample:
        sample = self.samples[index]
        npz_path, metadata_path = cache_paths(self.cache_root, sample)
        if not npz_path.exists() or not metadata_path.exists():
            raise CachedKeypointDatasetError(
                f"Missing cached keypoints for sample '{sample.sample_id}': {npz_path}, {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _validate_cache_metadata(metadata, expected_fingerprint=self.config.dataset.expected_fingerprint)
        arrays = load_npz_safely(npz_path)
        try:
            inputs, valid_mask, temporal_mask, valid_ratio = prepare_keypoint_arrays(
                arrays,
                feature_config=self.config.features,
                sequence_config=self.config.sequence,
                minimum_valid_joint_ratio=self.config.data_quality.minimum_valid_joint_ratio,
                reject_empty=self.config.data_quality.reject_empty,
            )
        except ValueError as error:
            raise CachedKeypointDatasetError(
                f"Invalid cached keypoints for sample '{sample.sample_id}': {error}"
            ) from error

        import torch

        return PreparedSample(
            sample_id=sample.sample_id,
            label=self.label_to_id[sample.label],
            label_name=sample.label,
            inputs=torch.from_numpy(inputs),
            valid_mask=torch.from_numpy(valid_mask),
            temporal_mask=torch.from_numpy(temporal_mask),
            valid_joint_ratio=valid_ratio,
            metadata=metadata,
        )


def load_samples_for_training(config: RecognitionTrainingConfig) -> list[SignSample]:
    dataset_config = load_dataset_config(config.dataset.config)
    return create_adapter(dataset_config).load_samples()


def build_class_mapping(samples: list[SignSample]) -> dict[str, int]:
    labels = sorted({sample.label for sample in samples})
    return {label: index for index, label in enumerate(labels)}


def split_samples(samples: list[SignSample]) -> tuple[list[SignSample], list[SignSample], list[SignSample]]:
    train = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split in {"validation", "val"}]
    test = [sample for sample in samples if sample.split == "test"]
    return train, validation, test


def signer_overlap_report(samples: list[SignSample]) -> dict[str, Any]:
    split_to_signers: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        canonical = {"validation", "val"} if split == "validation" else {split}
        split_to_signers[split] = {
            sample.signer_id
            for sample in samples
            if sample.signer_id and sample.split in canonical
        }
    overlap = {
        "train_validation": sorted(split_to_signers["train"] & split_to_signers["validation"]),
        "train_test": sorted(split_to_signers["train"] & split_to_signers["test"]),
        "validation_test": sorted(split_to_signers["validation"] & split_to_signers["test"]),
    }
    return {
        "split_strategy": "metadata",
        "num_signers": len(set().union(*split_to_signers.values())),
        "overlap": overlap,
        "has_overlap": any(overlap.values()),
    }


def _validate_cache_metadata(metadata: dict[str, Any], *, expected_fingerprint: str | None) -> None:
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise CachedKeypointDatasetError(
            f"Cache schema mismatch: expected {SCHEMA_VERSION}, found {metadata.get('schema_version')}"
        )
    if expected_fingerprint is not None and metadata.get("fingerprint") != expected_fingerprint:
        raise CacheMismatchError(
            f"Cache fingerprint mismatch: expected {expected_fingerprint}, found {metadata.get('fingerprint')}"
        )

