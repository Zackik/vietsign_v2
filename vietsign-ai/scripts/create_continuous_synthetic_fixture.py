"""Create a tiny synthetic CSLR fixture for Phase 09 software validation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from vietsign.data.schema import SignSample
from vietsign.preprocessing.cache import save_keypoint_cache
from vietsign.vision.constants import CANONICAL_JOINT_NAMES, SCHEMA_VERSION
from vietsign.vision.schema import KeypointTensors

ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "data" / "metadata" / "continuous_synthetic.csv"
CACHE_ROOT = ROOT / "data" / "cache" / "keypoints"
DATASET = "continuous_synthetic"

SAMPLES = (
    ("cs_train_001", "TOI MUON NUOC", "signer_a", "train"),
    ("cs_train_002", "TOI TOI NUOC", "signer_b", "train"),
    ("cs_train_003", "BAN MUON GI", "signer_c", "train"),
    ("cs_val_001", "TOI MUON NUOC", "signer_d", "validation"),
    ("cs_val_002", "TOI TOI NUOC", "signer_e", "validation"),
)

GLOSS_VALUES = {
    "TOI": 0.15,
    "MUON": 0.35,
    "NUOC": 0.55,
    "BAN": 0.75,
    "GI": 0.95,
}


def main() -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_id", "video_path", "glosses", "signer_id", "split"])
        writer.writeheader()
        for sample_id, gloss_text, signer_id, split in SAMPLES:
            glosses = gloss_text.split()
            _write_cache(sample_id, glosses, signer_id, split)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "video_path": f"data/raw/synthetic/{sample_id}.mp4",
                    "glosses": gloss_text,
                    "signer_id": signer_id,
                    "split": split,
                }
            )


def _write_cache(sample_id: str, glosses: list[str], signer_id: str, split: str) -> None:
    tensors = _make_tensors(glosses)
    sample = SignSample(
        sample_id=sample_id,
        video_path=Path(f"data/raw/synthetic/{sample_id}.mp4"),
        label=" ".join(glosses),
        label_id=0,
        split=split,
        dataset=DATASET,
        signer_id=signer_id,
    )
    save_keypoint_cache(
        CACHE_ROOT / DATASET / f"{sample_id}.npz",
        CACHE_ROOT / DATASET / f"{sample_id}.json",
        tensors,
        sample=sample,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "fingerprint": None,
            "synthetic": True,
            "warning": "SYNTHETIC CSLR SOFTWARE VALIDATION - NOT VSL PERFORMANCE",
        },
    )


def _make_tensors(glosses: list[str]) -> KeypointTensors:
    frames_per_gloss = 5
    separator_frames = 1
    total_frames = len(glosses) * frames_per_gloss + max(0, len(glosses) - 1) * separator_frames
    coordinates = np.zeros((total_frames, 55, 3), dtype=np.float32)
    valid_mask = np.zeros((total_frames, 55), dtype=np.bool_)
    confidence = np.full((total_frames, 55), np.nan, dtype=np.float32)
    source_mask = np.zeros((total_frames, 55), dtype=np.uint8)

    frame = 0
    for gloss in glosses:
        value = GLOSS_VALUES[gloss]
        for local_frame in range(frames_per_gloss):
            phase = local_frame / max(frames_per_gloss - 1, 1)
            coordinates[frame, :, 0] = value + 0.01 * phase
            coordinates[frame, :, 1] = np.linspace(0.0, 1.0, 55, dtype=np.float32)
            coordinates[frame, :, 2] = value * 0.5
            valid_mask[frame, :] = True
            confidence[frame, :] = 0.9
            source_mask[frame, :] = 1
            frame += 1
        if frame < total_frames:
            frame += separator_frames

    if total_frames > 4:
        valid_mask[3, 0:21] = False
        confidence[3, 0:21] = np.nan
        source_mask[3, 0:21] = 0

    return KeypointTensors(
        coordinates=coordinates,
        valid_mask=valid_mask,
        confidence=confidence,
        source_mask=source_mask,
        timestamps=np.arange(total_frames, dtype=np.float32) / 25.0,
        frame_indices=np.arange(total_frames, dtype=np.int64),
        joint_names=CANONICAL_JOINT_NAMES,
    )


if __name__ == "__main__":
    main()
