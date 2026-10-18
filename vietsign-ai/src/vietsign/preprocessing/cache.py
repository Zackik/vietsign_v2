"""Safe keypoint cache IO."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from vietsign.data.schema import SignSample
from vietsign.preprocessing.config import PreprocessingConfig
from vietsign.vision.assets import resolve_repo_path, sha256_file
from vietsign.vision.constants import SCHEMA_VERSION
from vietsign.vision.schema import KeypointTensors


class CacheMismatchError(RuntimeError):
    """Raised when cached features were produced by different settings."""


def preprocessing_fingerprint(config: PreprocessingConfig) -> str:
    """Create a deterministic fingerprint from preprocessing settings."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "extractor": _extractor_fingerprint(config),
        "input": config.input.model_dump(mode="json"),
        "video": config.video.model_dump(mode="json"),
        "normalization": config.normalization.model_dump(mode="json"),
        "interpolation": config.interpolation.model_dump(mode="json"),
        "smoothing": config.smoothing.model_dump(mode="json"),
        "features": config.features.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _extractor_fingerprint(config: PreprocessingConfig) -> dict[str, Any]:
    payload = config.extractor.model_dump(mode="json")
    for key in ("hand_model", "pose_model", "face_model", "rtmpose_config", "checkpoint"):
        value = payload.get(key)
        if value:
            path = resolve_repo_path(value)
            payload[f"{key}_resolved"] = str(path)
            payload[f"{key}_sha256"] = sha256_file(path) if path.exists() and path.stat().st_size > 0 else None
    if config.extractor.type == "rtmpose":
        payload["rtmpose_runtime"] = config.rtmpose.model_dump(mode="json")
        for key in ("det_model", "pose_model"):
            value = getattr(config.rtmpose, key)
            if value:
                path = resolve_repo_path(value)
                payload[f"rtmpose_{key}_resolved"] = str(path)
                payload[f"rtmpose_{key}_sha256"] = (
                    sha256_file(path) if path.exists() and path.stat().st_size > 0 else None
                )
    return payload


def cache_paths(root: str | Path, sample: SignSample) -> tuple[Path, Path]:
    base = Path(root) / sample.dataset / sample.sample_id
    return base.with_suffix(".npz"), base.with_suffix(".json")


def save_keypoint_cache(
    npz_path: Path,
    metadata_path: Path,
    tensors: KeypointTensors,
    *,
    sample: SignSample,
    metadata: dict[str, Any],
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    """Save tensors as NPZ plus JSON metadata. No pickle is used."""

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "coordinates": tensors.coordinates,
        "valid_mask": tensors.valid_mask,
        "confidence": tensors.confidence,
        "source_mask": tensors.source_mask,
        "timestamps": tensors.timestamps,
        "frame_indices": tensors.frame_indices,
        "joint_names": np.asarray(tensors.joint_names),
    }
    if extra_arrays:
        arrays.update(extra_arrays)
    np.savez_compressed(npz_path, **arrays)
    metadata_path.write_text(
        json.dumps(
            {
                "sample_id": sample.sample_id,
                "label": sample.label,
                "label_id": sample.label_id,
                "signer_id": sample.signer_id,
                "dataset": sample.dataset,
                "source_video": str(sample.video_path),
                "schema_version": SCHEMA_VERSION,
                **metadata,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_keypoint_cache(
    npz_path: Path,
    metadata_path: Path,
    *,
    expected_fingerprint: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not npz_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing cache files: {npz_path}, {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if expected_fingerprint is not None and metadata.get("fingerprint") != expected_fingerprint:
        raise CacheMismatchError(
            f"Cache fingerprint mismatch for {npz_path}: "
            f"expected {expected_fingerprint}, found {metadata.get('fingerprint')}"
        )
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    return arrays, metadata
