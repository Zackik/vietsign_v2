"""Dataset and split audit utilities."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vietsign.data.adapters import DatasetLoadError, create_adapter
from vietsign.data.config import load_dataset_config
from vietsign.data.schema import SignSample


def audit_dataset_config(config_path: str | Path) -> dict[str, Any]:
    try:
        dataset_config = load_dataset_config(config_path)
        samples = create_adapter(dataset_config).load_samples()
    except DatasetLoadError as error:
        return {"status": "unavailable", "config": str(config_path), "error": str(error)}
    return audit_samples(samples, dataset=str(dataset_config.dataset.name), config=str(config_path))


def audit_samples(samples: list[SignSample], *, dataset: str, config: str | None = None) -> dict[str, Any]:
    paths = [sample.video_path for sample in samples]
    labels = [sample.label for sample in samples]
    signers_by_split: dict[str, set[str]] = defaultdict(set)
    missing_files = []
    hashes = []
    for sample in samples:
        if sample.signer_id:
            signers_by_split[sample.split].add(sample.signer_id)
        if not sample.video_path.exists():
            missing_files.append(sample.sample_id)
        elif sample.video_path.is_file():
            hashes.append((sample.sample_id, _sha256(sample.video_path)))
    duplicate_paths = [str(path) for path, count in Counter(paths).items() if count > 1]
    duplicate_hashes = [
        {"sha256": sha, "sample_ids": [sample_id for sample_id, item_sha in hashes if item_sha == sha]}
        for sha, count in Counter(sha for _sample_id, sha in hashes).items()
        if count > 1
    ]
    signer_intersections = _signer_intersections(signers_by_split)
    return {
        "status": "audited",
        "config": config,
        "dataset": dataset,
        "samples": len(samples),
        "missing_files": missing_files,
        "duplicate_paths": duplicate_paths,
        "duplicate_hashes": duplicate_hashes,
        "labels": {"count": len(set(labels)), "distribution": dict(Counter(labels))},
        "splits": dict(Counter(sample.split for sample in samples)),
        "signers_by_split": {split: sorted(signers) for split, signers in signers_by_split.items()},
        "signer_intersections": signer_intersections,
        "signer_independent": all(count == 0 for count in signer_intersections.values()),
    }


def _signer_intersections(signers_by_split: dict[str, set[str]]) -> dict[str, int]:
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    return {
        f"{left}_{right}": len(signers_by_split.get(left, set()) & signers_by_split.get(right, set()))
        for left, right in pairs
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
