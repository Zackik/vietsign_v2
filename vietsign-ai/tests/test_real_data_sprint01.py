from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vietsign.data.sources import load_source_registry, source_status_matrix
from vietsign.data.voya import VOYAMappingError, audit_voya_directory, convert_voya_to_vietsign55
from vietsign.translation.parallel_corpus import load_parallel_vie_vsl_samples


def test_data_source_registry_contains_required_sources() -> None:
    registry = load_source_registry("configs/data/sources.yaml")
    names = {source.name for source in registry.sources}
    matrix = source_status_matrix(registry)

    assert {"VOYA_VSL", "VSL400", "Parallel-Corpus-Vie-VSL", "ViSL-News"}.issubset(names)
    assert any(row["dataset"] == "VOYA_VSL" and row["keypoints"] is True for row in matrix)
    assert any(row["dataset"] == "ViSL-News" and row["gloss_sequence"] is False for row in matrix)


def test_voya_audit_reads_local_npz_and_labels(tmp_path: Path) -> None:
    root = tmp_path / "voya_vsl"
    merged = root / "Merged"
    merged.mkdir(parents=True)
    np.savez_compressed(
        merged / "class_0001.npz",
        sequences=np.zeros((2, 60, 1605), dtype=np.float32),
        labels=np.array([0, 0], dtype=np.int32),
    )
    (root / "labels.json").write_text(json.dumps({"class_0001": "xin chào"}), encoding="utf-8")

    report = audit_voya_directory(root)

    assert report["available"] is True
    assert report["npz_file_count"] == 1
    assert report["labels_count"] == 1
    assert report["inspected"][0]["sequences_shape"] == [2, 60, 1605]


def test_voya_converter_refuses_unverified_mapping() -> None:
    with pytest.raises(VOYAMappingError):
        convert_voya_to_vietsign55(np.zeros((60, 1605), dtype=np.float32))


def test_parallel_corpus_adapter_preserves_provenance(tmp_path: Path) -> None:
    root = tmp_path / "parallel"
    root.mkdir()
    (root / "Vie10k.txt").write_text("Tôi uống nước.\nCảm ơn.\n", encoding="utf-8")
    (root / "VSL10k.txt").write_text("TOI UONG NUOC\nCAM-ON\n", encoding="utf-8")

    samples = load_parallel_vie_vsl_samples(root)

    assert len(samples) == 2
    assert samples[0].target_text == "Tôi uống nước."
    assert samples[0].glosses == ["TOI", "UONG", "NUOC"]
    assert samples[0].metadata["source_dataset"] == "Parallel-Corpus-Vie-VSL"


def test_parallel_corpus_missing_files_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Vie10k.txt"):
        load_parallel_vie_vsl_samples(tmp_path)
