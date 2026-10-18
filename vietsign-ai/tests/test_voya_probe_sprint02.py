from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vietsign.data import voya
from vietsign.data.voya import (
    MAPPING_STRONGLY_SUPPORTED,
    build_voya_probe_report,
    candidate_layout_report,
    direct_training_feasibility,
    fetch_voya_probe_files,
    inspect_voya_npz,
    mapping_confidence,
    normalize_class_name,
    validate_voya_labels,
    write_voya_probe_outputs,
)


def test_normalize_class_name_accepts_common_selectors() -> None:
    assert normalize_class_name(1) == "class_0001"
    assert normalize_class_name("0001") == "class_0001"
    assert normalize_class_name("class_0001") == "class_0001"
    assert normalize_class_name("class_0001.npz") == "class_0001"


def test_validate_voya_labels_reports_contiguous_class_keys(tmp_path: Path) -> None:
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps({"class_0001": "xin chào", "class_0002": "cảm ơn"}, ensure_ascii=False),
        encoding="utf-8",
    )

    report = validate_voya_labels(labels)

    assert report["mapping_count"] == 2
    assert report["class_key_format"] == "class_####"
    assert report["is_contiguous"] is True
    assert report["duplicate_labels"] == []


def test_candidate_layout_keeps_canonical_mapping_unverified() -> None:
    sequence = np.ones((60, 1605), dtype=np.float32)

    report = candidate_layout_report(sequence)

    assert report["mapping_status"] == MAPPING_STRONGLY_SUPPORTED
    assert report["canonical_conversion_allowed"] is False
    assert report["candidates"][0]["matches_feature_dim"] is True


def test_direct_training_feasible_for_source_layout_features() -> None:
    report = direct_training_feasibility(np.zeros((60, 1605), dtype=np.float32))

    assert report["status"] == "feasible"
    assert report["input_contract"] == "[T,1605] frame-level source-layout features"


def test_mapping_confidence_requires_authoritative_evidence_for_verified() -> None:
    candidates = [{"matches_feature_dim": True}]

    assert mapping_confidence(candidates, authoritative_evidence=False) == MAPPING_STRONGLY_SUPPORTED
    assert mapping_confidence(candidates, authoritative_evidence=True) == "VERIFIED"


def test_inspect_voya_npz_and_probe_report(tmp_path: Path) -> None:
    root = tmp_path / "voya_probe"
    root.mkdir()
    sequence = np.arange(2 * 60 * 1605, dtype=np.float32).reshape(2, 60, 1605)
    np.savez_compressed(root / "class_0001.npz", sequences=sequence, labels=np.array([0, 0]))
    (root / "labels.json").write_text(json.dumps({"class_0001": "xin chào"}), encoding="utf-8")

    npz_report = inspect_voya_npz(root / "class_0001.npz")
    probe_report = build_voya_probe_report(root=root, class_name="1")

    assert npz_report["arrays"]["sequences"]["shape"] == [2, 60, 1605]
    assert npz_report["sequence_statistics"]["shape"] == [60, 1605]
    assert probe_report["mapping_status"] == MAPPING_STRONGLY_SUPPORTED
    assert probe_report["safe_adapter_decision"].startswith("B.")


def test_write_voya_probe_outputs(tmp_path: Path) -> None:
    report = {
        "dataset": "VOYA_VSL",
        "downloaded_class": "class_0001",
        "mapping_status": MAPPING_STRONGLY_SUPPORTED,
        "canonical_conversion": "blocked_without_authoritative_layout",
        "safe_adapter_decision": "B. canonical conversion unverified, but direct 1605-feature training possible",
        "npz": {
            "path": "class_0001.npz",
            "sha256": "abc",
            "keys": ["sequences", "labels"],
            "number_of_sequences": 1,
            "sequence_statistics": {
                "shape": [60, 1605],
                "feature_dim": 1605,
                "nan_count": 0,
                "inf_count": 0,
                "zero_ratio": 0.0,
            },
            "candidate_layouts": {
                "candidates": [
                    {
                        "name": "pose25_face468_left21_right21_xyz",
                        "matches_feature_dim": True,
                        "expected_feature_dim": 1605,
                    }
                ]
            },
        },
        "direct_1605_feature_training": {
            "status": "feasible",
            "input_contract": "[T,1605] frame-level source-layout features",
            "reason": "ok",
        },
    }

    outputs = write_voya_probe_outputs(
        report,
        json_path=tmp_path / "probe.json",
        markdown_path=tmp_path / "probe.md",
    )

    assert Path(outputs["json"]).exists()
    assert "VOYA Structure Probe Report" in Path(outputs["markdown"]).read_text(encoding="utf-8")


def test_fetch_voya_probe_files_selects_one_class_without_network(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_remote_file_size(_url: str) -> int:
        return 12

    def fake_download(url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix == ".npz":
            np.savez_compressed(
                destination,
                sequences=np.zeros((1, 60, 1605), dtype=np.float32),
                labels=np.array([0]),
            )
        elif destination.name == "labels.json":
            destination.write_text(json.dumps({"class_0001": "xin chào"}), encoding="utf-8")
        else:
            destination.write_text(url, encoding="utf-8")

    monkeypatch.setattr(voya, "remote_file_size", fake_remote_file_size)
    monkeypatch.setattr(voya, "_download", fake_download)

    report = fetch_voya_probe_files(tmp_path, classes=["1"], max_class_size_mb=1)

    assert report["classes"] == ["class_0001"]
    assert (tmp_path / "class_0001.npz").exists()
    assert len(report["files"]) == 4
