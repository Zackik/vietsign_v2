from __future__ import annotations

import csv
from pathlib import Path

from vietsign.inference.config import ApiConfig, InferenceModelConfig
from vietsign.validation.audit import audit_dataset_config
from vietsign.validation.checkpoints import checkpoint_manifest
from vietsign.validation.discovery import discover_real_datasets, discover_videos
from vietsign.validation.reports import build_phase12_manifest, write_validation_docs


def test_discover_videos_and_real_dataset_absence(tmp_path: Path) -> None:
    video = tmp_path / "acceptance" / "one.mp4"
    video.parent.mkdir()
    video.write_bytes(b"not actually decoded in discovery")

    assert discover_videos(video.parent) == [video]
    discovery = discover_real_datasets(tmp_path)
    assert discovery["vsl400"]["available"] is False


def test_dataset_audit_detects_signer_leakage(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    video = tmp_path / "video.mp4"
    video.write_bytes(b"same bytes")
    with metadata.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample_id", "video_path", "label", "label_id", "split", "signer_id"],
        )
        writer.writeheader()
        writer.writerow({"sample_id": "a", "video_path": "video.mp4", "label": "HELLO", "label_id": 0, "split": "train", "signer_id": "s1"})
        writer.writerow({"sample_id": "b", "video_path": "video.mp4", "label": "HELLO", "label_id": 0, "split": "validation", "signer_id": "s1"})
    config = tmp_path / "dataset.yaml"
    config.write_text(
        f"""
dataset:
  name: tiny
  adapter: custom_csv
  root: {tmp_path}
  metadata_csv: {metadata}
validation:
  inspect_videos: false
""",
        encoding="utf-8",
    )

    audit = audit_dataset_config(config)

    assert audit["status"] == "audited"
    assert audit["signer_independent"] is False
    assert audit["signer_intersections"]["train_validation"] == 1
    assert audit["duplicate_paths"]


def test_phase12_manifest_and_docs_without_real_video(tmp_path: Path) -> None:
    api_config = ApiConfig(
        models=[
            InferenceModelConfig(
                name="missing_dev",
                task="gloss_translation",
                modality="text",
                checkpoint_path=tmp_path / "missing.pt",
                development_only=True,
            )
        ]
    )
    api_config_path = tmp_path / "api.yaml"
    api_config_path.write_text(api_config.model_dump_json(indent=2), encoding="utf-8")

    manifest = build_phase12_manifest(
        acceptance_dir=tmp_path / "acceptance",
        api_config_path=api_config_path,
        output_path=tmp_path / "phase12_manifest.json",
    )
    write_validation_docs(
        manifest,
        validation_report_path=tmp_path / "VALIDATION_REPORT.md",
        gaps_path=tmp_path / "REAL_DATA_GAPS.md",
    )
    ckpt_manifest = checkpoint_manifest(api_config)

    assert manifest["real_video_acceptance"]["status"] == "REAL VIDEO REQUIRED"
    assert "No real acceptance video" in manifest["known_blockers"][0]
    assert (tmp_path / "VALIDATION_REPORT.md").exists()
    assert ckpt_manifest["checkpoints"][0]["maturity"] == "development"
