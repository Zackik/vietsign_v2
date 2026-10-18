"""Phase 12 manifest and markdown report generation."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import torch

from vietsign.inference.config import load_api_config
from vietsign.inference.model_manager import ModelManager
from vietsign.inference.pipelines import pipeline_infos
from vietsign.preprocessing.config import load_preprocessing_config
from vietsign.validation.audit import audit_dataset_config
from vietsign.validation.checkpoints import checkpoint_manifest, write_checkpoint_manifest
from vietsign.validation.discovery import discover_real_datasets, discover_videos
from vietsign.vision.doctor import extractor_doctor


def build_phase12_manifest(
    *,
    acceptance_dir: str | Path = "data/acceptance",
    preprocessing_config: str | Path = "configs/preprocessing/rtmpose.yaml",
    api_config_path: str | Path = "configs/api/default.yaml",
    output_path: str | Path = "outputs/validation/phase12_manifest.json",
) -> dict[str, Any]:
    videos = discover_videos(acceptance_dir)
    pre_config = load_preprocessing_config(preprocessing_config)
    doctor = extractor_doctor(pre_config)
    api_config = load_api_config(api_config_path)
    manager = ModelManager(api_config)
    dataset_discovery = discover_real_datasets()
    dataset_audits = {
        "vsl400": audit_dataset_config("configs/data/vsl400.yaml"),
    }
    real_video_acceptance = {
        "status": "REAL VIDEO REQUIRED" if not videos else "PENDING_RUN",
        "acceptance_dir": str(acceptance_dir),
        "videos_found": [str(video) for video in videos],
        "extractor_doctor_ready": bool(doctor.get("ready_for_preprocessing")),
        "doctor": doctor,
    }
    readiness = {
        "real_extractor_ready": False,
        "real_islr_checkpoint_ready": False,
        "real_cslr_checkpoint_ready": False,
        "real_translation_checkpoint_ready": False,
    }
    payload = {
        "schema_version": "vietsign.phase12_manifest.v1",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": manager.device.type,
        },
        "real_video_acceptance": real_video_acceptance,
        "dataset_discovery": dataset_discovery,
        "dataset_audit": dataset_audits,
        "split_audit": _split_status(dataset_audits),
        "preprocessing_audit": {"status": "not_run", "reason": "real dataset/video unavailable"},
        "training_runs": [],
        "benchmark": {"status": "not_run", "reason": "real labeled dataset unavailable"},
        "api_readiness": {
            "pipelines": [info.model_dump(mode="json") for info in pipeline_infos(manager)],
            "readiness": readiness,
        },
        "realtime_readiness": {
            "status": "software_validated",
            "window_size": api_config.realtime.window_size,
            "stride": api_config.realtime.stride,
            "smoothing": api_config.realtime.smoothing,
            "real_latency_report": "not_available_without_real_pipeline",
        },
        "checkpoint_manifest": checkpoint_manifest(api_config),
        "known_blockers": _blockers(videos, dataset_discovery),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_checkpoint_manifest(api_config, output.parent / "checkpoint_manifest.json")
    return payload


def write_validation_docs(
    manifest: dict[str, Any],
    *,
    validation_report_path: str | Path = "docs/VALIDATION_REPORT.md",
    gaps_path: str | Path = "docs/REAL_DATA_GAPS.md",
) -> None:
    Path(validation_report_path).write_text(_validation_report_markdown(manifest), encoding="utf-8")
    Path(gaps_path).write_text(_real_data_gaps_markdown(manifest), encoding="utf-8")


def _split_status(dataset_audits: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "status": audit.get("status"),
            "signer_independent": audit.get("signer_independent"),
            "signer_intersections": audit.get("signer_intersections"),
        }
        for name, audit in dataset_audits.items()
    }


def _blockers(videos: list[Path], dataset_discovery: dict[str, Any]) -> list[str]:
    blockers = []
    if not videos:
        blockers.append("No real acceptance video found under data/acceptance.")
    if not any(item["available"] for item in dataset_discovery.values()):
        blockers.append("No real local VSL dataset videos found under data/raw.")
    blockers.append("No real continuous gloss-annotated VSL dataset found.")
    blockers.append("No paired gloss-Vietnamese real translation corpus found.")
    return blockers


def _validation_report_markdown(manifest: dict[str, Any]) -> str:
    acceptance = manifest["real_video_acceptance"]
    dataset_available = any(item["available"] for item in manifest["dataset_discovery"].values())
    return f"""# VietSignAI Validation Report

## Status Legend

- IMPLEMENTED: code exists.
- SOFTWARE-VALIDATED: tested with synthetic fixtures or static checks.
- REAL-DATA-VALIDATED: verified with real local VSL data.
- NOT YET VALIDATED: blocked by missing data or acceptance evidence.

## Subsystems

| Subsystem | Status | Evidence |
| --- | --- | --- |
| Extractor | NOT YET VALIDATED | {acceptance["status"]}; doctor ready={acceptance["extractor_doctor_ready"]} |
| ISLR | SOFTWARE-VALIDATED | Phase 04/06 tests pass; no real labeled subset available |
| RGB | SOFTWARE-VALIDATED | Phase 07 tests pass; no real subset RGB benchmark |
| Multimodal | SOFTWARE-VALIDATED | Phase 08 tests pass; no real aligned subset |
| CSLR | SOFTWARE-VALIDATED | Phase 09 synthetic CTC only |
| Translation | SOFTWARE-VALIDATED | Phase 10 synthetic paired text only |
| API | SOFTWARE-VALIDATED | Phase 11 API tests pass |
| Realtime | SOFTWARE-VALIDATED | Rolling-window/session tests pass |

## Real Video Acceptance

- Status: {acceptance["status"]}
- Videos found: {len(acceptance["videos_found"])}
- RTMPose doctor ready: {acceptance["extractor_doctor_ready"]}

## Real Dataset Availability

- Any real dataset available: {dataset_available}
- Discovery: `{json.dumps(manifest["dataset_discovery"], ensure_ascii=False)}`

## Blockers

{chr(10).join(f"- {item}" for item in manifest["known_blockers"])}
"""


def _real_data_gaps_markdown(manifest: dict[str, Any]) -> str:
    return f"""# Real Data Gaps

## ISLR

Real labeled isolated VSL clips are not available in the current repository
unless a local ignored dataset is added under `data/raw`. Current discovery:

```json
{json.dumps(manifest["dataset_discovery"], indent=2, ensure_ascii=False)}
```

## CSLR

No real continuous VSL videos with gloss sequence annotations were discovered.
Do not train Phase 09 on isolated clips and call it continuous recognition.

## SLT

No real paired gloss-sequence/Vietnamese-sentence corpus was discovered.
Phase 10 remains infrastructure-only until such data exists.
"""
