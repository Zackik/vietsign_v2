"""Checkpoint manifest helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from vietsign.inference.config import ApiConfig
from vietsign.training.checkpoint import load_checkpoint


def checkpoint_manifest(api_config: ApiConfig) -> dict[str, Any]:
    return {
        "schema_version": "vietsign.checkpoint_manifest.v1",
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "checkpoints": [_checkpoint_entry(model) for model in api_config.models],
    }


def write_checkpoint_manifest(api_config: ApiConfig, path: str | Path) -> dict[str, Any]:
    payload = checkpoint_manifest(api_config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _checkpoint_entry(model) -> dict[str, Any]:
    entry = {
        "name": model.name,
        "task": model.task,
        "modality": model.modality,
        "path": str(model.checkpoint_path) if model.checkpoint_path else None,
        "config_path": str(model.config_path) if model.config_path else None,
        "maturity": model.maturity,
        "development_only": model.development_only,
        "exists": bool(model.checkpoint_path and Path(model.checkpoint_path).exists()),
    }
    if entry["exists"]:
        checkpoint = load_checkpoint(model.checkpoint_path, map_location="cpu")
        metadata = checkpoint.get("model_metadata", {})
        entry.update(
            {
                "epoch": checkpoint.get("epoch"),
                "best_metric": checkpoint.get("best_metric"),
                "model_name": metadata.get("model_name"),
                "parameter_count": metadata.get("parameter_count"),
                "task_type": metadata.get("task_type"),
                "created_from_config": checkpoint.get("config", {}).get("run", {}),
            }
        )
    return entry


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "uncommitted"
