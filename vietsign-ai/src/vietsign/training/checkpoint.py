"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a checkpoint cannot be loaded into the current model."""


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: object | None,
    epoch: int,
    config: dict[str, Any],
    class_mapping: dict[str, int],
    best_metric: float,
    model_metadata: dict[str, Any],
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "config": config,
        "class_mapping": class_mapping,
        "best_metric": best_metric,
        "model_metadata": model_metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    required = {"model_state", "epoch", "config", "class_mapping", "model_metadata"}
    missing = required - set(checkpoint)
    if missing:
        raise CheckpointCompatibilityError(f"Checkpoint missing required keys: {sorted(missing)}")
    return checkpoint


def validate_model_compatibility(checkpoint: dict[str, Any], model_metadata: dict[str, Any]) -> None:
    expected = checkpoint.get("model_metadata", {})
    keys = (
        "model_name",
        "input_joints",
        "input_channels",
        "class_count",
        "embedding_dim",
        "num_layers",
        "num_heads",
        "channels",
        "graph_version",
        "graph_nodes",
        "modality",
        "pretrained_model",
        "pretrained",
        "freeze_backbone",
        "embedding_dim",
        "task_type",
        "vocab_size",
        "blank_index",
        "source_vocab_size",
        "target_vocab_size",
        "d_model",
        "encoder_layers",
        "decoder_layers",
    )
    mismatches = {
        key: (expected.get(key), model_metadata.get(key))
        for key in keys
        if key in expected and key in model_metadata and expected.get(key) != model_metadata.get(key)
    }
    if mismatches:
        raise CheckpointCompatibilityError(f"Checkpoint architecture mismatch: {mismatches}")
