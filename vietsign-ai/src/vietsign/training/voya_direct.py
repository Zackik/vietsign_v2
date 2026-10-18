"""Training loop for VOYA native [T,1605] real-data baseline."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from vietsign.data.voya import VOYA_REVISION
from vietsign.data.voya_direct import (
    VOYADirectFeatureDataset,
    attach_sequence_hashes,
    collate_voya_direct,
    compute_voya_direct_stats,
    deterministic_subset,
    duplicate_leakage_report,
    enforce_no_duplicate_hash_cross_split,
    fit_train_normalization,
    index_voya_npz_files,
    load_voya_label_names,
    select_eligible_labels,
    stratified_sample_split,
    write_per_class_metrics_csv,
    write_split_manifest,
    write_top_confusions_csv,
)
from vietsign.evaluation.metrics import compute_metrics
from vietsign.models.flat_features import FlatFeatureTemporalTransformerClassifier
from vietsign.training.checkpoint import save_checkpoint
from vietsign.utils.device import select_device
from vietsign.utils.reproducibility import set_seed


class TemporalMeanMLPClassifier(nn.Module):
    """Small sanity baseline: mean pool native features then classify."""

    def __init__(self, *, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor, temporal_mask: torch.Tensor) -> torch.Tensor:
        weights = temporal_mask.unsqueeze(-1).to(features.dtype)
        pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.network(pooled)


def load_voya_direct_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_voya_direct_experiment(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_voya_direct_config(config_path)
    start = time.perf_counter()
    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_config = config["data"]
    experiment_config = config["experiment"]
    training_config = config["training"]
    model_config = config["model"]
    root = Path(data_config["root"])
    run_dir = Path(config["run"]["output_dir"]) / config["run"]["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_voya_direct_stats(root, output_path=run_dir / "dataset_stats.json")
    all_samples = index_voya_npz_files(root)
    eligible_labels, eligibility = select_eligible_labels(
        all_samples,
        minimum_samples_per_class=int(experiment_config["minimum_samples_per_class"]),
        max_classes=int(experiment_config["max_classes"]),
    )
    if len(eligible_labels) < int(experiment_config["minimum_classes"]):
        raise RuntimeError(
            f"Only {len(eligible_labels)} eligible labels; need {experiment_config['minimum_classes']}."
        )
    selected = deterministic_subset(
        all_samples,
        labels=eligible_labels,
        max_samples_per_class=experiment_config.get("max_samples_per_class"),
        seed=seed,
    )
    split_samples = enforce_no_duplicate_hash_cross_split(
        attach_sequence_hashes(
            stratified_sample_split(
                selected,
                seed=seed,
                ratios={
                    "train": float(config["split"]["train"]),
                    "validation": float(config["split"]["validation"]),
                    "test": float(config["split"]["test"]),
                },
            )
        )
    )
    leakage = duplicate_leakage_report(split_samples)
    if not leakage["passes"]:
        raise RuntimeError("Exact duplicate feature arrays cross train/validation/test splits.")

    label_to_index = {label: index for index, label in enumerate(sorted(eligible_labels))}
    label_names_by_raw = load_voya_label_names(root / "labels.json")
    class_names = [label_names_by_raw[label] for label in sorted(eligible_labels)]
    normalization = (
        fit_train_normalization(split_samples)
        if config["normalization"]["type"] == "per_feature_zscore"
        else {"type": "none", "fit_split": "none"}
    )
    _write_json(run_dir / "normalization.json", normalization)
    split_manifest = write_split_manifest(
        split_samples,
        path=run_dir / "split_manifest.json",
        label_names={label: label_names_by_raw[label] for label in eligible_labels},
    )
    _write_json(
        run_dir / "config.yaml.json",
        {
            "config_path": str(config_path),
            "config": config,
            "eligibility": eligibility,
            "leakage": leakage,
        },
    )
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    device = _resolve_device(training_config.get("device", "auto"))
    train_samples = [sample for sample in split_samples if sample.split == "train"]
    validation_samples = [sample for sample in split_samples if sample.split == "validation"]
    test_samples = [sample for sample in split_samples if sample.split == "test"]
    cache_size = int(data_config.get("cache_size", 2))
    loaders = {
        "train": _loader(train_samples, label_to_index, normalization, training_config, cache_size, shuffle=True),
        "validation": _loader(validation_samples, label_to_index, normalization, training_config, cache_size, shuffle=False),
        "test": _loader(test_samples, label_to_index, normalization, training_config, cache_size, shuffle=False),
    }

    overfit = _run_overfit_check(
        split_samples,
        label_to_index=label_to_index,
        normalization=normalization,
        config=config,
        device=device,
    )

    model = FlatFeatureTemporalTransformerClassifier(
        input_dim=int(model_config["input_dim"]),
        num_classes=len(label_to_index),
        embedding_dim=int(model_config["d_model"]),
        num_layers=int(model_config["num_layers"]),
        num_heads=int(model_config["num_heads"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    best_metric = -math.inf
    best_epoch = -1
    history = []
    best_path = run_dir / "best.pt"
    epochs = int(training_config["epochs"])
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            loaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=training_config.get("gradient_clip_norm"),
        )
        validation_metrics = _evaluate(model, loaders["validation"], criterion, device, class_names)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1_accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_top1": validation_metrics["top1_accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "gradient_norm": train_metrics["gradient_norm"],
            "collapsed_prediction": validation_metrics["collapsed_prediction"],
        }
        history.append(row)
        if validation_metrics["macro_f1"] > best_metric:
            best_metric = validation_metrics["macro_f1"]
            best_epoch = epoch
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=epoch,
                config=config,
                class_mapping={str(label): index for label, index in label_to_index.items()},
                best_metric=best_metric,
                model_metadata=_checkpoint_metadata(model, config_path, config, len(label_to_index)),
            )
    _write_history(run_dir / "training_history.csv", history)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = _apply_top5_policy(_evaluate(model, loaders["test"], criterion, device, class_names), class_names)
    baseline_metrics = _apply_top5_policy(
        _run_mlp_baseline(loaders, config, device, class_names, len(label_to_index)),
        class_names,
    )

    evaluation = {
        "title": "REAL VOYA_VSL RESULTS",
        "not_signer_independent": True,
        "split_type": "sample_stratified",
        "signer_independent": False,
        "canonical_vietsign55": False,
        "representation": "native_1605",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_metric,
        "test": test_metrics,
        "mean_pool_mlp_baseline": baseline_metrics,
        "overfit_small_check": overfit,
        "label_mapping": {str(label): index for label, index in label_to_index.items()},
        "class_names": class_names,
    }
    _write_json(run_dir / "evaluation.json", evaluation)
    write_per_class_metrics_csv(
        test_metrics["per_class_accuracy"],
        supports=_supports_by_class(test_samples, class_names, label_to_index),
        path=run_dir / "per_class_metrics.csv",
    )
    write_top_confusions_csv(test_metrics["confusion_matrix"], class_names, path=run_dir / "top_confusions.csv")
    _write_confusion_png(test_metrics["confusion_matrix"], class_names, run_dir / "confusion_matrix.png")
    _register_checkpoint(run_dir, best_path, config_path, config, model, best_metric, best_epoch)

    runtime = time.perf_counter() - start
    summary = {
        "status": "complete",
        "run_dir": str(run_dir),
        "checkpoint_path": str(best_path),
        "checkpoint_maturity": "experimental_real_data",
        "runtime_seconds": runtime,
        "peak_memory_mb": _peak_memory_mb(),
        "voya_revision": VOYA_REVISION,
        "downloaded_files": [file["path"] for file in stats["files"]],
        "downloaded_size_bytes": stats["downloaded_size_bytes"],
        "npz_organization_conclusion": stats["npz_organization_conclusion"],
        "total_real_sequences_inspected": stats["total_sequences"],
        "unique_labels": stats["unique_labels"],
        "eligible_labels": eligible_labels,
        "selected_experimental_labels": eligible_labels,
        "samples_per_split": _split_sizes(split_manifest),
        "split_strategy": "sample_stratified",
        "signer_independent": False,
        "input_shape": "[B,T,1605]",
        "normalization": normalization["type"],
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "best_validation_metrics": {"macro_f1": best_metric, "epoch": best_epoch},
        "final_test_top1": test_metrics["top1_accuracy"],
        "final_test_top5": test_metrics["top5_accuracy"] if len(label_to_index) >= 5 else None,
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "overfit_small_check": overfit,
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


def _loader(
    samples: list,
    label_to_index: dict[int, int],
    normalization: dict[str, Any],
    training_config: dict[str, Any],
    cache_size: int,
    *,
    shuffle: bool,
    view: str = "sequence",
    temporal_shuffle_seed: int | None = None,
) -> DataLoader:
    dataset = VOYADirectFeatureDataset(
        samples,
        label_to_index=label_to_index,
        normalization=normalization,
        cache_size=cache_size,
        view=view,
        temporal_shuffle_seed=temporal_shuffle_seed,
    )
    return DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_voya_direct,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float | None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    grad_norm = 0.0
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["temporal_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, mask)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError("NaN/Inf detected in VOYA direct model logits.")
        loss = criterion(logits, labels)
        loss.backward()
        if gradient_clip_norm:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm)))
        optimizer.step()
        total_loss += float(loss.item()) * labels.numel()
        correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total += int(labels.numel())
    return {
        "loss": total_loss / max(total, 1),
        "top1_accuracy": correct / max(total, 1),
        "gradient_norm": grad_norm,
    }


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    model.eval()
    logits_chunks = []
    label_chunks = []
    total_loss = 0.0
    total = 0
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["temporal_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(features, mask)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError("NaN/Inf detected during VOYA direct evaluation.")
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.numel()
        total += int(labels.numel())
        logits_chunks.append(logits.detach().cpu())
        label_chunks.append(labels.detach().cpu())
    logits_all = torch.cat(logits_chunks, dim=0)
    labels_all = torch.cat(label_chunks, dim=0)
    metrics = compute_metrics(
        logits_all,
        labels_all,
        class_names=class_names,
        loss=total_loss / max(total, 1),
    ).__dict__
    predictions = logits_all.argmax(dim=-1)
    metrics["collapsed_prediction"] = len(torch.unique(predictions)) == 1 and len(class_names) > 1
    return metrics


def _run_overfit_check(
    samples: list,
    *,
    label_to_index: dict[int, int],
    normalization: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    labels = sorted(label_to_index)[: int(config["overfit_check"]["classes"])]
    tiny = deterministic_subset(
        samples,
        labels=labels,
        max_samples_per_class=int(config["overfit_check"]["samples_per_class"]),
        seed=int(config.get("seed", 42)),
    )
    for sample_index, sample in enumerate(tiny):
        tiny[sample_index] = sample.__class__(**(sample.__dict__ | {"split": "train"}))
    loader = _loader(
        tiny,
        label_to_index=label_to_index,
        normalization=normalization,
        training_config=config["overfit_check"],
        cache_size=int(config["data"].get("cache_size", 2)),
        shuffle=True,
    )
    model = FlatFeatureTemporalTransformerClassifier(
        input_dim=int(config["model"]["input_dim"]),
        num_classes=len(label_to_index),
        embedding_dim=int(config["model"]["d_model"]),
        num_layers=int(config["model"]["num_layers"]),
        num_heads=int(config["model"]["num_heads"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["overfit_check"]["learning_rate"]))
    criterion = nn.CrossEntropyLoss()
    final = {"top1_accuracy": 0.0, "loss": math.inf}
    for _ in range(int(config["overfit_check"]["epochs"])):
        final = _run_epoch(
            model,
            loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=config["training"].get("gradient_clip_norm"),
        )
    return {
        "classes": labels,
        "samples": len(tiny),
        "final_train_loss": final["loss"],
        "final_train_top1": final["top1_accuracy"],
        "passes": final["top1_accuracy"] >= float(config["overfit_check"]["minimum_top1"]),
        "not_benchmark": True,
    }


def _run_mlp_baseline(
    loaders: dict[str, DataLoader],
    config: dict[str, Any],
    device: torch.device,
    class_names: list[str],
    num_classes: int,
) -> dict[str, Any]:
    baseline_config = config["baseline"]
    if not baseline_config.get("enabled", True):
        return {"status": "disabled"}
    model = TemporalMeanMLPClassifier(
        input_dim=int(config["model"]["input_dim"]),
        hidden_dim=int(baseline_config["hidden_dim"]),
        num_classes=num_classes,
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(baseline_config["learning_rate"]))
    criterion = nn.CrossEntropyLoss()
    for _ in range(int(baseline_config["epochs"])):
        _run_epoch(
            model,
            loaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=config["training"].get("gradient_clip_norm"),
        )
    return _evaluate(model, loaders["test"], criterion, device, class_names)


def _apply_top5_policy(metrics: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    metrics = dict(metrics)
    metrics["top5_applicable"] = len(class_names) >= 5
    if len(class_names) < 5:
        metrics["top5_accuracy"] = None
    return metrics


def _checkpoint_metadata(
    model: nn.Module,
    config_path: Path,
    config: dict[str, Any],
    class_count: int,
) -> dict[str, Any]:
    metadata = model.introspect().__dict__ if hasattr(model, "introspect") else {}
    metadata.update(
        {
            "dataset": "VOYA_VSL",
            "real_data": True,
            "representation": "native_1605",
            "canonical_vietsign55": False,
            "signer_independent": False,
            "maturity": "experimental_real_data",
            "config_path": str(config_path),
            "class_count": class_count,
            "remote_revision": VOYA_REVISION,
            "split_type": config["split"]["type"],
        }
    )
    return metadata


def _register_checkpoint(
    run_dir: Path,
    checkpoint_path: Path,
    config_path: Path,
    config: dict[str, Any],
    model: nn.Module,
    best_metric: float,
    best_epoch: int,
) -> None:
    manifest_path = Path("outputs/validation/checkpoint_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "schema_version": "vietsign.checkpoint_manifest.v1",
        "checkpoints": [],
    }
    entry = {
        "name": config["run"]["name"],
        "task": "isolated_recognition",
        "modality": "flat_features",
        "path": str(checkpoint_path),
        "config_path": str(config_path),
        "maturity": "experimental_real_data",
        "development_only": False,
        "exists": checkpoint_path.exists(),
        "epoch": best_epoch,
        "best_metric": best_metric,
        "model_name": "flat_feature_temporal_transformer",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "task_type": "isolated_recognition",
        "dataset": "VOYA_VSL",
        "real_data": True,
        "representation": "native_1605",
        "canonical_vietsign55": False,
        "signer_independent": False,
        "maturity_note": "experimental_real_data",
        "run_dir": str(run_dir),
    }
    manifest["checkpoints"] = [
        item for item in manifest.get("checkpoints", []) if item.get("name") != config["run"]["name"]
    ] + [entry]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_confusion_png(confusion_matrix: list[list[int]], class_names: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(confusion_matrix)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), labels=class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("VOYA direct-feature confusion matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device(select_device().type)
    return torch.device(value)


def _split_sizes(split_manifest: dict[str, Any]) -> dict[str, int]:
    return {
        split: sum(counts.values())
        for split, counts in split_manifest["counts"].items()
    }


def _supports_by_class(samples: list, class_names: list[str], label_to_index: dict[int, int]) -> dict[str, int]:
    raw_counts = Counter(sample.raw_label for sample in samples)
    supports = {class_name: 0 for class_name in class_names}
    for raw_label, count in raw_counts.items():
        supports[class_names[label_to_index[raw_label]]] = count
    return supports


def _peak_memory_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024 / 1024 if usage > 10_000_000 else usage / 1024
