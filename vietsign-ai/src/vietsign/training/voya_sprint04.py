"""Scaled VOYA benchmark and leakage-audit runner."""

from __future__ import annotations

import csv
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from vietsign.data.voya import VOYA_REVISION
from vietsign.data.voya_direct import (
    VOYASampleIndex,
    analyze_voya_label_catalog,
    attach_sequence_hashes,
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
from vietsign.models.flat_features import FlatFeatureTemporalTransformerClassifier
from vietsign.training.checkpoint import save_checkpoint
from vietsign.training.voya_direct import (
    TemporalMeanMLPClassifier,
    _apply_top5_policy,
    _evaluate,
    _loader,
    _peak_memory_mb,
    _resolve_device,
    _run_epoch,
    _write_confusion_png,
    _write_history,
    _write_json,
)
from vietsign.utils.reproducibility import set_seed


class SingleFrameMLPClassifier(TemporalMeanMLPClassifier):
    """Mean MLP used with a single-frame dataset view."""


def load_voya_sprint04_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_voya_sprint04_benchmark(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_voya_sprint04_config(config_path)
    started = time.perf_counter()
    seed = int(config.get("seed", 42))
    set_seed(seed)
    root = Path(config["data"]["root"])
    run_dir = Path(config["run"]["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    label_analysis = analyze_voya_label_catalog(
        root / "labels.json",
        json_path="outputs/validation/voya_label_analysis.json",
        csv_path="outputs/validation/voya_label_groups.csv",
    )
    label_names = load_voya_label_names(root / "labels.json")
    samples = index_voya_npz_files(root)
    scale = int(config["experiment"]["class_count"])
    eligible_labels, eligibility = select_eligible_labels(
        samples,
        minimum_samples_per_class=int(config["experiment"]["minimum_samples_per_class"]),
        max_classes=scale,
    )
    if len(eligible_labels) < scale:
        raise RuntimeError(f"Only {len(eligible_labels)} classes available; requested {scale}.")
    selected = deterministic_subset(
        samples,
        labels=eligible_labels,
        max_samples_per_class=int(config["experiment"]["max_samples_per_class"]),
        seed=seed,
    )
    selected = attach_sequence_hashes(selected)
    duplicate_audit = exact_duplicate_audit(selected)
    split_samples = enforce_no_duplicate_hash_cross_split(
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
    leakage = duplicate_leakage_report(split_samples)
    if not leakage["passes"]:
        return {
            "status": "blocked",
            "reason": "cross_split_exact_duplicate_detected",
            "leakage": leakage,
        }

    normalization = fit_train_normalization(split_samples)
    label_to_index = {label: index for index, label in enumerate(sorted(eligible_labels))}
    class_names = [label_names[label] for label in sorted(eligible_labels)]
    split_manifest = write_split_manifest(
        split_samples,
        path=run_dir / "split_manifest.json",
        label_names={label: label_names[label] for label in eligible_labels},
    )
    _write_json(run_dir / "normalization.json", normalization)
    _write_json(run_dir / "eligibility.json", eligibility)
    _write_json(run_dir / "duplicate_audit.json", duplicate_audit)
    near_duplicate = near_duplicate_audit(split_samples, max_samples=int(config["audit"]["near_duplicate_samples"]))
    _write_json(run_dir / "near_duplicate_audit.json", near_duplicate)
    file_artifacts = file_artifact_audit(split_samples)
    _write_json(run_dir / "file_artifact_audit.json", file_artifacts)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    device = _resolve_device(config["training"].get("device", "auto"))
    experiment_results = []
    for variant in config["variants"]:
        result = run_variant(
            variant,
            split_samples=split_samples,
            label_to_index=label_to_index,
            normalization=normalization if variant.get("normalization", "zscore") == "zscore" else {"type": "none"},
            class_names=class_names,
            config=config,
            run_dir=run_dir,
            device=device,
            seed=seed,
        )
        experiment_results.append(result)
        if result["variant"] == "label_shuffle_negative_control" and result["metrics"]["top1_accuracy"] > (
            3.0 / len(class_names)
        ):
            summary = {
                "status": "potential_leakage",
                "reason": "label_shuffle_negative_control_above_threshold",
                "threshold": 3.0 / len(class_names),
                "negative_control_top1": result["metrics"]["top1_accuracy"],
            }
            _write_json(run_dir / "summary.json", summary)
            return summary

    summary = {
        "status": "complete",
        "voya_revision": VOYA_REVISION,
        "run_dir": str(run_dir),
        "label_catalog_size": label_analysis["label_catalog_size"],
        "unique_normalized_labels": label_analysis["unique_normalized_labels"],
        "duplicate_display_label_count": label_analysis["duplicate_display_label_count"],
        "downloaded_npz_files": sorted(str(path) for path in root.glob("class_*.npz")),
        "downloaded_npz_file_count": len(list(root.glob("class_*.npz"))),
        "downloaded_disk_size_bytes": sum(path.stat().st_size for path in root.glob("class_*.npz")),
        "largest_experiment_class_count": len(eligible_labels),
        "samples_per_class": int(config["experiment"]["max_samples_per_class"]),
        "split_counts": _split_sizes(split_manifest),
        "exact_duplicate_count": duplicate_audit["duplicate_sequence_instances"],
        "cross_split_duplicate_count": leakage["cross_split_duplicate_count"],
        "normalization": normalization["type"],
        "chance_top1": 1.0 / len(eligible_labels),
        "variants": experiment_results,
        "temporal_information_table": temporal_information_table(experiment_results),
        "regional_variant_analysis": regional_variant_analysis(experiment_results, class_names),
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_mb": _peak_memory_mb(),
        "signer_independent": False,
        "canonical_vietsign55": False,
        "checkpoint_maturity": "experimental_real_data",
    }
    _write_json(run_dir / "summary.json", summary)
    write_variant_table(run_dir / "variant_metrics.csv", experiment_results)
    return summary


def run_variant(
    variant: dict[str, Any],
    *,
    split_samples: list[VOYASampleIndex],
    label_to_index: dict[int, int],
    normalization: dict[str, Any],
    class_names: list[str],
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    variant_name = variant["name"]
    variant_dir = run_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    train_samples = [sample for sample in split_samples if sample.split == "train"]
    if variant.get("label_shuffle", False):
        train_samples = shuffle_training_labels(train_samples, labels=sorted(label_to_index), seed=seed)
    validation_samples = [sample for sample in split_samples if sample.split == "validation"]
    test_samples = [sample for sample in split_samples if sample.split == "test"]
    train_loader = _loader(
        train_samples,
        label_to_index,
        normalization,
        config["training"],
        int(config["data"].get("cache_size", 2)),
        shuffle=True,
        view=variant.get("view", "sequence"),
        temporal_shuffle_seed=seed if variant.get("view") == "temporal_shuffle" else None,
    )
    validation_loader = _loader(
        validation_samples,
        label_to_index,
        normalization,
        config["training"],
        int(config["data"].get("cache_size", 2)),
        shuffle=False,
        view=variant.get("view", "sequence"),
        temporal_shuffle_seed=seed if variant.get("view") == "temporal_shuffle" else None,
    )
    test_loader = _loader(
        test_samples,
        label_to_index,
        normalization,
        config["training"],
        int(config["data"].get("cache_size", 2)),
        shuffle=False,
        view=variant.get("view", "sequence"),
        temporal_shuffle_seed=seed if variant.get("view") == "temporal_shuffle" else None,
    )
    model = build_variant_model(variant, config, num_classes=len(label_to_index)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(variant.get("learning_rate", config["training"]["learning_rate"])),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    history = []
    best_metric = -math.inf
    best_epoch = -1
    best_path = variant_dir / "best.pt"
    for epoch in range(1, int(variant["epochs"]) + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=config["training"].get("gradient_clip_norm"),
        )
        validation_metrics = _apply_top5_policy(
            _evaluate(model, validation_loader, criterion, device, class_names),
            class_names,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_top1": train_metrics["top1_accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_top1": validation_metrics["top1_accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "gradient_norm": train_metrics["gradient_norm"],
            }
        )
        if validation_metrics["macro_f1"] > best_metric:
            best_metric = validation_metrics["macro_f1"]
            best_epoch = epoch
            if variant.get("save_checkpoint", True):
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    epoch=epoch,
                    config=config,
                    class_mapping={str(label): index for label, index in label_to_index.items()},
                    best_metric=best_metric,
                    model_metadata={
                        "model_name": variant["model"],
                        "variant": variant_name,
                        "dataset": "VOYA_VSL",
                        "real_data": True,
                        "representation": "native_1605",
                        "canonical_vietsign55": False,
                        "signer_independent": False,
                        "maturity": "experimental_real_data",
                        "remote_revision": VOYA_REVISION,
                        "class_count": len(label_to_index),
                    },
                )
    _write_history(variant_dir / "training_history.csv", history)
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
    metrics = _apply_top5_policy(_evaluate(model, test_loader, criterion, device, class_names), class_names)
    ci = bootstrap_metric_ci(metrics["confusion_matrix"], seed=seed)
    metrics["top1_ci"] = ci["top1"]
    metrics["macro_f1_ci"] = ci["macro_f1"]
    metrics["chance_top1"] = 1.0 / len(class_names)
    metrics["test_sample_count"] = len(test_samples)
    _write_json(variant_dir / "evaluation.json", metrics)
    write_per_class_metrics_csv(
        metrics["per_class_accuracy"],
        supports=_supports_by_class(test_samples, class_names, label_to_index),
        path=variant_dir / "per_class_metrics.csv",
    )
    write_top_confusions_csv(metrics["confusion_matrix"], class_names, path=variant_dir / "top_confusions.csv")
    _write_confusion_png(metrics["confusion_matrix"], class_names, variant_dir / "confusion_matrix.png")
    return {
        "variant": variant_name,
        "model": variant["model"],
        "view": variant.get("view", "sequence"),
        "normalization": normalization["type"],
        "label_shuffle": bool(variant.get("label_shuffle", False)),
        "top1_accuracy": metrics["top1_accuracy"],
        "top5_accuracy": metrics["top5_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "chance_top1": metrics["chance_top1"],
        "test_sample_count": metrics["test_sample_count"],
        "best_validation_macro_f1": best_metric,
        "best_epoch": best_epoch,
        "runtime_seconds": time.perf_counter() - started,
        "checkpoint_path": str(best_path) if best_path.exists() else None,
        "metrics": metrics,
    }


def build_variant_model(variant: dict[str, Any], config: dict[str, Any], *, num_classes: int) -> nn.Module:
    if variant["model"] == "transformer":
        return FlatFeatureTemporalTransformerClassifier(
            input_dim=int(config["model"]["input_dim"]),
            num_classes=num_classes,
            embedding_dim=int(config["model"]["d_model"]),
            num_layers=int(config["model"]["num_layers"]),
            num_heads=int(config["model"]["num_heads"]),
            dropout=float(config["model"]["dropout"]),
        )
    if variant["model"] in {"mean_mlp", "single_frame_mlp"}:
        return TemporalMeanMLPClassifier(
            input_dim=int(config["model"]["input_dim"]),
            hidden_dim=int(config["baseline"]["hidden_dim"]),
            num_classes=num_classes,
            dropout=float(config["model"]["dropout"]),
        )
    raise ValueError(f"Unsupported Sprint 04 model: {variant['model']}")


def exact_duplicate_audit(samples: list[VOYASampleIndex]) -> dict[str, Any]:
    by_hash: dict[str, list[VOYASampleIndex]] = defaultdict(list)
    for sample in samples:
        by_hash[sample.sequence_hash or ""].append(sample)
    duplicate_groups = [
        {
            "hash": hash_value,
            "count": len(group),
            "labels": sorted({sample.raw_label for sample in group}),
            "sample_ids": [sample.sample_id for sample in group[:20]],
        }
        for hash_value, group in by_hash.items()
        if len(group) > 1
    ]
    within = sum(1 for group in duplicate_groups if len(group["labels"]) == 1)
    across = sum(1 for group in duplicate_groups if len(group["labels"]) > 1)
    instances = sum(group["count"] for group in duplicate_groups)
    return {
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_sequence_instances": instances,
        "duplicate_percentage": instances / max(len(samples), 1),
        "within_class_duplicate_groups": within,
        "across_class_duplicate_groups": across,
        "groups": duplicate_groups[:100],
    }


def near_duplicate_audit(samples: list[VOYASampleIndex], *, max_samples: int = 300) -> dict[str, Any]:
    rng = random.Random(42)
    subset = list(samples)
    rng.shuffle(subset)
    subset = subset[:max_samples]
    vectors = []
    metadata = []
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sample in subset:
        if sample.source_file not in arrays:
            with np.load(sample.source_file, allow_pickle=False) as data:
                arrays[sample.source_file] = (np.asarray(data["sequences"]), np.asarray(data["labels"]))
        sequence = arrays[sample.source_file][0][sample.source_index]
        vector = np.concatenate([sequence.mean(axis=0), sequence.std(axis=0)]).astype(np.float32)
        norm = np.linalg.norm(vector)
        vectors.append(vector / max(norm, 1e-12))
        metadata.append(sample)
    matrix = np.stack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)
    suspicious = []
    for i in range(len(metadata)):
        sims = matrix[i + 1 :] @ matrix[i]
        for offset in np.where(sims > 0.9995)[0][:5]:
            j = i + 1 + int(offset)
            suspicious.append(
                {
                    "sample_a": metadata[i].sample_id,
                    "label_a": metadata[i].raw_label,
                    "sample_b": metadata[j].sample_id,
                    "label_b": metadata[j].raw_label,
                    "cosine_similarity": float(sims[offset]),
                }
            )
    return {
        "method": "cosine over concatenated temporal mean and std vectors",
        "sampled_sequences": len(metadata),
        "threshold": 0.9995,
        "suspicious_pair_count": len(suspicious),
        "suspicious_pairs": suspicious[:100],
        "diagnostic_only": True,
    }


def file_artifact_audit(samples: list[VOYASampleIndex]) -> dict[str, Any]:
    by_label: dict[int, list[np.ndarray]] = defaultdict(list)
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sample in samples:
        if sample.source_file not in arrays:
            with np.load(sample.source_file, allow_pickle=False) as data:
                arrays[sample.source_file] = (np.asarray(data["sequences"]), np.asarray(data["labels"]))
        by_label[sample.raw_label].append(arrays[sample.source_file][0][sample.source_index])
    rows = {}
    for label, sequences in sorted(by_label.items()):
        stacked = np.stack(sequences)
        rows[str(label)] = {
            "sample_count": int(stacked.shape[0]),
            "feature_mean": float(stacked.mean()),
            "feature_std": float(stacked.std()),
            "zero_ratio": float(np.count_nonzero(stacked == 0) / stacked.size),
            "sequence_variance_mean": float(stacked.var(axis=1).mean()),
            "min": float(stacked.min()),
            "max": float(stacked.max()),
        }
    means = [item["feature_mean"] for item in rows.values()]
    stds = [item["feature_std"] for item in rows.values()]
    return {
        "per_class": rows,
        "feature_mean_range": [min(means), max(means)] if means else None,
        "feature_std_range": [min(stds), max(stds)] if stds else None,
        "diagnostic": "Class-level distribution differences may indicate static separability or file-specific artifacts; not proof of leakage.",
    }


def shuffle_training_labels(samples: list[VOYASampleIndex], *, labels: list[int], seed: int) -> list[VOYASampleIndex]:
    rng = random.Random(seed)
    shuffled = []
    for sample in samples:
        shuffled.append(sample.__class__(**(sample.__dict__ | {"raw_label": rng.choice(labels)})))
    return shuffled


def bootstrap_metric_ci(confusion_matrix: list[list[int]], *, seed: int, iterations: int = 300) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    pairs = []
    for actual, row in enumerate(confusion_matrix):
        for predicted, count in enumerate(row):
            pairs.extend([(actual, predicted)] * int(count))
    if len(pairs) < 10:
        return {"top1": [float("nan"), float("nan")], "macro_f1": [float("nan"), float("nan")]}
    pairs_array = np.asarray(pairs)
    top1_values = []
    macro_values = []
    class_count = len(confusion_matrix)
    for _ in range(iterations):
        sample = pairs_array[rng.integers(0, len(pairs_array), size=len(pairs_array))]
        confusion = np.zeros((class_count, class_count), dtype=np.int64)
        for actual, predicted in sample:
            confusion[int(actual), int(predicted)] += 1
        top1_values.append(float(np.trace(confusion) / confusion.sum()))
        macro_values.append(_macro_f1_from_confusion(confusion))
    return {
        "top1": [float(np.percentile(top1_values, 2.5)), float(np.percentile(top1_values, 97.5))],
        "macro_f1": [float(np.percentile(macro_values, 2.5)), float(np.percentile(macro_values, 97.5))],
    }


def temporal_information_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "variant": result["variant"],
            "model": result["model"],
            "view": result["view"],
            "top1": result["top1_accuracy"],
            "macro_f1": result["macro_f1"],
            "interpretation": "diagnostic, not signer-independent",
        }
        for result in results
        if result["variant"] in {"transformer_normal", "mean_pool_mlp", "single_frame_mlp", "transformer_temporal_shuffle"}
    ]


def regional_variant_analysis(results: list[dict[str, Any]], class_names: list[str]) -> dict[str, Any]:
    first = next((result for result in results if result["variant"] == "transformer_normal"), None)
    if first is None:
        return {"status": "not_available"}
    regional_indices = [
        index
        for index, name in enumerate(class_names)
        if any(marker in name.lower() for marker in ("(bắc)", "(nam)", "(trung)"))
    ]
    confusion = first["metrics"]["confusion_matrix"]
    correct = sum(confusion[index][index] for index in regional_indices)
    total = sum(sum(confusion[index]) for index in regional_indices)
    return {
        "regional_class_names": [class_names[index] for index in regional_indices],
        "regional_test_samples": total,
        "regional_top1": correct / total if total else None,
        "caution": "May reflect real regional signs or dataset artifacts; no linguistic conclusion is claimed.",
    }


def write_variant_table(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "model",
                "view",
                "normalization",
                "top1_accuracy",
                "top5_accuracy",
                "macro_f1",
                "weighted_f1",
                "chance_top1",
                "test_sample_count",
                "runtime_seconds",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in writer.fieldnames})


def _macro_f1_from_confusion(confusion: np.ndarray) -> float:
    f1s = []
    for class_id in range(confusion.shape[0]):
        tp = int(confusion[class_id, class_id])
        support = int(confusion[class_id].sum())
        predicted = int(confusion[:, class_id].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1s))


def _supports_by_class(samples: list[VOYASampleIndex], class_names: list[str], label_to_index: dict[int, int]) -> dict[str, int]:
    raw_counts = Counter(sample.raw_label for sample in samples)
    return {class_names[label_to_index[label]]: count for label, count in raw_counts.items()}


def _split_sizes(split_manifest: dict[str, Any]) -> dict[str, int]:
    return {split: sum(counts.values()) for split, counts in split_manifest["counts"].items()}
