"""Model comparison and synthetic smoke benchmark tooling."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from vietsign.data.schema import SignSample
from vietsign.models.graph import graph_for_features, visualize_graph
from vietsign.models.registry import (
    InputSpec,
    build_model,
    model_metadata,
    model_modality,
    parameter_count,
)
from vietsign.preprocessing.cache import save_keypoint_cache
from vietsign.training.config import RecognitionTrainingConfig, load_training_config
from vietsign.training.runner import (
    evaluate_checkpoint,
    predict_cached_features,
    predict_multimodal,
    predict_video,
    train_model,
)
from vietsign.vision.constants import CANONICAL_JOINT_NAMES, SCHEMA_VERSION
from vietsign.vision.schema import KeypointTensors

SYNTHETIC_LABEL = "SYNTHETIC SOFTWARE VALIDATION - NON-VSL / NOT A VSL BENCHMARK"


def model_info(config: RecognitionTrainingConfig, *, num_classes: int = 2) -> dict[str, Any]:
    if model_modality(config.model.type) == "rgb":
        model = build_model(config, num_classes=num_classes, input_spec=InputSpec(0, 3))
        metadata = model_metadata(model)
        metadata["input_frames"] = config.video.num_frames
        metadata["input_resolution"] = [config.rgb.crop_size, config.rgb.crop_size]
        metadata["rgb_preprocessing"] = config.rgb.model_dump(mode="json")
        return metadata
    graph = graph_for_features(config.features)
    input_spec = InputSpec(joint_count=graph.node_count, channel_count=_channel_count(config))
    model = build_model(config, num_classes=num_classes, input_spec=input_spec)
    metadata = model_metadata(model)
    metadata["selected_regions"] = config.features.model_dump(mode="json")
    metadata["graph_edges"] = len(graph.edges)
    metadata["graph_nodes"] = graph.node_count
    metadata["graph_visualization"] = str(visualize_graph(graph, Path("outputs/graphs/vietsign55.png")))
    return metadata


def run_synthetic_smoke_benchmark(
    config_paths: list[str | Path],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    synthetic_dataset_config, cache_root, feature_path, video_path = _write_synthetic_project(
        root / "synthetic_data"
    )
    rows = []
    for config_path in config_paths:
        config = load_training_config(config_path)
        config = _synthetic_config(config, synthetic_dataset_config, cache_root, root / "runs")
        started = time.perf_counter()
        summary = train_model(config)
        train_elapsed = time.perf_counter() - started
        checkpoint = Path(summary["best_checkpoint"])
        evaluation = evaluate_checkpoint(config, checkpoint_path=checkpoint, split="validation")
        if model_modality(config.model.type) == "multimodal":
            prediction = predict_multimodal(
                config,
                video_path=video_path,
                keypoints_path=feature_path,
                checkpoint_path=checkpoint,
            )
        elif model_modality(config.model.type) == "rgb":
            prediction = predict_video(config, video_path=video_path, checkpoint_path=checkpoint)
        else:
            prediction = predict_cached_features(config, features_path=feature_path, checkpoint_path=checkpoint)
        latency = _measure_latency(config)
        rows.append(
            {
                "model": config.model.type,
                "parameters": summary["model"]["parameter_count"],
                "top1": evaluation["metrics"]["top1_accuracy"],
                "top5": evaluation["metrics"]["top5_accuracy"],
                "macro_f1": evaluation["metrics"]["macro_f1"],
                "weighted_f1": evaluation["metrics"]["weighted_f1"],
                "inference_ms": latency["inference_ms"],
                "checkpoint": str(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "dataset": "synthetic_non_vsl",
                "modality": model_modality(config.model.type),
                "fusion_type": config.model.fusion_type if config.model.type == "multimodal" else None,
                "pretrained_backbone": config.model.pretrained_model if config.model.type == "videomae" else None,
                "pretrained": config.model.pretrained if config.model.type == "videomae" else None,
                "backbone_used": (
                    config.model.pretrained_model
                    if config.model.type == "videomae" and config.model.pretrained
                    else ("tiny-offline-videomae-compatible" if config.model.type == "videomae" else None)
                ),
                "freeze_backbone": config.model.freeze_backbone if config.model.type == "videomae" else None,
                "split_strategy": config.dataset.split_strategy,
                "seed": config.seed,
                "input_shape": latency["input_shape"],
                "feature_channels": latency["feature_channels"],
                "graph_nodes": summary["model"].get("graph_nodes"),
                "graph_edges": summary["model"].get("graph_edges"),
                "train_seconds": train_elapsed,
                "prediction_label": prediction["label"],
                "report_label": SYNTHETIC_LABEL,
            }
        )
    comparison_csv = root / "comparison.csv"
    comparison_json = root / "comparison.json"
    _write_csv(comparison_csv, rows)
    payload = {"label": SYNTHETIC_LABEL, "results": rows}
    comparison_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "label": SYNTHETIC_LABEL,
        "comparison_csv": str(comparison_csv),
        "comparison_json": str(comparison_json),
        "results": rows,
    }


def _synthetic_config(
    config: RecognitionTrainingConfig,
    dataset_config: Path,
    cache_root: Path,
    output_dir: Path,
) -> RecognitionTrainingConfig:
    config = config.model_copy(deep=True)
    suffix = f"_{config.model.fusion_type}" if config.model.type == "multimodal" else ""
    config.run.name = f"synthetic_{config.model.type}{suffix}"
    config.run.output_dir = output_dir
    config.dataset.config = dataset_config
    config.dataset.cache_root = cache_root
    config.dataset.expected_fingerprint = None
    config.sequence.mode = "fixed"
    config.sequence.length = 8
    config.video.num_frames = 4
    config.video.sampling = "uniform"
    config.rgb.resize = 32
    config.rgb.crop_size = 32
    config.data_quality.minimum_valid_joint_ratio = 0.0
    config.training.epochs = 1
    config.training.batch_size = 2
    config.training.device = "cpu"
    config.training.learning_rate = 0.001
    config.training.weight_decay = 0.0
    config.early_stopping.enabled = False
    if config.model.type == "temporal_transformer":
        config.model.embedding_dim = 16
        config.model.num_layers = 1
        config.model.num_heads = 4
    else:
        config.model.channels = [16, 16]
    if config.model.type == "videomae":
        config.model.pretrained = False
        config.model.freeze_backbone = True
        config.model.hidden_dim = 16
    if config.model.type == "multimodal":
        config.model.pretrained = False
        config.model.pretrained_model = "tiny-offline-videomae"
        config.model.freeze_rgb = True
        config.model.freeze_pose = False
        config.model.hidden_dim = 16
        config.model.embedding_dim = 16
        config.model.num_layers = 1
        config.model.num_heads = 4
        config.model.shared_dim = 16
    return config


def _write_synthetic_project(root: Path) -> tuple[Path, Path, Path, Path]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache_root = root / "cache"
    metadata = root / "metadata.csv"
    dataset_config = root / "dataset.yaml"
    samples = [
        _sample("a_train", "hello", 0, "train"),
        _sample("b_train", "thanks", 1, "train"),
        _sample("a_val", "hello", 0, "validation"),
        _sample("b_val", "thanks", 1, "validation"),
    ]
    with metadata.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample_id", "video_path", "label", "label_id", "signer_id", "split"],
        )
        writer.writeheader()
        for sample in samples:
            video_path = root / f"{sample.sample_id}.mp4"
            _write_synthetic_video(video_path, label_id=sample.label_id)
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "video_path": str(video_path),
                    "label": sample.label,
                    "label_id": sample.label_id,
                    "signer_id": sample.signer_id,
                    "split": sample.split,
                }
            )
            cached_sample = sample.model_copy(update={"video_path": video_path})
            _write_cache(cache_root, cached_sample, _arrays(label_offset=float(sample.label_id)))
    dataset_config.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "synthetic",
                    "adapter": "custom_csv",
                    "root": str(root),
                    "metadata_csv": str(metadata),
                },
                "validation": {"inspect_videos": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return dataset_config, cache_root, cache_root / "synthetic" / "a_train.npz", root / "a_train.mp4"


def _arrays(t: int = 8, *, label_offset: float = 0.0) -> dict[str, np.ndarray]:
    coordinates = np.zeros((t, 55, 3), dtype=np.float32)
    valid_mask = np.zeros((t, 55), dtype=bool)
    coordinates[:, 0, 0] = 1.0 + label_offset
    coordinates[:, 21, 1] = 0.5 + label_offset
    coordinates[:, 42, 0] = 0.25 + label_offset
    valid_mask[:, [0, 21, 42, 52, 53]] = True
    return {
        "coordinates": coordinates,
        "valid_mask": valid_mask,
        "confidence": np.full((t, 55), 0.8, dtype=np.float32),
        "source_mask": valid_mask.astype(np.uint8),
        "timestamps": np.arange(t, dtype=np.float32),
        "frame_indices": np.arange(t, dtype=np.int64),
        "joint_names": np.asarray(CANONICAL_JOINT_NAMES),
        "velocity": np.zeros((t, 55, 3), dtype=np.float32),
    }


def _sample(sample_id: str, label: str, label_id: int, split: str) -> SignSample:
    return SignSample(
        sample_id=sample_id,
        video_path=Path(f"{sample_id}.mp4"),
        label=label,
        label_id=label_id,
        split=split,
        dataset="synthetic",
        signer_id=f"signer_{sample_id}",
    )


def _write_cache(root: Path, sample: SignSample, arrays: dict[str, np.ndarray]) -> None:
    tensors = KeypointTensors(
        coordinates=arrays["coordinates"],
        valid_mask=arrays["valid_mask"],
        confidence=arrays["confidence"],
        source_mask=arrays["source_mask"],
        timestamps=arrays["timestamps"],
        frame_indices=arrays["frame_indices"],
        joint_names=tuple(str(item) for item in arrays["joint_names"]),
    )
    save_keypoint_cache(
        root / sample.dataset / f"{sample.sample_id}.npz",
        root / sample.dataset / f"{sample.sample_id}.json",
        tensors,
        sample=sample,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "fingerprint": None,
            "coverage": {"total": 1.0},
            "tensor_shape": {"coordinates": list(arrays["coordinates"].shape)},
        },
        extra_arrays={"velocity": arrays["velocity"]},
    )


def _measure_latency(config: RecognitionTrainingConfig) -> dict[str, Any]:
    if model_modality(config.model.type) == "multimodal":
        channels = _channel_count(config)
        model = build_model(config, num_classes=2, input_spec=InputSpec(joint_count=55, channel_count=channels))
        model.eval()
        pixel_values = torch.zeros((1, config.video.num_frames, 3, config.rgb.crop_size, config.rgb.crop_size))
        coordinates = torch.zeros((1, config.sequence.length or 8, 55, channels))
        valid_mask = torch.ones(coordinates.shape[:3], dtype=torch.bool)
        temporal_mask = torch.ones(coordinates.shape[:2], dtype=torch.bool)
        with torch.no_grad():
            model(
                pixel_values=pixel_values,
                coordinates=coordinates,
                valid_mask=valid_mask,
                temporal_mask=temporal_mask,
            )
            started = time.perf_counter()
            for _ in range(5):
                logits = model(
                    pixel_values=pixel_values,
                    coordinates=coordinates,
                    valid_mask=valid_mask,
                    temporal_mask=temporal_mask,
                )
            elapsed = time.perf_counter() - started
        return {
            "inference_ms": elapsed * 1000 / 5,
            "input_shape": {
                "rgb": list(pixel_values.shape),
                "pose": list(coordinates.shape),
            },
            "feature_channels": channels,
            "finite": bool(torch.isfinite(logits).all().item()),
            "parameter_count": parameter_count(model),
        }
    if model_modality(config.model.type) == "rgb":
        model = build_model(config, num_classes=2, input_spec=InputSpec(joint_count=0, channel_count=3))
        model.eval()
        coordinates = torch.zeros(
            (1, config.video.num_frames, 3, config.rgb.crop_size, config.rgb.crop_size)
        )
        with torch.no_grad():
            model(coordinates)
            started = time.perf_counter()
            for _ in range(5):
                logits = model(coordinates)
            elapsed = time.perf_counter() - started
        return {
            "inference_ms": elapsed * 1000 / 5,
            "input_shape": list(coordinates.shape),
            "feature_channels": 3,
            "finite": bool(torch.isfinite(logits).all().item()),
            "parameter_count": parameter_count(model),
        }
    graph = graph_for_features(config.features)
    channels = _channel_count(config)
    model = build_model(
        config,
        num_classes=2,
        input_spec=InputSpec(joint_count=graph.node_count, channel_count=channels),
    )
    model.eval()
    coordinates = torch.zeros((1, config.sequence.length or 8, graph.node_count, channels))
    valid_mask = torch.ones(coordinates.shape[:3], dtype=torch.bool)
    temporal_mask = torch.ones(coordinates.shape[:2], dtype=torch.bool)
    with torch.no_grad():
        model(coordinates, valid_mask, temporal_mask)
        started = time.perf_counter()
        for _ in range(5):
            logits = model(coordinates, valid_mask, temporal_mask)
        elapsed = time.perf_counter() - started
    return {
        "inference_ms": elapsed * 1000 / 5,
        "input_shape": list(coordinates.shape),
        "feature_channels": channels,
        "finite": bool(torch.isfinite(logits).all().item()),
        "parameter_count": parameter_count(model),
    }


def _channel_count(config: RecognitionTrainingConfig) -> int:
    count = 3
    if config.features.velocity:
        count += 3
    if config.features.confidence:
        count += 1
    if config.features.validity:
        count += 1
    if config.features.source_mask:
        count += 1
    return count


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_synthetic_video(path: Path, *, label_id: int) -> None:
    import cv2  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (64, 64),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create synthetic video: {path}")
    try:
        for index in range(8):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            color = (40 + label_id * 80, 80 + index * 8, 180 - label_id * 50)
            frame[:, :] = color
            cv2.putText(frame, str(label_id), (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            writer.write(frame)
    finally:
        writer.release()
