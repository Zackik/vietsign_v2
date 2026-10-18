from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2  # type: ignore
import numpy as np
import pytest
import torch

from vietsign.data.schema import SignSample
from vietsign.models.registry import InputSpec, build_model, model_metadata, model_modality
from vietsign.preprocessing.cache import save_keypoint_cache
from vietsign.training.benchmark import run_synthetic_smoke_benchmark
from vietsign.training.config import RecognitionTrainingConfig
from vietsign.training.multimodal import MultimodalDatasetError, MultimodalVSLDataset
from vietsign.training.runner import evaluate_checkpoint, predict_multimodal, train_model
from vietsign.vision.constants import CANONICAL_JOINT_NAMES, SCHEMA_VERSION
from vietsign.vision.schema import KeypointTensors


def _write_video(path: Path, *, label_id: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (64, 64))
    assert writer.isOpened()
    try:
        for index in range(8):
            frame = np.full((64, 64, 3), (40 + label_id * 80, 120, 160), dtype=np.uint8)
            cv2.putText(frame, str(index), (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(frame)
    finally:
        writer.release()


def _arrays(t: int = 6) -> dict[str, np.ndarray]:
    coordinates = np.zeros((t, 55, 3), dtype=np.float32)
    valid_mask = np.zeros((t, 55), dtype=bool)
    coordinates[:, 0, 0] = 1.0
    coordinates[:, 21, 1] = 0.5
    coordinates[:, 42, 0] = 0.25
    valid_mask[:, [0, 21, 42, 52, 53]] = True
    return {
        "coordinates": coordinates,
        "valid_mask": valid_mask,
        "confidence": np.full((t, 55), 0.8, dtype=np.float32),
        "source_mask": valid_mask.astype(np.uint8),
        "timestamps": np.arange(t, dtype=np.float32),
        "frame_indices": np.arange(t, dtype=np.int64),
        "joint_names": np.asarray(CANONICAL_JOINT_NAMES),
    }


def _write_cache(root: Path, sample: SignSample, *, label: str | None = None) -> None:
    arrays = _arrays()
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
        sample=sample.model_copy(update={"label": label or sample.label}),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "fingerprint": None,
            "coverage": {"total": 1.0},
            "tensor_shape": {"coordinates": list(arrays["coordinates"].shape)},
        },
    )


def _project(tmp_path: Path, *, fusion_type: str = "concat") -> Path:
    metadata = tmp_path / "metadata.csv"
    dataset_config = tmp_path / "dataset.yaml"
    cache_root = tmp_path / "cache"
    rows = [
        ("a_train", "hello", 0, "train"),
        ("b_train", "thanks", 1, "train"),
        ("a_val", "hello", 0, "validation"),
        ("b_val", "thanks", 1, "validation"),
    ]
    with metadata.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample_id", "video_path", "label", "label_id", "signer_id", "split"],
        )
        writer.writeheader()
        for sample_id, label, label_id, split in rows:
            video_path = tmp_path / f"{sample_id}.mp4"
            _write_video(video_path, label_id=label_id)
            sample = SignSample(
                sample_id=sample_id,
                video_path=video_path,
                label=label,
                label_id=label_id,
                split=split,
                dataset="synthetic",
                signer_id=f"signer_{sample_id}",
            )
            _write_cache(cache_root, sample)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "video_path": str(video_path),
                    "label": label,
                    "label_id": label_id,
                    "signer_id": sample.signer_id,
                    "split": split,
                }
            )
    dataset_config.write_text(
        f"""
dataset:
  name: synthetic
  adapter: custom_csv
  root: {tmp_path}
  metadata_csv: {metadata}
validation:
  inspect_videos: false
""",
        encoding="utf-8",
    )
    config_path = tmp_path / f"multimodal_{fusion_type}.yaml"
    config_path.write_text(
        f"""
seed: 42
run:
  name: mm_smoke
  output_dir: {tmp_path / "runs"}
dataset:
  config: {dataset_config}
  cache_root: {cache_root}
sequence:
  mode: fixed
  length: 6
features:
  left_hand: true
  right_hand: true
  pose: true
  validity: true
video:
  num_frames: 4
  sampling: uniform
rgb:
  resize: 32
  crop_size: 32
  horizontal_flip: false
  random_crop: false
data_quality:
  minimum_valid_joint_ratio: 0.0
  reject_empty: true
model:
  type: multimodal
  fusion_type: {fusion_type}
  shared_dim: 16
  pretrained_model: tiny-offline-videomae
  pretrained: false
  freeze_rgb: true
  freeze_pose: false
  hidden_dim: 16
  embedding_dim: 16
  num_layers: 1
  num_heads: 4
  dropout: 0.0
training:
  epochs: 1
  batch_size: 2
  gradient_accumulation_steps: 1
  learning_rate: 0.001
  weight_decay: 0.0
  device: cpu
early_stopping:
  enabled: false
  patience: 2
""",
        encoding="utf-8",
    )
    return config_path


@pytest.mark.parametrize("fusion_type", ["concat", "gated", "cross_attention"])
def test_multimodal_forward_shapes(fusion_type: str) -> None:
    config = RecognitionTrainingConfig.model_validate(
        {
            "dataset": {"config": "configs/data/vsl400.yaml"},
            "model": {
                "type": "multimodal",
                "fusion_type": fusion_type,
                "pretrained": False,
                "pretrained_model": "tiny",
                "hidden_dim": 16,
                "embedding_dim": 16,
                "shared_dim": 16,
                "num_layers": 1,
                "num_heads": 4,
            },
        }
    )
    model = build_model(config, num_classes=2, input_spec=InputSpec(joint_count=55, channel_count=4))
    batch = {
        "pixel_values": torch.zeros((2, 4, 3, 32, 32)),
        "coordinates": torch.zeros((2, 6, 55, 4)),
        "valid_mask": torch.ones((2, 6, 55), dtype=torch.bool),
        "temporal_mask": torch.ones((2, 6), dtype=torch.bool),
    }
    logits = model(**batch)
    report = model.embedding_report(**batch)

    assert logits.shape == (2, 2)
    assert report["projected_rgb"] == [2, 16]
    assert report["projected_pose"] == [2, 16]
    assert report["fused"] == [2, 16]
    assert torch.isfinite(logits).all()


def test_multimodal_dataset_alignment_and_mismatch(tmp_path: Path) -> None:
    config_path = _project(tmp_path)
    config = RecognitionTrainingConfig.model_validate(json.loads(json.dumps(__import__("yaml").safe_load(config_path.read_text()))))
    samples = __import__("vietsign.training.dataset", fromlist=["load_samples_for_training"]).load_samples_for_training(config)
    dataset = MultimodalVSLDataset(samples[:1], cache_root=config.dataset.cache_root, label_to_id={"hello": 0, "thanks": 1}, config=config, training=False)

    item = dataset[0]
    assert item.sample_id == samples[0].sample_id
    assert item.modality_mask["rgb"].item() is True
    assert item.modality_mask["pose"].item() is True

    _write_cache(Path(config.dataset.cache_root), samples[0], label="wrong")
    with pytest.raises(MultimodalDatasetError, match="label mismatch"):
        _ = dataset[0]


def test_frozen_and_trainable_gradients() -> None:
    config = RecognitionTrainingConfig.model_validate(
        {
            "dataset": {"config": "configs/data/vsl400.yaml"},
            "model": {
                "type": "multimodal",
                "pretrained": False,
                "pretrained_model": "tiny",
                "freeze_rgb": True,
                "freeze_pose": False,
                "hidden_dim": 16,
                "embedding_dim": 16,
                "shared_dim": 16,
                "num_layers": 1,
                "num_heads": 4,
            },
        }
    )
    model = build_model(config, num_classes=2, input_spec=InputSpec(joint_count=55, channel_count=4))
    logits = model(
        pixel_values=torch.zeros((2, 4, 3, 32, 32)),
        coordinates=torch.zeros((2, 6, 55, 4)),
        valid_mask=torch.ones((2, 6, 55), dtype=torch.bool),
        temporal_mask=torch.ones((2, 6), dtype=torch.bool),
    )
    logits.sum().backward()

    assert all(parameter.grad is None for parameter in model.rgb_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.pose_encoder.parameters() if parameter.requires_grad)


def test_multimodal_train_evaluate_predict_smoke(tmp_path: Path) -> None:
    config_path = _project(tmp_path, fusion_type="concat")
    config = RecognitionTrainingConfig.model_validate(json.loads(json.dumps(__import__("yaml").safe_load(config_path.read_text()))))

    summary = train_model(config)
    checkpoint = Path(summary["best_checkpoint"])
    evaluation = evaluate_checkpoint(config, checkpoint_path=checkpoint)
    prediction = predict_multimodal(
        config,
        video_path=tmp_path / "a_train.mp4",
        keypoints_path=tmp_path / "cache" / "synthetic" / "a_train.npz",
        checkpoint_path=checkpoint,
    )

    assert checkpoint.exists()
    assert evaluation["metrics"]["top5_accuracy"] >= 0.0
    assert prediction["top_k"]
    assert model_metadata(build_model(config, num_classes=2, input_spec=InputSpec(55, 4)))["modality"] == "multimodal"


def test_multimodal_benchmark_registration(tmp_path: Path) -> None:
    config_path = _project(tmp_path, fusion_type="cross_attention")

    report = run_synthetic_smoke_benchmark([config_path], output_dir=tmp_path / "benchmark")

    assert "NON-VSL" in report["label"]
    assert report["results"][0]["model"] == "multimodal"
    assert report["results"][0]["modality"] == "multimodal"
    assert model_modality("multimodal") == "multimodal"
