from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2  # type: ignore
import numpy as np
import pytest
import torch

from vietsign.models.registry import InputSpec, build_model, model_metadata, model_modality
from vietsign.models.videomae import RGBEmbeddingClassifier, VideoMAEClassifier
from vietsign.training.benchmark import run_synthetic_smoke_benchmark
from vietsign.training.config import RecognitionTrainingConfig
from vietsign.training.rgb import (
    RGBVideoDataset,
    collate_rgb_batch,
    load_rgb_clip,
    load_rgb_embedding_cache,
    rgb_cache_fingerprint,
    save_rgb_embedding_cache,
    select_frame_indices,
)
from vietsign.training.runner import evaluate_checkpoint, predict_video, train_model


def _write_video(path: Path, *, color: tuple[int, int, int] = (80, 120, 160), frames: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (64, 64))
    assert writer.isOpened()
    try:
        for index in range(frames):
            frame = np.full((64, 64, 3), color, dtype=np.uint8)
            cv2.putText(frame, str(index), (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(frame)
    finally:
        writer.release()


def _rgb_project(tmp_path: Path) -> Path:
    metadata = tmp_path / "metadata.csv"
    dataset_config = tmp_path / "dataset.yaml"
    rows = [
        ("a_train", "hello", 0, "train", (80, 20, 120)),
        ("b_train", "thanks", 1, "train", (20, 180, 80)),
        ("a_val", "hello", 0, "validation", (90, 30, 130)),
        ("b_val", "thanks", 1, "validation", (30, 190, 90)),
    ]
    with metadata.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample_id", "video_path", "label", "label_id", "signer_id", "split"],
        )
        writer.writeheader()
        for sample_id, label, label_id, split, color in rows:
            video_path = tmp_path / f"{sample_id}.mp4"
            _write_video(video_path, color=color)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "video_path": str(video_path),
                    "label": label,
                    "label_id": label_id,
                    "signer_id": f"signer_{sample_id}",
                    "split": split,
                }
            )
    dataset_config.write_text(
        f"""
dataset:
  name: synthetic_rgb
  adapter: custom_csv
  root: {tmp_path}
  metadata_csv: {metadata}
validation:
  inspect_videos: false
""",
        encoding="utf-8",
    )
    training_config = tmp_path / "videomae.yaml"
    training_config.write_text(
        f"""
seed: 42
run:
  name: rgb_smoke
  output_dir: {tmp_path / "runs"}
dataset:
  config: {dataset_config}
video:
  num_frames: 4
  sampling: uniform
rgb:
  resize: 32
  crop_size: 32
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  horizontal_flip: false
  random_crop: false
  cache_root: {tmp_path / "rgb_cache"}
model:
  type: videomae
  pretrained_model: tiny-offline-videomae
  pretrained: false
  freeze_backbone: true
  hidden_dim: 16
  dropout: 0.0
training:
  epochs: 1
  batch_size: 2
  gradient_accumulation_steps: 2
  learning_rate: 0.001
  weight_decay: 0.0
  gradient_clip_norm: 1.0
  mixed_precision: false
  device: cpu
early_stopping:
  enabled: false
  patience: 2
  metric: macro_f1
""",
        encoding="utf-8",
    )
    return training_config


def test_rgb_tensor_contract_and_deterministic_sampling(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    _write_video(video)
    config = RecognitionTrainingConfig.model_validate(
        {
            "dataset": {"config": "configs/data/vsl400.yaml"},
            "video": {"num_frames": 4, "sampling": "uniform"},
            "rgb": {"resize": 32, "crop_size": 32},
        }
    )

    clip, metadata = load_rgb_clip(video, video_config=config.video, preprocessing=config.rgb, training=False)

    assert clip.shape == (4, 3, 32, 32)
    assert metadata["selected_frame_indices"] == [0, 2, 4, 7]
    assert metadata["horizontal_flip_enabled"] is False


def test_random_sampling_only_for_training() -> None:
    eval_indices = select_frame_indices(10, num_frames=4, sampling="random", training=False)
    train_indices = select_frame_indices(10, num_frames=4, sampling="random", training=True)

    assert eval_indices.tolist() == [0, 3, 6, 9]
    assert len(train_indices) == 4
    assert np.diff(train_indices).tolist() == [1, 1, 1]


def test_videomae_adapter_frozen_backbone_and_counts() -> None:
    model = VideoMAEClassifier(
        num_classes=3,
        pretrained_model="tiny-offline-videomae",
        pretrained=False,
        freeze_backbone=True,
        hidden_dim=16,
    )
    logits = model(torch.zeros((2, 4, 3, 32, 32)))
    metadata = model.introspect()

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert metadata.modality == "rgb"
    assert metadata.trainable_parameter_count < metadata.parameter_count


def test_rgb_embedding_cache_fingerprint_and_reload(tmp_path: Path) -> None:
    config = RecognitionTrainingConfig.model_validate(
        {
            "dataset": {"config": "configs/data/vsl400.yaml"},
            "model": {"type": "videomae", "pretrained": False, "pretrained_model": "tiny"},
        }
    )
    changed = config.model_copy(deep=True)
    changed.video.num_frames = 8
    sample = __import__("vietsign.data.schema", fromlist=["SignSample"]).SignSample(
        sample_id="one",
        video_path=Path("one.mp4"),
        label="hello",
        label_id=0,
        split="train",
        dataset="synthetic",
    )
    npz_path = tmp_path / "one.npz"
    json_path = tmp_path / "one.json"
    fingerprint = rgb_cache_fingerprint(config)
    save_rgb_embedding_cache(
        npz_path,
        json_path,
        embedding=np.ones((16,), dtype=np.float32),
        sample=sample,
        metadata={"fingerprint": fingerprint},
    )
    embedding, metadata = load_rgb_embedding_cache(npz_path, json_path, expected_fingerprint=fingerprint)

    assert embedding.shape == (16,)
    assert metadata["schema_version"] == "rgb_embeddings.v1"
    assert rgb_cache_fingerprint(config) != rgb_cache_fingerprint(changed)
    with pytest.raises(ValueError, match="fingerprint"):
        load_rgb_embedding_cache(npz_path, json_path, expected_fingerprint=rgb_cache_fingerprint(changed))


def test_rgb_dataset_collate_and_registry(tmp_path: Path) -> None:
    config_path = _rgb_project(tmp_path)
    config = RecognitionTrainingConfig.model_validate(json.loads(json.dumps(__import__("yaml").safe_load(config_path.read_text()))))
    samples = __import__("vietsign.training.dataset", fromlist=["load_samples_for_training"]).load_samples_for_training(config)
    dataset = RGBVideoDataset(samples[:1], label_to_id={"hello": 0, "thanks": 1}, config=config, training=False)
    batch = collate_rgb_batch([dataset[0]])
    model = build_model(config, num_classes=2, input_spec=InputSpec(0, 3))

    assert batch["pixel_values"].shape == (1, 4, 3, 32, 32)
    assert model_modality(config.model.type) == "rgb"
    assert model_metadata(model)["model_name"] == "videomae"


def test_rgb_training_evaluate_predict_smoke(tmp_path: Path) -> None:
    config_path = _rgb_project(tmp_path)
    config = RecognitionTrainingConfig.model_validate(json.loads(json.dumps(__import__("yaml").safe_load(config_path.read_text()))))

    summary = train_model(config)
    checkpoint = Path(summary["best_checkpoint"])
    evaluation = evaluate_checkpoint(config, checkpoint_path=checkpoint)
    prediction = predict_video(config, video_path=tmp_path / "a_train.mp4", checkpoint_path=checkpoint)

    assert checkpoint.exists()
    assert evaluation["metrics"]["top5_accuracy"] >= 0.0
    assert prediction["top_k"]


def test_rgb_embedding_classifier_forward() -> None:
    classifier = RGBEmbeddingClassifier(embedding_dim=16, num_classes=2)
    logits = classifier(torch.zeros((3, 16)))

    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_synthetic_benchmark_can_represent_videomae(tmp_path: Path) -> None:
    config_path = _rgb_project(tmp_path)

    report = run_synthetic_smoke_benchmark([config_path], output_dir=tmp_path / "benchmark")

    assert "NON-VSL" in report["label"]
    assert report["results"][0]["model"] == "videomae"
    assert report["results"][0]["modality"] == "rgb"
