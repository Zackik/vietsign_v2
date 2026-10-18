from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vietsign.data.schema import SignSample
from vietsign.evaluation.metrics import compute_metrics, topk_predictions
from vietsign.models.temporal_transformer import TemporalTransformerClassifier, masked_mean_pooling
from vietsign.preprocessing.cache import save_keypoint_cache
from vietsign.training.checkpoint import (
    CheckpointCompatibilityError,
    load_checkpoint,
    save_checkpoint,
    validate_model_compatibility,
)
from vietsign.training.config import RecognitionTrainingConfig
from vietsign.training.dataset import CachedKeypointDataset
from vietsign.training.input import (
    EmptySequenceError,
    collate_keypoint_batch,
    prepare_keypoint_arrays,
)
from vietsign.training.runner import predict_cached_features, train_model
from vietsign.vision.constants import CANONICAL_JOINT_NAMES, SCHEMA_VERSION
from vietsign.vision.schema import KeypointTensors


def _arrays(t: int = 5, *, label_offset: float = 0.0, empty: bool = False) -> dict[str, np.ndarray]:
    coordinates = np.zeros((t, 55, 3), dtype=np.float32)
    valid_mask = np.zeros((t, 55), dtype=bool)
    if not empty:
        coordinates[:, 0, 0] = 1.0 + label_offset
        coordinates[:, 21, 1] = 0.5 + label_offset
        valid_mask[:, [0, 21, 42]] = True
    return {
        "coordinates": coordinates,
        "valid_mask": valid_mask,
        "confidence": np.full((t, 55), np.nan, dtype=np.float32),
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


def _synthetic_project(tmp_path: Path) -> tuple[Path, Path, Path, list[SignSample]]:
    cache_root = tmp_path / "cache"
    metadata = tmp_path / "metadata.csv"
    dataset_config = tmp_path / "dataset.yaml"
    output_dir = tmp_path / "runs"
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
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "video_path": str(sample.video_path),
                    "label": sample.label,
                    "label_id": sample.label_id,
                    "signer_id": sample.signer_id,
                    "split": sample.split,
                }
            )
            _write_cache(cache_root, sample, _arrays(label_offset=float(sample.label_id)))
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
    training_config = tmp_path / "training.yaml"
    training_config.write_text(
        f"""
seed: 42
run:
  name: smoke
  output_dir: {output_dir}
dataset:
  config: {dataset_config}
  cache_root: {cache_root}
sequence:
  mode: fixed
  length: 4
features:
  left_hand: true
  right_hand: true
  pose: true
  validity: true
  confidence: false
  source_mask: false
  velocity: true
data_quality:
  minimum_valid_joint_ratio: 0.0
  reject_empty: true
model:
  type: temporal_transformer
  embedding_dim: 16
  num_layers: 1
  num_heads: 4
  dropout: 0.0
training:
  epochs: 1
  batch_size: 2
  learning_rate: 0.001
  weight_decay: 0.0
  gradient_clip_norm: 1.0
  device: cpu
optimizer:
  type: adamw
scheduler:
  type: none
loss:
  class_weighting: none
early_stopping:
  enabled: false
  patience: 2
  metric: macro_f1
evaluation:
  fail_on_signer_overlap: false
  top_k: 5
""",
        encoding="utf-8",
    )
    return training_config, cache_root, output_dir, samples


def test_prepare_keypoint_arrays_contract_and_feature_groups() -> None:
    config = RecognitionTrainingConfig.model_validate(
        {"dataset": {"config": "configs/data/vsl400.yaml"}, "sequence": {"mode": "fixed", "length": 3}}
    )
    inputs, valid_mask, temporal_mask, ratio = prepare_keypoint_arrays(
        _arrays(t=5),
        feature_config=config.features,
        sequence_config=config.sequence,
        minimum_valid_joint_ratio=0.0,
    )

    assert inputs.shape == (3, 55, 4)
    assert valid_mask.shape == (3, 55)
    assert temporal_mask.shape == (3,)
    assert ratio > 0
    assert np.isfinite(inputs).all()


def test_empty_sequence_is_rejected() -> None:
    config = RecognitionTrainingConfig.model_validate(
        {"dataset": {"config": "configs/data/vsl400.yaml"}, "data_quality": {"minimum_valid_joint_ratio": 0.01}}
    )
    with pytest.raises(EmptySequenceError):
        prepare_keypoint_arrays(
            _arrays(empty=True),
            feature_config=config.features,
            sequence_config=config.sequence,
            minimum_valid_joint_ratio=config.data_quality.minimum_valid_joint_ratio,
            reject_empty=True,
        )


def test_cached_keypoint_dataset_and_collate(tmp_path: Path) -> None:
    sample = _sample("one", "hello", 0, "train")
    _write_cache(tmp_path, sample, _arrays(t=3))
    config = RecognitionTrainingConfig.model_validate(
        {"dataset": {"config": "configs/data/vsl400.yaml", "cache_root": tmp_path}, "sequence": {"mode": "variable"}}
    )
    dataset = CachedKeypointDataset([sample], cache_root=tmp_path, label_to_id={"hello": 0}, config=config)
    item = dataset[0]
    batch = collate_keypoint_batch([item, item])

    assert batch["coordinates"].shape == (2, 3, 55, 4)
    assert batch["temporal_mask"].all()
    assert batch["labels"].tolist() == [0, 0]


def test_variable_padding_temporal_masks() -> None:
    config = RecognitionTrainingConfig.model_validate(
        {"dataset": {"config": "configs/data/vsl400.yaml"}, "sequence": {"mode": "variable"}}
    )
    first = prepare_keypoint_arrays(_arrays(t=3), feature_config=config.features, sequence_config=config.sequence)
    second = prepare_keypoint_arrays(_arrays(t=5), feature_config=config.features, sequence_config=config.sequence)
    samples = [
        torch_sample("a", 0, first),
        torch_sample("b", 1, second),
    ]
    batch = collate_keypoint_batch(samples)

    assert batch["coordinates"].shape == (2, 5, 55, 4)
    assert batch["temporal_mask"][0].tolist() == [True, True, True, False, False]
    assert batch["temporal_mask"][1].tolist() == [True, True, True, True, True]


def torch_sample(sample_id: str, label: int, prepared: tuple[np.ndarray, np.ndarray, np.ndarray, float]):
    from vietsign.training.input import PreparedSample

    inputs, valid_mask, temporal_mask, ratio = prepared
    return PreparedSample(
        sample_id=sample_id,
        label=label,
        label_name=str(label),
        inputs=torch.from_numpy(inputs),
        valid_mask=torch.from_numpy(valid_mask),
        temporal_mask=torch.from_numpy(temporal_mask),
        valid_joint_ratio=ratio,
        metadata={},
    )


def test_model_forward_output_shape_and_no_nans() -> None:
    model = TemporalTransformerClassifier(
        joint_count=55,
        input_channels=4,
        num_classes=3,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    logits = model(
        torch.zeros((2, 4, 55, 4)),
        torch.zeros((2, 4, 55), dtype=torch.bool),
        torch.ones((2, 4), dtype=torch.bool),
    )

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()


def test_padding_invariance_eval_mode() -> None:
    torch.manual_seed(0)
    model = TemporalTransformerClassifier(
        joint_count=55,
        input_channels=4,
        num_classes=2,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    model.eval()
    coordinates = torch.randn((1, 3, 55, 4))
    valid = torch.ones((1, 3, 55), dtype=torch.bool)
    with torch.no_grad():
        base = model(coordinates, valid, torch.ones((1, 3), dtype=torch.bool))
        padded = model(
            torch.cat([coordinates, torch.randn((1, 2, 55, 4))], dim=1),
            torch.cat([valid, torch.zeros((1, 2, 55), dtype=torch.bool)], dim=1),
            torch.tensor([[True, True, True, False, False]]),
        )

    assert torch.allclose(base, padded, atol=1e-5)


def test_masked_pooling_ignores_padding() -> None:
    values = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    pooled = masked_mean_pooling(values, torch.tensor([[True, True, False]]))

    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


def test_metrics_and_topk() -> None:
    logits = torch.tensor([[5.0, 1.0], [0.5, 2.0]])
    labels = torch.tensor([0, 1])
    metrics = compute_metrics(logits, labels, class_names=["hello", "thanks"], top_k=5)
    top = topk_predictions(logits, id_to_label={0: "hello", 1: "thanks"}, k=5)

    assert metrics.top1_accuracy == 1.0
    assert metrics.top5_accuracy == 1.0
    assert top[0][0]["label"] == "hello"


def test_checkpoint_save_load_and_compatibility(tmp_path: Path) -> None:
    model = TemporalTransformerClassifier(
        joint_count=55,
        input_channels=4,
        num_classes=2,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
    )
    metadata = model.introspect().__dict__
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=0,
        config={},
        class_mapping={"hello": 0, "thanks": 1},
        best_metric=0.0,
        model_metadata=metadata,
    )
    checkpoint = load_checkpoint(path)
    validate_model_compatibility(checkpoint, metadata)
    with pytest.raises(CheckpointCompatibilityError):
        validate_model_compatibility(checkpoint, {**metadata, "input_channels": 99})


def test_training_smoke_evaluate_predict_and_cli(tmp_path: Path) -> None:
    config_path, _cache_root, _output_dir, samples = _synthetic_project(tmp_path)
    config = RecognitionTrainingConfig.model_validate(
        json.loads(json.dumps(__import__("yaml").safe_load(config_path.read_text())))
    )
    summary = train_model(config)
    best = Path(summary["best_checkpoint"])
    assert best.exists()
    assert (Path(summary["run_dir"]) / "config.yaml").exists()
    checkpoint = load_checkpoint(best)
    assert checkpoint["class_mapping"] == {"hello": 0, "thanks": 1}

    feature = tmp_path / "cache" / "synthetic" / f"{samples[0].sample_id}.npz"
    prediction = predict_cached_features(config, features_path=feature, checkpoint_path=best)
    assert prediction["label"] in {"hello", "thanks"}
    assert 0.0 <= prediction["confidence"] <= 1.0

    eval_result = subprocess.run(
        [sys.executable, "-m", "vietsign.cli", "evaluate", "--config", str(config_path), "--checkpoint", str(best)],
        check=True,
        capture_output=True,
        text=True,
    )
    eval_payload = json.loads(eval_result.stdout)
    assert eval_payload["metrics"]["top5_accuracy"] >= 0.0
    assert Path(eval_payload["confusion_matrix_png"]).exists()

    predict_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vietsign.cli",
            "predict",
            "--config",
            str(config_path),
            "--features",
            str(feature),
            "--checkpoint",
            str(best),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(predict_result.stdout)["top_k"]
