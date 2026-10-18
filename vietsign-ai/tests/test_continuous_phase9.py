from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vietsign.continuous.decoding import CTCGreedyDecoder
from vietsign.continuous.metrics import aggregate_gloss_metrics, gloss_edit_stats
from vietsign.continuous.vocabulary import GlossVocabulary
from vietsign.data.schema import ContinuousSignSample, SignSample
from vietsign.models.continuous import ContinuousSignRecognizer, validate_ctc_lengths
from vietsign.preprocessing.cache import save_keypoint_cache
from vietsign.training.config import RecognitionTrainingConfig
from vietsign.training.continuous import (
    ContinuousKeypointDataset,
    collate_continuous_batch,
    load_continuous_samples,
)
from vietsign.training.runner import evaluate_checkpoint, predict_continuous, train_model
from vietsign.vision.constants import CANONICAL_JOINT_NAMES, SCHEMA_VERSION
from vietsign.vision.schema import KeypointTensors


def _arrays(t: int = 8, *, empty: bool = False) -> dict[str, np.ndarray]:
    coordinates = np.zeros((t, 55, 3), dtype=np.float32)
    valid_mask = np.zeros((t, 55), dtype=bool)
    if not empty:
        coordinates[:, 0, 0] = np.linspace(0.0, 1.0, t)
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


def _write_cache(root: Path, sample_id: str, glosses: list[str], *, split: str) -> None:
    arrays = _arrays()
    sample = SignSample(
        sample_id=sample_id,
        video_path=Path(f"{sample_id}.mp4"),
        label=" ".join(glosses),
        label_id=0,
        split=split,
        dataset="continuous",
        signer_id=f"signer_{sample_id}",
    )
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
        metadata={"schema_version": SCHEMA_VERSION, "fingerprint": None, "coverage": {"total": 1.0}},
    )


def _continuous_project(tmp_path: Path) -> Path:
    metadata = tmp_path / "continuous.csv"
    dataset_config = tmp_path / "continuous.yaml"
    cache_root = tmp_path / "cache"
    rows = [
        ("s1_train", "TOI MUON", "train"),
        ("s2_train", "TOI TOI NUOC", "train"),
        ("s1_val", "TOI MUON", "validation"),
        ("s2_val", "TOI TOI NUOC", "validation"),
    ]
    with metadata.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_id", "video_path", "glosses", "signer_id", "split"])
        writer.writeheader()
        for sample_id, gloss_text, split in rows:
            glosses = gloss_text.split()
            _write_cache(cache_root, sample_id, glosses, split=split)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "video_path": f"{sample_id}.mp4",
                    "glosses": gloss_text,
                    "signer_id": f"signer_{sample_id}",
                    "split": split,
                }
            )
    dataset_config.write_text(
        f"""
dataset:
  name: continuous
  adapter: custom_csv
  root: {tmp_path}
  metadata_csv: {metadata}
validation:
  inspect_videos: false
""",
        encoding="utf-8",
    )
    config = tmp_path / "continuous_ctc.yaml"
    config.write_text(
        f"""
seed: 42
task:
  type: continuous_recognition
run:
  name: cslr_smoke
  output_dir: {tmp_path / "runs"}
dataset:
  config: {dataset_config}
  cache_root: {cache_root}
sequence:
  mode: variable
features:
  left_hand: true
  right_hand: true
  pose: true
  validity: true
data_quality:
  minimum_valid_joint_ratio: 0.0
  reject_empty: true
model:
  type: ctc_recognizer
  embedding_dim: 16
  num_layers: 1
  num_heads: 4
  dropout: 0.0
ctc:
  blank_token: "<blank>"
  zero_infinity: true
training:
  epochs: 1
  batch_size: 2
  learning_rate: 0.001
  weight_decay: 0.0
  gradient_clip_norm: 1.0
  device: cpu
early_stopping:
  enabled: false
  patience: 2
  metric: loss
""",
        encoding="utf-8",
    )
    return config


def test_gloss_vocabulary_blank_and_continuous_schema() -> None:
    vocab = GlossVocabulary.build([["TOI", "MUON"], ["NUOC"]])
    sample = ContinuousSignSample(
        sample_id="one",
        video_path=Path("one.mp4"),
        glosses=["TOI", "MUON"],
        split="train",
        dataset="continuous",
    )

    assert vocab.blank_index == 0
    assert vocab.id_to_token[0] == "<blank>"
    assert vocab.decode(vocab.encode(["TOI"])) == ["TOI"]
    assert sample.glosses == ["TOI", "MUON"]


def test_ctc_greedy_decoder_duplicate_collapse_and_repeated_target() -> None:
    vocab = GlossVocabulary(id_to_token=("<blank>", "A", "B"))
    logits = torch.full((1, 7, 3), -10.0)
    path = [0, 1, 1, 0, 1, 0, 2]
    for t, token in enumerate(path):
        logits[0, t, token] = 10.0

    decoded = CTCGreedyDecoder(vocab).decode(logits)

    assert decoded == [["A", "A", "B"]]


def test_ger_edit_distance_counts() -> None:
    stats = gloss_edit_stats(["TOI", "MUON", "NUOC"], ["TOI", "CAN", "NUOC", "DI"])
    aggregate = aggregate_gloss_metrics([(["A"], ["A"]), (["A", "B"], ["B"])])

    assert stats.substitutions == 1
    assert stats.insertions == 1
    assert stats.deletions == 0
    assert stats.ger == pytest.approx(2 / 3)
    assert aggregate["gloss_error_rate"] >= 0.0


def test_ctc_length_validation() -> None:
    with pytest.raises(ValueError, match="input_length"):
        validate_ctc_lengths(torch.tensor([2]), torch.tensor([3]))


def test_continuous_dataset_and_collate(tmp_path: Path) -> None:
    config_path = _continuous_project(tmp_path)
    config = RecognitionTrainingConfig.model_validate(json.loads(json.dumps(__import__("yaml").safe_load(config_path.read_text()))))
    samples = load_continuous_samples(__import__("yaml").safe_load(Path(config.dataset.config).read_text())["dataset"]["metadata_csv"])
    vocab = GlossVocabulary.build([sample.glosses for sample in samples])
    dataset = ContinuousKeypointDataset(samples[:2], cache_root=config.dataset.cache_root, vocabulary=vocab, config=config)
    batch = collate_continuous_batch([dataset[0], dataset[1]])

    assert batch["coordinates"].shape[0] == 2
    assert batch["targets"].ndim == 1
    assert batch["target_lengths"].tolist() == [2, 3]


def test_continuous_model_ctc_loss_and_shapes() -> None:
    model = ContinuousSignRecognizer(joint_count=55, input_channels=4, vocab_size=4, embedding_dim=16, num_layers=1, num_heads=4)
    coordinates = torch.zeros((2, 8, 55, 4))
    valid_mask = torch.ones((2, 8, 55), dtype=torch.bool)
    temporal_mask = torch.ones((2, 8), dtype=torch.bool)
    targets = torch.tensor([1, 2, 1, 1, 3], dtype=torch.long)
    target_lengths = torch.tensor([2, 3], dtype=torch.long)
    logits = model(coordinates, valid_mask, temporal_mask)
    loss = model.ctc_loss(coordinates, valid_mask, temporal_mask, targets, target_lengths)

    assert logits.shape == (2, 8, 4)
    assert torch.isfinite(torch.log_softmax(logits, dim=-1)).all()
    assert torch.isfinite(loss)


def test_continuous_train_evaluate_predict_smoke(tmp_path: Path) -> None:
    config_path = _continuous_project(tmp_path)
    config = RecognitionTrainingConfig.model_validate(json.loads(json.dumps(__import__("yaml").safe_load(config_path.read_text()))))

    summary = train_model(config)
    checkpoint = Path(summary["best_checkpoint"])
    evaluation = evaluate_checkpoint(config, checkpoint_path=checkpoint)
    prediction = predict_continuous(
        features_path=tmp_path / "cache" / "continuous" / "s1_train.npz",
        checkpoint_path=checkpoint,
        config=config,
    )

    assert checkpoint.exists()
    assert evaluation["metrics"]["gloss_error_rate"] >= 0.0
    assert "gloss_sequence" in prediction
    assert (checkpoint.parent / "evaluation.json").exists()
    assert (checkpoint.parent / "predictions.csv").exists()
