from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
import yaml

from vietsign.data.schema import TranslationSample
from vietsign.models.translation import GlossToVietnameseTransformer
from vietsign.training.config import RecognitionTrainingConfig
from vietsign.training.runner import evaluate_checkpoint, train_model, translate_gloss
from vietsign.translation.dataset import (
    GlossTranslationDataset,
    build_translation_vocabularies,
    collate_translation_batch,
    load_translation_samples,
)
from vietsign.translation.metrics import (
    aggregate_translation_metrics,
    chrf,
    exact_match,
    sentence_bleu,
)
from vietsign.translation.normalization import normalize_gloss_sequence, normalize_vietnamese_text
from vietsign.translation.vocabulary import TextVocabulary


def _write_translation_project(tmp_path: Path) -> Path:
    metadata = tmp_path / "translation.csv"
    rows = [
        ("tr1", "TOI", "Tôi.", "train"),
        ("tr2", "TOI UONG NUOC", "Tôi uống nước.", "train"),
        ("tr3", "CAM-ON", "Cảm ơn.", "train"),
        ("val1", "TOI UONG NUOC", "Tôi uống nước.", "validation"),
        ("val2", "CAM-ON", "Cảm ơn.", "validation"),
    ]
    with metadata.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_id", "gloss_sequence", "vietnamese_text", "split"])
        writer.writeheader()
        for sample_id, gloss, target, split in rows:
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "gloss_sequence": gloss,
                    "vietnamese_text": target,
                    "split": split,
                }
            )
    dataset_config = tmp_path / "translation.yaml"
    dataset_config.write_text(
        f"""
dataset:
  name: translation_synthetic
  adapter: custom_csv
  root: {tmp_path}
  metadata_csv: {metadata}
validation:
  inspect_videos: false
""",
        encoding="utf-8",
    )
    config = tmp_path / "gloss_translation.yaml"
    config.write_text(
        f"""
seed: 42
task:
  type: gloss_translation
run:
  name: translation_smoke
  output_dir: {tmp_path / "runs"}
dataset:
  config: {dataset_config}
model:
  type: gloss_transformer
  d_model: 32
  num_heads: 4
  encoder_layers: 1
  decoder_layers: 1
  dim_feedforward: 64
  dropout: 0.0
translation:
  max_source_length: 8
  max_target_length: 8
decoding:
  type: greedy
  max_new_tokens: 8
training:
  epochs: 2
  batch_size: 2
  learning_rate: 0.003
  weight_decay: 0.0
  gradient_clip_norm: 1.0
  device: cpu
early_stopping:
  enabled: false
  patience: 2
  metric: chrf
""",
        encoding="utf-8",
    )
    return config


def test_translation_schema_and_unicode_normalization() -> None:
    sample = TranslationSample(
        sample_id="one",
        glosses=["TOI", "UONG", "NUOC"],
        source_text="TOI UONG NUOC",
        target_text="Tôi uống nước.",
        split="train",
        dataset="synthetic",
    )

    assert sample.glosses == ["TOI", "UONG", "NUOC"]
    assert normalize_vietnamese_text("Tôi   uống   nước .") == "Tôi uống nước."
    assert normalize_gloss_sequence("toi   uong nuoc") == ["TOI", "UONG", "NUOC"]


def test_translation_vocabularies_do_not_use_ctc_blank() -> None:
    vocab = TextVocabulary.build([["TOI"], ["Tôi."]])

    assert vocab.id_to_token[:4] == ("<PAD>", "<BOS>", "<EOS>", "<UNK>")
    assert "<blank>" not in vocab.token_to_id
    assert vocab.decode(vocab.encode(["MISSING"], add_bos=True, add_eos=True)) == []


def test_translation_dataset_and_collate(tmp_path: Path) -> None:
    config_path = _write_translation_project(tmp_path)
    config = RecognitionTrainingConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    dataset_yaml = yaml.safe_load(Path(config.dataset.config).read_text(encoding="utf-8"))
    samples = load_translation_samples(dataset_yaml["dataset"]["metadata_csv"], dataset=dataset_yaml["dataset"]["name"])
    source_vocab, target_vocab = build_translation_vocabularies(samples)
    dataset = GlossTranslationDataset(
        samples[:2],
        source_vocabulary=source_vocab,
        target_vocabulary=target_vocab,
        max_source_length=8,
        max_target_length=8,
    )
    batch = collate_translation_batch([dataset[0], dataset[1]])

    assert batch["source_input_ids"].shape[0] == 2
    assert batch["target_input_ids"].shape == batch["target_labels"].shape
    assert batch["source_attention_mask"].dtype == torch.bool


def test_translation_metrics_known_cases() -> None:
    assert exact_match("Tôi uống nước.", "Tôi uống nước.") == 1.0
    assert sentence_bleu("Tôi uống nước.", "Tôi uống nước.") > 0.99
    assert chrf("Cảm ơn.", "Cảm ơn.") > 0.99
    metrics = aggregate_translation_metrics([("Tôi uống nước.", "Tôi uống nước."), ("Cảm ơn.", "Cảm.")])

    assert 0.0 <= metrics["bleu"] <= 1.0
    assert 0.0 <= metrics["chrf"] <= 1.0
    assert metrics["exact_match"] == pytest.approx(0.5)


def test_gloss_transformer_forward_loss_and_greedy_decode() -> None:
    model = GlossToVietnameseTransformer(
        source_vocab_size=8,
        target_vocab_size=9,
        pad_index=0,
        bos_index=1,
        eos_index=2,
        d_model=16,
        num_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        max_length=12,
    )
    source = torch.tensor([[4, 5, 0], [6, 0, 0]])
    target_in = torch.tensor([[1, 4, 5], [1, 6, 0]])
    labels = torch.tensor([[4, 5, 2], [6, 2, 0]])
    logits = model(source, target_in, source_attention_mask=source.ne(0), target_attention_mask=target_in.ne(0))
    loss = model.loss(source, target_in, labels, source_attention_mask=source.ne(0), target_attention_mask=target_in.ne(0))
    generated = model.greedy_decode(source, source_attention_mask=source.ne(0), max_new_tokens=3)

    assert logits.shape == (2, 3, 9)
    assert torch.isfinite(loss)
    assert generated.shape[0] == 2


def test_translation_train_evaluate_translate_smoke(tmp_path: Path) -> None:
    config_path = _write_translation_project(tmp_path)
    config = RecognitionTrainingConfig.model_validate(json.loads(json.dumps(yaml.safe_load(config_path.read_text()))))

    summary = train_model(config)
    checkpoint = Path(summary["best_checkpoint"])
    evaluation = evaluate_checkpoint(config, checkpoint_path=checkpoint)
    prediction = translate_gloss(gloss="TOI UONG NUOC", checkpoint_path=checkpoint, config=config)

    assert checkpoint.exists()
    assert evaluation["task"] == "gloss_translation"
    assert "bleu" in evaluation["metrics"]
    assert prediction["source_token_ids"]
    assert "translation" in prediction
    assert (checkpoint.parent / "evaluation.json").exists()
    assert (checkpoint.parent / "translations.csv").exists()
