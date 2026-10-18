from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from vietsign.data.voya_direct import (
    VOYADirectFeatureDataset,
    attach_sequence_hashes,
    deterministic_subset,
    duplicate_leakage_report,
    enforce_no_duplicate_hash_cross_split,
    fit_train_normalization,
    index_voya_npz_files,
    load_voya_label_names,
    select_eligible_labels,
    stratified_sample_split,
    summarize_voya_labels,
)
from vietsign.models.flat_features import FlatFeatureTemporalTransformerClassifier
from vietsign.training.checkpoint import save_checkpoint


def test_native_1605_contract_and_label_indexing(tmp_path: Path) -> None:
    root = _make_voya_fixture(tmp_path)
    samples = index_voya_npz_files(root)
    labels = load_voya_label_names(root / "labels.json")
    label_to_index = {1: 0, 2: 1}

    dataset = VOYADirectFeatureDataset(samples, label_to_index=label_to_index)
    item = dataset[0]

    assert labels[1] == "xin chào"
    assert item["features"].shape == (4, 1605)
    assert item["temporal_mask"].shape == (4,)
    assert item["label"].item() in {0, 1}
    assert item["provenance"]["dataset"] == "VOYA_VSL"


def test_lazy_npz_cache_and_multi_file_loading(tmp_path: Path) -> None:
    root = _make_voya_fixture(tmp_path)
    samples = index_voya_npz_files(root)
    dataset = VOYADirectFeatureDataset(samples, label_to_index={1: 0, 2: 1}, cache_size=1)

    _ = dataset[0]
    assert len(dataset.cache.cached_files) == 1
    _ = dataset[-1]
    assert len(dataset.cache.cached_files) == 1
    assert dataset[0]["source_file"] != dataset[-1]["source_file"]


def test_deterministic_subset_and_stratified_split(tmp_path: Path) -> None:
    root = _make_voya_fixture(tmp_path, samples_per_class=12)
    samples = index_voya_npz_files(root)
    eligible, report = select_eligible_labels(samples, minimum_samples_per_class=10, max_classes=2)
    subset = deterministic_subset(samples, labels=eligible, max_samples_per_class=10, seed=7)
    split_a = stratified_sample_split(subset, seed=7)
    split_b = stratified_sample_split(subset, seed=7)

    assert report["eligible_labels"] == [1, 2]
    assert [sample.sample_id + str(sample.split) for sample in split_a] == [
        sample.sample_id + str(sample.split) for sample in split_b
    ]
    assert {sample.split for sample in split_a} == {"train", "validation", "test"}


def test_duplicate_leakage_prevention_report(tmp_path: Path) -> None:
    root = _make_voya_fixture(tmp_path)
    samples = stratified_sample_split(index_voya_npz_files(root), seed=42)
    hashed = attach_sequence_hashes(samples)

    report = duplicate_leakage_report(hashed)

    assert report["passes"] is True
    assert report["cross_split_duplicate_count"] == 0


def test_duplicate_cross_split_enforcement(tmp_path: Path) -> None:
    root = _make_voya_fixture(tmp_path)
    samples = attach_sequence_hashes(stratified_sample_split(index_voya_npz_files(root), seed=42))
    duplicate_hash = samples[0].sequence_hash
    forced = [
        sample.__class__(**(sample.__dict__ | {"sequence_hash": duplicate_hash, "split": "train"}))
        if index == 0
        else sample.__class__(**(sample.__dict__ | {"sequence_hash": duplicate_hash, "split": "test"}))
        if index == 1
        else sample
        for index, sample in enumerate(samples)
    ]

    fixed = enforce_no_duplicate_hash_cross_split(forced)

    assert duplicate_leakage_report(fixed)["passes"] is True


def test_train_only_normalization_uses_train_split(tmp_path: Path) -> None:
    root = _make_voya_fixture(tmp_path, samples_per_class=10)
    samples = stratified_sample_split(index_voya_npz_files(root), seed=42)

    normalization = fit_train_normalization(samples)

    train_count = sum(1 for sample in samples if sample.split == "train")
    assert normalization["type"] == "per_feature_zscore"
    assert normalization["fit_split"] == "train"
    assert normalization["frame_count"] == train_count * 4
    assert len(normalization["mean"]) == 1605


def test_flat_feature_transformer_forward_shape() -> None:
    model = FlatFeatureTemporalTransformerClassifier(
        input_dim=1605,
        num_classes=3,
        embedding_dim=32,
        num_layers=1,
        num_heads=4,
    )

    logits = model(torch.zeros((2, 4, 1605)), torch.ones((2, 4), dtype=torch.bool))

    assert logits.shape == (2, 3)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_checkpoint_metadata_and_real_output_separation(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/training/voya_direct_baseline.yaml").read_text())
    model = FlatFeatureTemporalTransformerClassifier(input_dim=1605, num_classes=2)
    checkpoint = tmp_path / "best.pt"

    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=1,
        config=config,
        class_mapping={"1": 0, "2": 1},
        best_metric=0.5,
        model_metadata={
            "dataset": "VOYA_VSL",
            "real_data": True,
            "representation": "native_1605",
            "canonical_vietsign55": False,
            "signer_independent": False,
            "maturity": "experimental_real_data",
        },
    )

    assert checkpoint.exists()
    assert config["run"]["output_dir"].startswith("outputs/real_experiments/")
    assert "phase06-smoke" not in config["run"]["output_dir"]


def test_labels_json_semantics_fixture(tmp_path: Path) -> None:
    root = _make_voya_fixture(tmp_path)

    report = summarize_voya_labels(root / "labels.json")

    assert report["exact_number"] == 2
    assert report["key_format"] == "class_####"
    assert report["blank_labels"] == []


def _make_voya_fixture(tmp_path: Path, samples_per_class: int = 6) -> Path:
    root = tmp_path / "voya"
    root.mkdir()
    labels = {"class_0001": "xin chào", "class_0002": "cảm ơn"}
    (root / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    for class_id in (1, 2):
        rng = np.random.default_rng(class_id)
        sequences = rng.normal(size=(samples_per_class, 4, 1605)).astype(np.float32)
        sequences += class_id
        np.savez_compressed(
            root / f"class_{class_id:04d}.npz",
            sequences=sequences,
            labels=np.full((samples_per_class,), class_id, dtype=np.int32),
        )
    return root
