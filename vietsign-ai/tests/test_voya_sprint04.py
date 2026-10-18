from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vietsign.data.voya_direct import (
    analyze_voya_label_catalog,
    attach_sequence_hashes,
    deterministic_subset,
    index_voya_npz_files,
    normalize_label_for_analysis,
    strip_parenthetical_markers,
    transform_temporal_features,
)
from vietsign.training.voya_sprint04 import (
    bootstrap_metric_ci,
    exact_duplicate_audit,
    shuffle_training_labels,
)


def test_label_normalization_analysis_groups_regional_variants(tmp_path: Path) -> None:
    labels = {
        "class_0001": "Địa chỉ (Bắc)",
        "class_0002": "địa chỉ (nam)",
        "class_0003": "địa chỉ (trung)",
        "class_0004": "tỉnh",
    }
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")

    report = analyze_voya_label_catalog(path, json_path=tmp_path / "analysis.json", csv_path=tmp_path / "groups.csv")

    assert normalize_label_for_analysis(" Địa   chỉ (Bắc) ") == "địa chỉ (bắc)"
    assert strip_parenthetical_markers("địa chỉ (bắc)") == "địa chỉ"
    assert report["label_catalog_size"] == 4
    assert report["same_root_group_count"] == 1
    assert "regional_variants" in report["class_selection_strategies"]
    assert (tmp_path / "groups.csv").exists()


def test_temporal_shuffle_and_single_frame_views_are_deterministic() -> None:
    features = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)

    shuffled_a = transform_temporal_features(features, view="temporal_shuffle", sample_id="a", temporal_shuffle_seed=7)
    shuffled_b = transform_temporal_features(features, view="temporal_shuffle", sample_id="a", temporal_shuffle_seed=7)
    single = transform_temporal_features(features, view="single_middle_frame", sample_id="a")

    assert np.array_equal(shuffled_a, shuffled_b)
    assert sorted(shuffled_a[:, 0].tolist()) == sorted(features[:, 0].tolist())
    assert np.array_equal(single, np.repeat(features[2:3], 5, axis=0))


def test_sample_cap_and_label_shuffle_fixture(tmp_path: Path) -> None:
    root = _make_fixture(tmp_path)
    samples = index_voya_npz_files(root)
    selected = deterministic_subset(samples, labels=[1, 2], max_samples_per_class=3, seed=42)
    shuffled = shuffle_training_labels(selected, labels=[1, 2], seed=42)

    assert len(selected) == 6
    assert {sample.raw_label for sample in shuffled}.issubset({1, 2})


def test_duplicate_grouping_and_bootstrap_ci(tmp_path: Path) -> None:
    root = _make_fixture(tmp_path)
    samples = attach_sequence_hashes(index_voya_npz_files(root))
    audit = exact_duplicate_audit(samples)
    ci = bootstrap_metric_ci([[3, 1], [1, 3]], seed=42, iterations=20)

    assert audit["duplicate_group_count"] >= 1
    assert "top1" in ci
    assert "macro_f1" in ci


def _make_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "voya"
    root.mkdir()
    (root / "labels.json").write_text(
        json.dumps({"class_0001": "a (bắc)", "class_0002": "a (nam)"}, ensure_ascii=False),
        encoding="utf-8",
    )
    duplicate = np.zeros((4, 1605), dtype=np.float32)
    for class_id in (1, 2):
        rng = np.random.default_rng(class_id)
        sequences = rng.normal(size=(6, 4, 1605)).astype(np.float32)
        sequences[0] = duplicate
        np.savez_compressed(
            root / f"class_{class_id:04d}.npz",
            sequences=sequences,
            labels=np.full((6,), class_id, dtype=np.int32),
        )
    return root
