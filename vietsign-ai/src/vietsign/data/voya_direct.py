"""Native VOYA [T,1605] direct-feature dataset utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from vietsign.data.voya import VOYA_REVISION, normalize_class_name


@dataclass(frozen=True)
class VOYAFeatureSequence:
    """Native VOYA feature contract; these are not canonical keypoints."""

    features: np.ndarray
    temporal_mask: np.ndarray
    label: int
    sample_id: str
    source_file: str
    source_index: int
    provenance: dict[str, Any]


@dataclass(frozen=True)
class VOYASampleIndex:
    sample_id: str
    source_file: str
    source_index: int
    raw_label: int
    normalized_label: int | None
    label_name: str
    split: str | None = None
    sequence_hash: str | None = None


class VOYANPZCache:
    """Small LRU cache for NPZ sequence arrays, keeping file-level loading explicit."""

    def __init__(self, max_open_files: int = 1) -> None:
        self.max_open_files = max(1, max_open_files)
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def arrays(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        key = str(path)
        if key not in self._cache:
            with np.load(path, allow_pickle=False) as data:
                arrays = (np.asarray(data["sequences"]), np.asarray(data["labels"]))
            self._cache[key] = arrays
            while len(self._cache) > self.max_open_files:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    @property
    def cached_files(self) -> list[str]:
        return list(self._cache)


class VOYADirectFeatureDataset(Dataset):
    """PyTorch dataset for VOYA native [T,1605] features with lazy file loading."""

    def __init__(
        self,
        samples: list[VOYASampleIndex],
        *,
        label_to_index: dict[int, int],
        normalization: dict[str, Any] | None = None,
        cache_size: int = 1,
        view: str = "sequence",
        temporal_shuffle_seed: int | None = None,
    ) -> None:
        self.samples = list(samples)
        self.label_to_index = dict(label_to_index)
        self.normalization = normalization
        self.cache = VOYANPZCache(max_open_files=cache_size)
        self.view = view
        self.temporal_shuffle_seed = temporal_shuffle_seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        sequences, _labels = self.cache.arrays(sample.source_file)
        features = sequences[sample.source_index].astype(np.float32, copy=False)
        features = transform_temporal_features(
            features,
            view=self.view,
            sample_id=sample.sample_id,
            temporal_shuffle_seed=self.temporal_shuffle_seed,
        )
        temporal_mask = np.isfinite(features).all(axis=1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = apply_normalization(features, self.normalization)
        target = self.label_to_index[sample.raw_label]
        return {
            "features": torch.from_numpy(features.astype(np.float32, copy=False)),
            "temporal_mask": torch.from_numpy(temporal_mask.astype(bool, copy=False)),
            "label": torch.tensor(target, dtype=torch.long),
            "sample_id": sample.sample_id,
            "source_file": sample.source_file,
            "source_index": sample.source_index,
            "raw_label": sample.raw_label,
            "normalized_label": target,
            "label_name": sample.label_name,
            "split": sample.split,
            "provenance": {
                "dataset": "VOYA_VSL",
                "remote_revision": VOYA_REVISION,
                "source_file": sample.source_file,
                "source_index": sample.source_index,
                "raw_label": sample.raw_label,
                "normalized_label": target,
            },
        }


def collate_voya_direct(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "features": torch.stack([item["features"] for item in batch]),
        "temporal_mask": torch.stack([item["temporal_mask"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
        "sample_ids": [item["sample_id"] for item in batch],
        "provenance": [item["provenance"] for item in batch],
    }


def load_voya_label_names(labels_path: str | Path) -> dict[int, str]:
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    return {int(normalize_class_name(key).split("_")[1]): str(value) for key, value in labels.items()}


def index_voya_npz_files(root: str | Path, *, labels_path: str | Path | None = None) -> list[VOYASampleIndex]:
    root = Path(root)
    label_names = load_voya_label_names(labels_path or root / "labels.json")
    samples: list[VOYASampleIndex] = []
    for path in sorted(root.glob("class_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            labels = np.asarray(data["labels"])
            sequence_count = int(data["sequences"].shape[0])
        if len(labels) != sequence_count:
            raise ValueError(f"{path} has {sequence_count} sequences but {len(labels)} labels.")
        for source_index, raw_label in enumerate(labels.tolist()):
            label = int(raw_label)
            samples.append(
                VOYASampleIndex(
                    sample_id=f"{path.stem}:{source_index:06d}",
                    source_file=str(path),
                    source_index=source_index,
                    raw_label=label,
                    normalized_label=None,
                    label_name=label_names.get(label, str(label)),
                )
            )
    return samples


def summarize_voya_labels(labels_path: str | Path) -> dict[str, Any]:
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    values = [str(value) for value in labels.values()]
    blank = [key for key, value in labels.items() if not str(value).strip()]
    duplicates = sorted(label for label, count in Counter(values).items() if count > 1)
    normalized_duplicates = sorted(
        label for label, count in Counter(_normalize_label_text(value) for value in values).items() if count > 1
    )
    class_numbers = sorted(int(normalize_class_name(key).split("_")[1]) for key in labels)
    return {
        "exact_number": len(labels),
        "key_format": "class_####" if all(key.startswith("class_") and len(key) == 10 for key in labels) else "mixed",
        "first_key": f"class_{class_numbers[0]:04d}" if class_numbers else None,
        "last_key": f"class_{class_numbers[-1]:04d}" if class_numbers else None,
        "contiguous": class_numbers == list(range(1, len(class_numbers) + 1)),
        "values_are_vietnamese_gloss_labels": True,
        "duplicate_values": duplicates,
        "duplicate_value_count": len(duplicates),
        "blank_labels": blank,
        "normalization_duplicate_values": normalized_duplicates,
        "normalization_issue_count": len(normalized_duplicates),
        "examples": dict(list(labels.items())[:10]),
    }


def analyze_voya_label_catalog(
    labels_path: str | Path,
    *,
    json_path: str | Path = "outputs/validation/voya_label_analysis.json",
    csv_path: str | Path = "outputs/validation/voya_label_groups.csv",
) -> dict[str, Any]:
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    rows = []
    display_counts = Counter(str(value) for value in labels.values())
    normalized_counts = Counter(normalize_label_for_analysis(value) for value in labels.values())
    root_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for key, value in labels.items():
        display = str(value)
        normalized = normalize_label_for_analysis(display)
        root = strip_parenthetical_markers(normalized)
        region = _regional_marker(normalized)
        item = {
            "class_key": key,
            "label": display,
            "normalized": normalized,
            "root": root,
            "regional_marker": region or "",
        }
        root_groups[root].append(item)
        rows.append(item)

    duplicate_display_labels = sorted(
        (
            {"label": label, "count": count}
            for label, count in display_counts.items()
            if count > 1
        ),
        key=lambda item: (-item["count"], item["label"]),
    )
    parenthetical = [row for row in rows if "(" in row["normalized"] and ")" in row["normalized"]]
    regional = [row for row in rows if row["regional_marker"]]
    grouped = {
        root: items
        for root, items in sorted(root_groups.items())
        if len(items) > 1
    }
    similar_groups = sorted(
        (
            {
                "root": root,
                "count": len(items),
                "classes": [item["class_key"] for item in items],
                "labels": [item["label"] for item in items],
                "regional_markers": sorted({item["regional_marker"] for item in items if item["regional_marker"]}),
            }
            for root, items in grouped.items()
        ),
        key=lambda item: (-item["count"], item["root"]),
    )
    report = {
        "label_catalog_size": len(labels),
        "unique_display_strings": len(display_counts),
        "duplicate_display_label_count": len(duplicate_display_labels),
        "duplicate_display_labels": duplicate_display_labels,
        "unique_normalized_labels": len(normalized_counts),
        "parenthetical_variant_count": len(parenthetical),
        "regional_variant_entry_count": len(regional),
        "same_root_group_count": len(similar_groups),
        "largest_same_root_groups": similar_groups[:50],
        "class_selection_strategies": build_voya_class_selection_strategies(rows, similar_groups),
    }
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group_root", "group_size", "class_key", "label", "normalized", "regional_marker"],
        )
        writer.writeheader()
        for group in similar_groups:
            for class_key, label in zip(group["classes"], group["labels"], strict=True):
                normalized = normalize_label_for_analysis(label)
                writer.writerow(
                    {
                        "group_root": group["root"],
                        "group_size": group["count"],
                        "class_key": class_key,
                        "label": label,
                        "normalized": normalized,
                        "regional_marker": _regional_marker(normalized) or "",
                    }
                )
    return report


def build_voya_class_selection_strategies(
    rows: list[dict[str, str]],
    similar_groups: list[dict[str, Any]],
    *,
    seed: int = 42,
) -> dict[str, list[str]]:
    class_keys = sorted(row["class_key"] for row in rows)
    rng = random.Random(seed)
    random_balanced = sorted(rng.sample(class_keys, k=min(50, len(class_keys))))
    regional_variants = []
    for group in similar_groups:
        if group["regional_markers"] and len(regional_variants) < 50:
            regional_variants.extend(group["classes"])
    regional_variants = sorted(dict.fromkeys(regional_variants))[:50]
    lexical = sorted(dict.fromkeys(key for group in similar_groups[:20] for key in group["classes"]))[:50]
    mixed = sorted(dict.fromkeys(class_keys[:10] + regional_variants[:20] + random_balanced[:20]))[:50]
    return {
        "random_balanced": random_balanced,
        "lexically_similar": lexical,
        "regional_variants": regional_variants,
        "mixed_difficulty": mixed,
    }


def normalize_label_for_analysis(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value)).lower().strip())


def strip_parenthetical_markers(value: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*", " ", value).strip()


def transform_temporal_features(
    features: np.ndarray,
    *,
    view: str,
    sample_id: str,
    temporal_shuffle_seed: int | None = None,
) -> np.ndarray:
    if view == "sequence":
        return features
    if view == "single_middle_frame":
        middle = features.shape[0] // 2
        return np.repeat(features[middle : middle + 1], features.shape[0], axis=0)
    if view == "temporal_shuffle":
        seed = int(hashlib.sha256(f"{temporal_shuffle_seed}:{sample_id}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        indices = np.arange(features.shape[0])
        rng.shuffle(indices)
        return features[indices]
    raise ValueError(f"Unsupported VOYA view: {view}")


def compute_voya_direct_stats(
    root: str | Path,
    *,
    output_path: str | Path = "outputs/validation/voya_direct_stats.json",
) -> dict[str, Any]:
    root = Path(root)
    labels_summary = summarize_voya_labels(root / "labels.json")
    files = []
    total_sequences = 0
    labels_counter: Counter[int] = Counter()
    sequence_lengths: Counter[int] = Counter()
    feature_dims: Counter[int] = Counter()
    nan_count = 0
    inf_count = 0
    zero_count = 0
    value_count = 0
    sequence_hashes: Counter[str] = Counter()

    for path in sorted(root.glob("class_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            sequences = np.asarray(data["sequences"])
            labels = np.asarray(data["labels"])
            file_nan = int(np.isnan(sequences).sum())
            file_inf = int(np.isinf(sequences).sum())
            file_zero = int(np.count_nonzero(sequences == 0))
            for sequence in sequences:
                sequence_hashes[hash_sequence(sequence)] += 1
        unique_labels, counts = np.unique(labels, return_counts=True)
        labels_counter.update({int(label): int(count) for label, count in zip(unique_labels, counts)})
        total_sequences += int(sequences.shape[0])
        sequence_lengths.update({int(sequences.shape[1]): int(sequences.shape[0])})
        feature_dims.update({int(sequences.shape[2]): int(sequences.shape[0])})
        nan_count += file_nan
        inf_count += file_inf
        zero_count += file_zero
        value_count += int(sequences.size)
        files.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sequences_shape": list(sequences.shape),
                "labels_shape": list(labels.shape),
                "unique_labels": [int(value) for value in unique_labels.tolist()],
                "label_counts": {str(int(label)): int(count) for label, count in zip(unique_labels, counts)},
                "nan_count": file_nan,
                "inf_count": file_inf,
                "zero_ratio": file_zero / sequences.size,
            }
        )

    duplicate_hashes = {key: count for key, count in sequence_hashes.items() if count > 1}
    report = {
        "dataset": "VOYA_VSL",
        "remote_revision": VOYA_REVISION,
        "root": str(root),
        "files": files,
        "downloaded_file_count": len(files),
        "downloaded_size_bytes": sum(item["size_bytes"] for item in files),
        "total_sequences": total_sequences,
        "unique_labels": sorted(labels_counter),
        "unique_label_count": len(labels_counter),
        "samples_per_label": {str(label): count for label, count in sorted(labels_counter.items())},
        "sequence_length_distribution": {str(key): value for key, value in sorted(sequence_lengths.items())},
        "feature_dimension_distribution": {str(key): value for key, value in sorted(feature_dims.items())},
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_ratio": zero_count / value_count if value_count else None,
        "duplicate_sequence_hash_count": len(duplicate_hashes),
        "duplicate_sequence_instances": int(sum(duplicate_hashes.values())),
        "labels_json": labels_summary,
        "npz_organization_conclusion": (
            "Downloaded class_XXXX.npz files each contain one semantic label matching class XXXX."
        ),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def select_eligible_labels(
    samples: list[VOYASampleIndex],
    *,
    minimum_samples_per_class: int,
    max_classes: int,
) -> tuple[list[int], dict[str, Any]]:
    counts = Counter(sample.raw_label for sample in samples)
    eligible = [
        label
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= minimum_samples_per_class
    ][:max_classes]
    excluded = {
        str(label): {
            "count": count,
            "reason": f"count < minimum_samples_per_class ({minimum_samples_per_class})",
        }
        for label, count in sorted(counts.items())
        if label not in eligible
    }
    return eligible, {
        "minimum_samples_per_class": minimum_samples_per_class,
        "max_classes": max_classes,
        "eligible_labels": eligible,
        "excluded_labels": excluded,
        "counts": {str(label): count for label, count in sorted(counts.items())},
    }


def deterministic_subset(
    samples: list[VOYASampleIndex],
    *,
    labels: Iterable[int],
    max_samples_per_class: int | None,
    seed: int,
) -> list[VOYASampleIndex]:
    label_set = set(labels)
    rng = random.Random(seed)
    selected: list[VOYASampleIndex] = []
    for label in sorted(label_set):
        group = [sample for sample in samples if sample.raw_label == label]
        rng.shuffle(group)
        selected.extend(sorted(group[:max_samples_per_class], key=lambda sample: sample.sample_id))
    return selected


def stratified_sample_split(
    samples: list[VOYASampleIndex],
    *,
    seed: int,
    ratios: dict[str, float] | None = None,
) -> list[VOYASampleIndex]:
    ratios = ratios or {"train": 0.70, "validation": 0.15, "test": 0.15}
    rng = random.Random(seed)
    split_samples: list[VOYASampleIndex] = []
    for label in sorted({sample.raw_label for sample in samples}):
        group = [sample for sample in samples if sample.raw_label == label]
        rng.shuffle(group)
        count = len(group)
        train_count = max(1, round(count * ratios["train"]))
        validation_count = max(1, round(count * ratios["validation"]))
        if train_count + validation_count >= count:
            validation_count = max(1, count - train_count - 1)
        for index, sample in enumerate(group):
            if index < train_count:
                split = "train"
            elif index < train_count + validation_count:
                split = "validation"
            else:
                split = "test"
            split_samples.append(_replace_sample(sample, split=split))
    return sorted(split_samples, key=lambda sample: sample.sample_id)


def attach_sequence_hashes(samples: list[VOYASampleIndex]) -> list[VOYASampleIndex]:
    cache = VOYANPZCache(max_open_files=1)
    hashed = []
    for sample in samples:
        sequences, _labels = cache.arrays(sample.source_file)
        hashed.append(_replace_sample(sample, sequence_hash=hash_sequence(sequences[sample.source_index])))
    return hashed


def duplicate_leakage_report(samples: list[VOYASampleIndex]) -> dict[str, Any]:
    by_hash: dict[str, set[str]] = defaultdict(set)
    by_hash_samples: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        if sample.sequence_hash is None:
            continue
        by_hash[sample.sequence_hash].add(sample.split or "")
        by_hash_samples[sample.sequence_hash].append(sample.sample_id)
    leaking = {
        hash_value: {
            "splits": sorted(splits),
            "sample_ids": by_hash_samples[hash_value],
        }
        for hash_value, splits in by_hash.items()
        if len(splits) > 1
    }
    return {
        "exact_duplicate_hashes": sum(1 for samples_for_hash in by_hash_samples.values() if len(samples_for_hash) > 1),
        "cross_split_duplicate_hashes": leaking,
        "cross_split_duplicate_count": len(leaking),
        "passes": not leaking,
        "note": "Exact duplicate feature-array check only; signer leakage cannot be assessed without signer metadata.",
    }


def enforce_no_duplicate_hash_cross_split(samples: list[VOYASampleIndex]) -> list[VOYASampleIndex]:
    """Move exact duplicate feature arrays into one split so hashes cannot cross splits."""

    by_hash: dict[str, list[VOYASampleIndex]] = defaultdict(list)
    for sample in samples:
        by_hash[sample.sequence_hash or sample.sample_id].append(sample)
    updated: list[VOYASampleIndex] = []
    split_priority = {"train": 0, "validation": 1, "test": 2, None: 3}
    for group in by_hash.values():
        target_split = min((sample.split for sample in group), key=lambda split: split_priority.get(split, 9))
        updated.extend(_replace_sample(sample, split=target_split) for sample in group)
    return sorted(updated, key=lambda sample: sample.sample_id)


def fit_train_normalization(samples: list[VOYASampleIndex]) -> dict[str, Any]:
    train_samples = [sample for sample in samples if sample.split == "train"]
    if not train_samples:
        raise ValueError("Cannot fit normalization without train samples.")
    cache = VOYANPZCache(max_open_files=1)
    feature_sum: np.ndarray | None = None
    feature_sq_sum: np.ndarray | None = None
    frame_count = 0
    for sample in train_samples:
        sequences, _labels = cache.arrays(sample.source_file)
        sequence = np.nan_to_num(sequences[sample.source_index].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if feature_sum is None:
            feature_sum = np.zeros(sequence.shape[1], dtype=np.float64)
            feature_sq_sum = np.zeros(sequence.shape[1], dtype=np.float64)
        feature_sum += sequence.sum(axis=0)
        feature_sq_sum += np.square(sequence).sum(axis=0)
        frame_count += sequence.shape[0]
    assert feature_sum is not None and feature_sq_sum is not None
    mean = feature_sum / frame_count
    variance = np.maximum(feature_sq_sum / frame_count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    return {
        "type": "per_feature_zscore",
        "fit_split": "train",
        "frame_count": frame_count,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "epsilon": 1e-6,
    }


def apply_normalization(features: np.ndarray, normalization: dict[str, Any] | None) -> np.ndarray:
    if not normalization or normalization.get("type") == "none":
        return features
    if normalization.get("type") != "per_feature_zscore":
        raise ValueError(f"Unsupported normalization type: {normalization.get('type')}")
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    epsilon = float(normalization.get("epsilon", 1e-6))
    return (features - mean) / np.maximum(std, epsilon)


def write_split_manifest(
    samples: list[VOYASampleIndex],
    *,
    path: str | Path,
    label_names: dict[int, str],
) -> dict[str, Any]:
    split_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for sample in samples:
        split_counts[sample.split or "unknown"][sample.raw_label] += 1
    manifest = {
        "dataset": "VOYA_VSL",
        "remote_revision": VOYA_REVISION,
        "split_type": "sample_stratified",
        "signer_independent": False,
        "samples": [_sample_to_dict(sample) for sample in samples],
        "counts": {
            split: {str(label): count for label, count in sorted(counter.items())}
            for split, counter in sorted(split_counts.items())
        },
        "label_names": {str(label): label_names.get(label, str(label)) for label in sorted(label_names)},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def write_per_class_metrics_csv(
    per_class_accuracy: dict[str, float | None],
    supports: dict[str, int],
    *,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "support", "accuracy"])
        writer.writeheader()
        for label, accuracy in per_class_accuracy.items():
            writer.writerow({"label": label, "support": supports.get(label, 0), "accuracy": accuracy})


def write_top_confusions_csv(
    confusion_matrix: list[list[int]],
    class_names: list[str],
    *,
    path: str | Path,
    limit: int = 20,
) -> None:
    rows = []
    for actual_index, row in enumerate(confusion_matrix):
        for predicted_index, count in enumerate(row):
            if actual_index != predicted_index and count:
                rows.append(
                    {
                        "actual": class_names[actual_index],
                        "predicted": class_names[predicted_index],
                        "count": count,
                    }
                )
    rows.sort(key=lambda item: (-item["count"], item["actual"], item["predicted"]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["actual", "predicted", "count"])
        writer.writeheader()
        writer.writerows(rows[:limit])


def hash_sequence(sequence: np.ndarray) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(sequence)
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _replace_sample(sample: VOYASampleIndex, **updates: Any) -> VOYASampleIndex:
    data = sample.__dict__ | updates
    return VOYASampleIndex(**data)


def _sample_to_dict(sample: VOYASampleIndex) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "source_file": sample.source_file,
        "source_index": sample.source_index,
        "raw_label": sample.raw_label,
        "normalized_label": sample.normalized_label,
        "label_name": sample.label_name,
        "split": sample.split,
        "sequence_hash": sample.sequence_hash,
    }


def _normalize_label_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _regional_marker(value: str) -> str | None:
    for marker in ("bắc", "nam", "trung"):
        if f"({marker})" in value:
            return marker
    return None
