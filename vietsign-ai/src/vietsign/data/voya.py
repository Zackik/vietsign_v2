"""VOYA_VSL source-layout inspection and guarded conversion."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

VOYA_REPO = "Kateht/VOYA_VSL"
VOYA_REVISION = "9686a7c784846b5ba5008c8bbbbcb97ee889fc4c"
VOYA_RAW_BASE_URL = f"https://huggingface.co/datasets/{VOYA_REPO}/raw/main"
VOYA_RESOLVE_BASE_URL = f"https://huggingface.co/datasets/{VOYA_REPO}/resolve/main"

MAPPING_VERIFIED = "VERIFIED"
MAPPING_STRONGLY_SUPPORTED = "STRONGLY_SUPPORTED_BUT_UNVERIFIED"
MAPPING_UNKNOWN = "UNKNOWN"

METADATA_FILES = ("README.md", "dataset.py", "labels.json")
DEFAULT_PROBE_CLASS = "class_0001"


class VOYAMappingError(RuntimeError):
    """Raised when VOYA source features cannot be safely mapped to VietSign55."""


def normalize_class_name(value: str | int) -> str:
    """Normalize class selectors like 1, 0001, class_0001, or class_0001.npz."""

    text = str(value).strip().removesuffix(".npz")
    match = re.fullmatch(r"(?:class_)?(\d+)", text)
    if not match:
        raise ValueError(f"Invalid VOYA class selector: {value!r}")
    return f"class_{int(match.group(1)):04d}"


def audit_voya_directory(root: str | Path = "data/raw/voya_vsl") -> dict[str, Any]:
    root = Path(root)
    merged = root / "Merged"
    labels_path = root / "labels.json"
    files = sorted(merged.glob("*.npz")) if merged.exists() else sorted(root.glob("class_*.npz"))
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    shape_counts: dict[str, int] = {}
    sample_count = 0
    inspected = []
    total_bytes = sum(path.stat().st_size for path in files) if files else 0
    for path in files[:20]:
        with np.load(path, allow_pickle=False) as data:
            keys = list(data.files)
            sequences = data.get("sequences", None)
            labels_array = data.get("labels", None)
            if sequences is not None:
                shape_counts[str(tuple(sequences.shape))] = (
                    shape_counts.get(str(tuple(sequences.shape)), 0) + 1
                )
                sample_count += int(sequences.shape[0]) if sequences.ndim == 3 else 1
            inspected.append(
                {
                    "file": str(path),
                    "keys": keys,
                    "sequences_shape": list(sequences.shape) if sequences is not None else None,
                    "labels_shape": list(labels_array.shape) if labels_array is not None else None,
                }
            )
    return {
        "dataset": "VOYA_VSL",
        "root": str(root),
        "available": bool(files),
        "npz_file_count": len(files),
        "inspected_file_count": len(inspected),
        "labels_count": len(labels),
        "estimated_size_bytes": total_bytes,
        "observed_shapes_first_20": shape_counts,
        "estimated_sample_count_from_first_20": sample_count,
        "signer_metadata": "not_found",
        "split_metadata": "not_found",
        "feature_layout_status": "unverified_source_layout",
        "inspected": inspected,
    }


def fetch_voya_probe_files(
    target: str | Path = "data/raw/voya_vsl_probe",
    classes: list[str] | tuple[str, ...] | None = None,
    *,
    metadata_only: bool = False,
    max_class_size_mb: float = 1024.0,
    max_total_size_gb: float | None = None,
) -> dict[str, Any]:
    """Download VOYA metadata and a selected class subset for structure probing."""

    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for filename in METADATA_FILES:
        url = f"{VOYA_RAW_BASE_URL}/{filename}"
        destination = target / filename
        _download(url, destination)
        records.append(_file_record(destination, url, kind="metadata"))

    selected_classes = [] if metadata_only else [
        normalize_class_name(name) for name in classes or [DEFAULT_PROBE_CLASS]
    ]
    max_bytes = int(max_class_size_mb * 1024 * 1024)
    size_by_class: dict[str, int | None] = {}
    for class_name in selected_classes:
        filename = f"Merged/{class_name}.npz"
        url = f"{VOYA_RESOLVE_BASE_URL}/{filename}"
        size = remote_file_size(url)
        size_by_class[class_name] = size
        if size is not None and size > max_bytes:
            raise ValueError(
                f"{class_name}.npz is {size} bytes, above max_class_size_mb={max_class_size_mb}."
            )
    known_total = sum(size for size in size_by_class.values() if size is not None)
    if max_total_size_gb is not None and known_total > int(max_total_size_gb * 1024**3):
        raise ValueError(
            f"Selected VOYA files total {known_total} bytes, above max_total_size_gb={max_total_size_gb}."
        )
    for class_name in selected_classes:
        filename = f"Merged/{class_name}.npz"
        url = f"{VOYA_RESOLVE_BASE_URL}/{filename}"
        destination = target / f"{class_name}.npz"
        _download(url, destination)
        records.append(_file_record(destination, url, kind="class_npz"))

    return {
        "dataset": "VOYA_VSL",
        "repo": VOYA_REPO,
        "revision_observed": VOYA_REVISION,
        "target": str(target),
        "metadata_only": metadata_only,
        "classes": selected_classes,
        "remote_class_sizes": size_by_class,
        "remote_class_size_total_bytes": known_total,
        "files": records,
    }


def validate_voya_labels(labels_path: str | Path) -> dict[str, Any]:
    labels_path = Path(labels_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    class_numbers = sorted(_class_number(key) for key in labels)
    counts = _value_counts(str(value) for value in labels.values())
    duplicate_labels = sorted(label for label, count in counts.items() if count > 1)
    missing = (
        [number for number in range(class_numbers[0], class_numbers[-1] + 1) if number not in class_numbers]
        if class_numbers
        else []
    )
    return {
        "path": str(labels_path),
        "mapping_count": len(labels),
        "class_key_format": "class_####"
        if all(re.fullmatch(r"class_\d{4}", key) for key in labels)
        else "mixed",
        "first_class": f"class_{class_numbers[0]:04d}" if class_numbers else None,
        "last_class": f"class_{class_numbers[-1]:04d}" if class_numbers else None,
        "is_contiguous": not missing,
        "missing_class_numbers": missing,
        "duplicate_labels": duplicate_labels,
        "examples": dict(list(labels.items())[:5]),
    }


def inspect_voya_npz(path: str | Path, sequence_index: int = 0) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)
        array_reports = {key: _array_report(data[key]) for key in keys}
        sequences = data.get("sequences")
        labels = data.get("labels")
        if sequences is None:
            raise ValueError(f"{path} does not contain a 'sequences' array.")
        if sequences.ndim != 3:
            raise ValueError(f"Expected sequences shape [N,T,F], got {sequences.shape}.")
        if not 0 <= sequence_index < sequences.shape[0]:
            raise IndexError(f"sequence_index={sequence_index} outside [0,{sequences.shape[0] - 1}].")
        sequence = np.asarray(sequences[sequence_index])
        sequence_count = int(sequences.shape[0])
        labels_list = labels.tolist() if labels is not None else []

    candidates = candidate_layout_report(sequence)
    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "keys": keys,
        "arrays": array_reports,
        "number_of_sequences": sequence_count,
        "sampled_sequence_index": sequence_index,
        "label_values": sorted({int(value) for value in labels_list}) if labels_list else [],
        "sequence_statistics": sequence_statistics(sequence),
        "candidate_layouts": candidates,
        "direct_training_feasibility": direct_training_feasibility(sequence),
    }


def sequence_statistics(sequence: np.ndarray) -> dict[str, Any]:
    array = np.asarray(sequence)
    finite = array[np.isfinite(array)]
    per_column_std = np.nanstd(array, axis=0) if array.ndim == 2 else np.array([])
    per_frame_std = np.nanstd(array, axis=1) if array.ndim == 2 else np.array([])
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "feature_dim": int(array.shape[-1]) if array.ndim >= 1 else None,
        "nan_count": int(np.isnan(array).sum()) if np.issubdtype(array.dtype, np.number) else None,
        "inf_count": int(np.isinf(array).sum()) if np.issubdtype(array.dtype, np.number) else None,
        "zero_ratio": _zero_ratio(array),
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "mean": float(np.mean(finite)) if finite.size else None,
        "std": float(np.std(finite)) if finite.size else None,
        "per_column_std_mean": float(np.mean(per_column_std)) if per_column_std.size else None,
        "per_column_std_max": float(np.max(per_column_std)) if per_column_std.size else None,
        "per_frame_std_mean": float(np.mean(per_frame_std)) if per_frame_std.size else None,
        "per_frame_std_max": float(np.max(per_frame_std)) if per_frame_std.size else None,
    }


def candidate_layout_report(sequence: np.ndarray) -> dict[str, Any]:
    feature_dim = int(sequence.shape[-1]) if sequence.ndim == 2 else None
    candidates = [
        {
            "name": "pose25_face468_left21_right21_xyz",
            "blocks": [("pose", 25), ("face", 468), ("left_hand", 21), ("right_hand", 21)],
            "source": "mathematical hypothesis: 25 + 468 + 21 + 21 = 535 triplets",
        },
        {
            "name": "face468_pose25_left21_right21_xyz",
            "blocks": [("face", 468), ("pose", 25), ("left_hand", 21), ("right_hand", 21)],
            "source": "mathematical hypothesis: same landmarks, face-first ordering",
        },
        {
            "name": "pose33_face460_left21_right21_xyz",
            "blocks": [("pose", 33), ("face", 460), ("left_hand", 21), ("right_hand", 21)],
            "source": "mathematical hypothesis: alternate pose/face split",
        },
    ]
    reports = []
    for candidate in candidates:
        expected_dim = sum(count for _, count in candidate["blocks"]) * 3
        matched = feature_dim == expected_dim
        block_reports = []
        if matched:
            start = 0
            for block_name, joint_count in candidate["blocks"]:
                width = joint_count * 3
                block = sequence[:, start : start + width]
                block_reports.append(
                    {
                        "name": block_name,
                        "joint_count": joint_count,
                        "feature_slice": [start, start + width],
                        **_block_statistics(block),
                    }
                )
                start += width
        reports.append(
            {
                "name": candidate["name"],
                "source": candidate["source"],
                "expected_feature_dim": expected_dim,
                "matches_feature_dim": matched,
                "blocks": block_reports,
            }
        )
    return {
        "feature_dim": feature_dim,
        "mapping_status": mapping_confidence(reports, authoritative_evidence=False),
        "authoritative_evidence": False,
        "canonical_conversion_allowed": False,
        "candidates": reports,
        "warning": "Dimension matching and statistical plausibility are not authoritative evidence.",
    }


def mapping_confidence(
    candidate_reports: list[dict[str, Any]],
    *,
    authoritative_evidence: bool,
) -> str:
    if authoritative_evidence:
        return MAPPING_VERIFIED
    if any(candidate.get("matches_feature_dim") for candidate in candidate_reports):
        return MAPPING_STRONGLY_SUPPORTED
    return MAPPING_UNKNOWN


def direct_training_feasibility(sequence: np.ndarray) -> dict[str, Any]:
    shape = list(sequence.shape)
    feasible = sequence.ndim == 2 and shape[1] == 1605
    return {
        "status": "feasible" if feasible else "not_feasible",
        "input_contract": "[T,1605] frame-level source-layout features" if feasible else None,
        "reason": (
            "A temporal model can project 1605-dim VOYA frames directly. This is a source-layout "
            "baseline, not VietSign55 graph/canonical training."
        )
        if feasible
        else f"Expected [T,1605], got {shape}.",
    }


def build_voya_probe_report(
    root: str | Path = "data/raw/voya_vsl_probe",
    class_name: str = DEFAULT_PROBE_CLASS,
    *,
    sequence_index: int = 0,
) -> dict[str, Any]:
    root = Path(root)
    normalized_class = normalize_class_name(class_name)
    npz_path = root / f"{normalized_class}.npz"
    labels_path = root / "labels.json"
    labels = validate_voya_labels(labels_path) if labels_path.exists() else {"status": "missing"}
    npz_report = inspect_voya_npz(npz_path, sequence_index=sequence_index)
    direct = npz_report["direct_training_feasibility"]
    mapping_status = npz_report["candidate_layouts"]["mapping_status"]
    return {
        "dataset": "VOYA_VSL",
        "repo": VOYA_REPO,
        "revision_observed": VOYA_REVISION,
        "root": str(root),
        "downloaded_class": normalized_class,
        "labels": labels,
        "npz": npz_report,
        "mapping_status": mapping_status,
        "canonical_conversion": "blocked_without_authoritative_layout",
        "direct_1605_feature_training": direct,
        "remaining_blocker": (
            "Need authoritative VOYA feature ordering before converting to VietSign55 [T,55,3]."
        ),
        "safe_adapter_decision": (
            "B. canonical conversion unverified, but direct 1605-feature training possible"
            if direct["status"] == "feasible"
            else "C. feature layout still blocked"
        ),
    }


def write_voya_probe_outputs(
    report: dict[str, Any],
    json_path: str | Path = "outputs/validation/voya_probe.json",
    markdown_path: str | Path = "docs/VOYA_PROBE_REPORT.md",
) -> dict[str, str]:
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(_probe_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def write_voya_structure_diagnostic(
    npz_path: str | Path,
    output_path: str | Path = "outputs/validation/voya_probe_structure.png",
    *,
    sequence_index: int = 0,
) -> str:
    """Write a diagnostic plot for one plausible mathematical candidate."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path, allow_pickle=False) as data:
        sequence = np.asarray(data["sequences"][sequence_index])

    layout = [("pose", 25), ("face", 468), ("left_hand", 21), ("right_hand", 21)]
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    start = 0
    for axis, (name, joint_count) in zip(axes, layout, strict=True):
        block = sequence[:, start : start + joint_count * 3].reshape(sequence.shape[0], joint_count, 3)
        frame_index = sequence.shape[0] // 2
        xy = block[frame_index, :, :2]
        axis.scatter(xy[:, 0], xy[:, 1], s=8)
        axis.set_title(f"{name} ({joint_count})")
        axis.set_aspect("equal", adjustable="box")
        axis.invert_yaxis()
        axis.grid(True, alpha=0.25)
        start += joint_count * 3
    figure.suptitle("VOYA structure diagnostic only - unverified candidate layout")
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return str(output_path)


def voya_feature_layout_report(feature_dim: int = 1605) -> dict[str, Any]:
    return {
        "dataset": "VOYA_VSL",
        "reported_frame_shape": [60, feature_dim],
        "divisible_by_xyz": feature_dim % 3 == 0,
        "landmark_triplets": feature_dim // 3 if feature_dim % 3 == 0 else None,
        "canonical_vietsign55_mapping": "not_implemented",
        "reason": (
            "Remote dataset.py declares [60,1605], but public metadata does not define which indices "
            "correspond to pose/face/left hand/right hand or whether confidence/visibility exists. "
            "VietSign55 conversion is blocked until the source feature layout is verified."
        ),
        "safe_training_path": "source-layout adapter/audit only; do not fabricate anatomical mapping",
    }


def convert_voya_to_vietsign55(_sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raise VOYAMappingError(
        "VOYA_VSL [60,1605] feature layout is not sufficiently documented to map safely to VietSign55."
    )


def remote_file_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        for header in ("x-linked-size", "content-length"):
            value = response.headers.get(header)
            if value and value.isdigit():
                return int(value)
    return None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        destination.write_bytes(response.read())


def _file_record(path: Path, url: str, *, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": str(path),
        "url": url,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _class_number(key: str) -> int:
    return int(normalize_class_name(key).split("_")[1])


def _value_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _array_report(array: np.ndarray) -> dict[str, Any]:
    report: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if np.issubdtype(array.dtype, np.number):
        finite = array[np.isfinite(array)]
        report.update(
            {
                "nan_count": int(np.isnan(array).sum()),
                "inf_count": int(np.isinf(array).sum()),
                "min": float(np.min(finite)) if finite.size else None,
                "max": float(np.max(finite)) if finite.size else None,
            }
        )
    return report


def _zero_ratio(array: np.ndarray) -> float | None:
    if not np.issubdtype(array.dtype, np.number) or array.size == 0:
        return None
    return float(np.count_nonzero(array == 0) / array.size)


def _block_statistics(block: np.ndarray) -> dict[str, Any]:
    finite = block[np.isfinite(block)]
    reshaped = block.reshape(block.shape[0], -1, 3)
    diffs = np.linalg.norm(np.diff(reshaped[:, :, :2], axis=0), axis=2)
    return {
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "mean": float(np.mean(finite)) if finite.size else None,
        "std": float(np.std(finite)) if finite.size else None,
        "zero_ratio": _zero_ratio(block),
        "median_temporal_movement": float(np.median(diffs)) if diffs.size else None,
        "p90_temporal_movement": float(np.percentile(diffs, 90)) if diffs.size else None,
    }


def _probe_markdown(report: dict[str, Any]) -> str:
    npz = report["npz"]
    stats = npz["sequence_statistics"]
    candidates = npz["candidate_layouts"]
    direct = report["direct_1605_feature_training"]
    candidate_lines = [
        f"- `{candidate['name']}`: matches_feature_dim={candidate['matches_feature_dim']}, "
        f"expected_feature_dim={candidate['expected_feature_dim']}"
        for candidate in candidates["candidates"]
    ]
    return "\n".join(
        [
            "# VOYA Structure Probe Report",
            "",
            (
                "This report probes one selected VOYA_VSL class file. It does not verify canonical "
                "VietSign55 landmark ordering."
            ),
            "",
            "## Summary",
            "",
            f"- Dataset: `{report['dataset']}`",
            f"- Class: `{report['downloaded_class']}`",
            f"- Mapping status: `{report['mapping_status']}`",
            f"- Canonical conversion: `{report['canonical_conversion']}`",
            f"- Safe adapter decision: `{report['safe_adapter_decision']}`",
            "",
            "## Observed NPZ",
            "",
            f"- Path: `{npz['path']}`",
            f"- SHA256: `{npz['sha256']}`",
            f"- Keys: `{npz['keys']}`",
            f"- Number of sequences: `{npz['number_of_sequences']}`",
            f"- Sampled sequence shape: `{stats['shape']}`",
            f"- Feature dim: `{stats['feature_dim']}`",
            f"- NaN count: `{stats['nan_count']}`",
            f"- Inf count: `{stats['inf_count']}`",
            f"- Zero ratio: `{stats['zero_ratio']}`",
            "",
            "## Candidate Layouts",
            "",
            *candidate_lines,
            "",
            "## Direct Feature Training",
            "",
            f"- Status: `{direct['status']}`",
            f"- Input contract: `{direct['input_contract']}`",
            f"- Reason: {direct['reason']}",
            "",
            "## Canonical Mapping Decision",
            "",
            (
                "The `[T,1605]` tensor can be used for a direct VOYA source-layout baseline. "
                "Conversion into VietSign55 `[T,55,3]` remains blocked until the VOYA feature "
                "ordering is confirmed from authoritative documentation or source code."
            ),
            "",
        ]
    )
