"""Dataset validation for normalized samples."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from vietsign.data.schema import SignSample
from vietsign.data.video import inspect_video

IssueSeverity = Literal["error", "warning"]

VALID_SPLITS = {"train", "validation", "val", "test"}


class DatasetIssue(BaseModel):
    """One validation issue."""

    code: str
    message: str
    severity: IssueSeverity = "error"
    sample_id: str | None = None


class DatasetValidationReport(BaseModel):
    """Validation result for a normalized dataset."""

    issues: list[DatasetIssue] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def add(
        self,
        code: str,
        message: str,
        *,
        severity: IssueSeverity = "error",
        sample_id: str | None = None,
    ) -> None:
        self.issues.append(
            DatasetIssue(code=code, message=message, severity=severity, sample_id=sample_id)
        )


def validate_samples(
    samples: list[SignSample],
    *,
    inspect_videos: bool = True,
) -> DatasetValidationReport:
    """Validate normalized dataset samples."""

    report = DatasetValidationReport()
    if not samples:
        report.add("empty_dataset", "Dataset contains no samples.")
        return report

    _detect_duplicates(samples, report)
    _validate_fields(samples, report)
    _detect_empty_class_ids(samples, report)
    _detect_signer_leakage(samples, report)

    if inspect_videos:
        _validate_videos(samples, report)

    return report


def _detect_duplicates(samples: list[SignSample], report: DatasetValidationReport) -> None:
    sample_counts = Counter(sample.sample_id for sample in samples)
    for sample_id, count in sample_counts.items():
        if count > 1:
            report.add("duplicate_sample_id", f"Duplicate sample_id '{sample_id}' appears {count} times.")

    path_counts = Counter(str(sample.video_path) for sample in samples)
    for video_path, count in path_counts.items():
        if count > 1:
            report.add("duplicate_video_path", f"Duplicate video_path '{video_path}' appears {count} times.")


def _validate_fields(samples: list[SignSample], report: DatasetValidationReport) -> None:
    for sample in samples:
        if not sample.label.strip():
            report.add("invalid_label", "Label must not be empty.", sample_id=sample.sample_id)
        if sample.label_id < 0:
            report.add("invalid_label_id", "label_id must be non-negative.", sample_id=sample.sample_id)
        if sample.split not in VALID_SPLITS:
            report.add(
                "unknown_split",
                f"Unknown split '{sample.split}'. Expected one of {sorted(VALID_SPLITS)}.",
                sample_id=sample.sample_id,
            )


def _detect_empty_class_ids(samples: list[SignSample], report: DatasetValidationReport) -> None:
    label_ids = {sample.label_id for sample in samples if sample.label_id >= 0}
    if not label_ids:
        report.add("empty_classes", "Dataset has no valid label IDs.")
        return

    missing = sorted(set(range(max(label_ids) + 1)) - label_ids)
    if missing:
        report.add(
            "empty_class_ids",
            f"Missing label IDs in the contiguous range 0..{max(label_ids)}: {missing}.",
            severity="warning",
        )


def _detect_signer_leakage(samples: list[SignSample], report: DatasetValidationReport) -> None:
    signer_splits: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        if sample.signer_id:
            signer_splits[sample.signer_id].add(_canonical_split(sample.split))

    for signer_id, splits in sorted(signer_splits.items()):
        if len(splits) > 1:
            report.add(
                "signer_leakage",
                f"Signer '{signer_id}' appears across splits: {sorted(splits)}.",
            )


def _validate_videos(samples: list[SignSample], report: DatasetValidationReport) -> None:
    for sample in samples:
        video_path = Path(sample.video_path)
        if not video_path.exists():
            report.add(
                "missing_video",
                f"Video file is missing: {video_path}",
                sample_id=sample.sample_id,
            )
            continue

        metadata = inspect_video(video_path)
        if not metadata.readable:
            report.add(
                "corrupt_video",
                f"Video file is unreadable or corrupt: {video_path} ({metadata.error})",
                sample_id=sample.sample_id,
            )


def _canonical_split(split: str) -> str:
    return "validation" if split == "val" else split

