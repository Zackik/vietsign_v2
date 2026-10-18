from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from vietsign.data.adapters import (
    CustomDatasetAdapter,
    MultiViewCSVAdapter,
    ViSignLanguageVideoAdapter,
)
from vietsign.data.schema import SignSample
from vietsign.data.splits import make_group_independent_splits, make_signer_independent_splits
from vietsign.data.stats import compute_dataset_statistics
from vietsign.data.torch_dataset import SignVideoDataset
from vietsign.data.validation import validate_samples
from vietsign.data.video import inspect_video


def _write_video(path: Path, *, frames: int = 4, fps: float = 5.0, size: tuple[int, int] = (32, 24)) -> None:
    import cv2  # type: ignore
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    assert writer.isOpened()
    try:
        for index in range(frames):
            frame = np.full((size[1], size[0], 3), index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sample_id",
        "video_path",
        "label",
        "label_id",
        "signer_id",
        "split",
        "camera",
        "view_id",
        "domain_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sample(
    sample_id: str,
    video_path: Path,
    *,
    label: str = "xin_chao",
    label_id: int = 0,
    signer_id: str | None = "s1",
    split: str = "train",
) -> SignSample:
    return SignSample(
        sample_id=sample_id,
        video_path=video_path,
        label=label,
        label_id=label_id,
        signer_id=signer_id,
        split=split,
        dataset="test",
    )


def test_normalized_sample_schema(tmp_path: Path) -> None:
    sample = _sample("sample-1", tmp_path / "video.mp4")

    assert sample.sample_id == "sample-1"
    assert sample.video_path == tmp_path / "video.mp4"
    assert sample.dataset == "test"
    assert sample.metadata == {}


def test_custom_csv_loading(tmp_path: Path) -> None:
    video_path = tmp_path / "videos" / "one.mp4"
    video_path.parent.mkdir()
    _write_video(video_path)
    csv_path = tmp_path / "metadata.csv"
    _write_csv(
        csv_path,
        [
            {
                "sample_id": "one",
                "video_path": "videos/one.mp4",
                "label": "xin_chao",
                "label_id": 0,
                "signer_id": "s1",
                "split": "train",
            }
        ],
    )

    samples = CustomDatasetAdapter("custom", tmp_path, csv_path).load_samples()

    assert len(samples) == 1
    assert samples[0].video_path == video_path
    assert samples[0].dataset == "custom"

def test_multiview_csv_loading(tmp_path: Path) -> None:
    video_path = tmp_path / "front" / "one.mp4"
    video_path.parent.mkdir()
    _write_video(video_path)
    csv_path = tmp_path / "metadata.csv"
    _write_csv(
        csv_path,
        [
            {
                "sample_id": "one-front",
                "video_path": "front/one.mp4",
                "label": "xin_chao",
                "label_id": 0,
                "signer_id": "s1",
                "split": "train",
                "camera": "cam_a",
                "view_id": "front",
                "domain_id": "studio",
            }
        ],
    )

    samples = MultiViewCSVAdapter("multiview", tmp_path, csv_path).load_samples()

    assert len(samples) == 1
    assert samples[0].camera == "cam_a"
    assert samples[0].view_id == "front"
    assert samples[0].domain_id == "studio"

def test_visignlanguage_video_metadata_loading(tmp_path: Path) -> None:
    metadata_path = tmp_path / "dataset_metadata.csv"
    fieldnames = [
        "split",
        "label",
        "filename",
        "filepath",
        "size_mb",
        "fps",
        "total_frames",
        "width",
        "height",
        "duration_s",
    ]
    with metadata_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "split": "train",
                "label": "An ủi",
                "filename": "334053.mp4",
                "filepath": "/content/drive/MyDrive/train/An ủi/334053.mp4",
                "size_mb": "0.052",
                "fps": "18.73",
                "total_frames": "30",
                "width": "224",
                "height": "224",
                "duration_s": "1.602",
            }
        )

    samples = ViSignLanguageVideoAdapter(
        "ViSignLanguage-Video",
        tmp_path,
        metadata_path,
    ).load_samples()

    assert len(samples) == 1
    assert samples[0].label == "An ủi"
    assert samples[0].video_path == tmp_path / "train" / "An ủi" / "334053.mp4"
    assert samples[0].fps == 18.73
    assert samples[0].frame_count == 30


def test_missing_video_detection(tmp_path: Path) -> None:
    report = validate_samples([_sample("missing", tmp_path / "missing.mp4")])

    assert any(issue.code == "missing_video" for issue in report.issues)
    assert not report.is_valid


def test_duplicate_sample_detection(tmp_path: Path) -> None:
    video_path = tmp_path / "same.mp4"
    _write_video(video_path)
    samples = [_sample("dup", video_path), _sample("dup", video_path, signer_id="s2")]

    report = validate_samples(samples, inspect_videos=False)

    assert any(issue.code == "duplicate_sample_id" for issue in report.issues)
    assert any(issue.code == "duplicate_video_path" for issue in report.issues)


def test_video_metadata_extraction(tmp_path: Path) -> None:
    video_path = tmp_path / "meta.mp4"
    _write_video(video_path, frames=5, fps=10.0, size=(40, 30))

    metadata = inspect_video(video_path)

    assert metadata.readable
    assert metadata.frame_count == 5
    assert metadata.width == 40
    assert metadata.height == 30
    assert metadata.fps and metadata.fps > 0
    assert metadata.duration and metadata.duration > 0


def test_corrupt_video_detection(tmp_path: Path) -> None:
    corrupt_path = tmp_path / "corrupt.mp4"
    corrupt_path.write_text("not a video", encoding="utf-8")

    report = validate_samples([_sample("bad", corrupt_path)])

    assert any(issue.code == "corrupt_video" for issue in report.issues)


def test_dataset_statistics() -> None:
    samples = [
        _sample("a", Path("a.mp4"), label="a", label_id=0, signer_id="s1", split="train").model_copy(
            update={"duration": 1.0, "fps": 25.0, "width": 32, "height": 24}
        ),
        _sample("b", Path("b.mp4"), label="b", label_id=1, signer_id="s2", split="validation").model_copy(
            update={"duration": 3.0, "fps": 30.0, "width": 32, "height": 24}
        ),
        _sample("c", Path("c.mp4"), label="b", label_id=1, signer_id="s2", split="test").model_copy(
            update={
                "duration": 2.0,
                "fps": 30.0,
                "width": 64,
                "height": 48,
                "camera": "cam_b",
                "view_id": "right",
                "domain_id": "webcam",
            }
        ),
    ]

    stats = compute_dataset_statistics(samples)

    assert stats.total_samples == 3
    assert stats.num_labels == 2
    assert stats.num_signers == 2
    assert stats.samples_per_class["b"] == 2
    assert stats.num_cameras == 1
    assert stats.num_views == 1
    assert stats.num_domains == 1
    assert stats.samples_per_view["right"] == 1
    assert stats.duration.mean == 2.0
    assert stats.resolutions["32x24"] == 2
    assert stats.class_imbalance["ratio"] == 2.0


def test_signer_leakage_detection(tmp_path: Path) -> None:
    samples = [
        _sample("train", tmp_path / "a.mp4", signer_id="same", split="train"),
        _sample("test", tmp_path / "b.mp4", signer_id="same", split="test"),
    ]

    report = validate_samples(samples, inspect_videos=False)

    assert any(issue.code == "signer_leakage" for issue in report.issues)


def test_signer_independent_splitting_is_deterministic() -> None:
    samples = [
        _sample(f"{signer}-{index}", Path(f"{signer}-{index}.mp4"), signer_id=signer)
        for signer in ("s1", "s2", "s3", "s4", "s5")
        for index in range(2)
    ]

    first = make_signer_independent_splits(samples, seed=123)
    second = make_signer_independent_splits(samples, seed=123)

    assert [sample.split for sample in first] == [sample.split for sample in second]
    signer_to_split = {}
    for sample in first:
        signer_to_split.setdefault(sample.signer_id, sample.split)
        assert signer_to_split[sample.signer_id] == sample.split


def test_signer_independent_splitting_requires_signers() -> None:
    with pytest.raises(ValueError, match="requires signer_id"):
        make_signer_independent_splits([_sample("no-signer", Path("x.mp4"), signer_id=None)])

def test_group_independent_splitting_by_view_is_deterministic() -> None:
    samples = [
        _sample(f"{view}-{index}", Path(f"{view}-{index}.mp4")).model_copy(update={"view_id": view})
        for view in ("front", "left", "right", "top")
        for index in range(2)
    ]

    first = make_group_independent_splits(samples, group_field="view_id", seed=7)
    second = make_group_independent_splits(samples, group_field="view_id", seed=7)

    assert [sample.split for sample in first] == [sample.split for sample in second]
    view_to_split = {}
    for sample in first:
        view_to_split.setdefault(sample.view_id, sample.split)
        assert view_to_split[sample.view_id] == sample.split

def test_group_independent_splitting_requires_group_values() -> None:
    with pytest.raises(ValueError, match="requires camera"):
        make_group_independent_splits([_sample("no-camera", Path("x.mp4"))], group_field="camera")


def test_torch_dataset_basic_behavior() -> None:
    dataset = SignVideoDataset([_sample("one", Path("one.mp4"), label_id=7)])

    item = dataset[0]

    assert len(dataset) == 1
    assert item["sample_id"] == "one"
    assert item["video_path"] == "one.mp4"
    assert item["label_id"] == 7


def test_cli_dataset_commands(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    _write_video(video_path)
    csv_path = tmp_path / "metadata.csv"
    _write_csv(
        csv_path,
        [
            {
                "sample_id": "clip",
                "video_path": "clip.mp4",
                "label": "xin_chao",
                "label_id": 0,
                "signer_id": "s1",
                "split": "train",
            }
        ],
    )
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        f"""
dataset:
  name: custom
  adapter: custom_csv
  root: {tmp_path}
  metadata_csv: {csv_path}
validation:
  inspect_videos: true
""",
        encoding="utf-8",
    )

    inspect_result = subprocess.run(
        [sys.executable, "-m", "vietsign.cli", "dataset", "inspect", "--config", str(config_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    stats_result = subprocess.run(
        [sys.executable, "-m", "vietsign.cli", "dataset", "stats", "--config", str(config_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    validate_result = subprocess.run(
        [sys.executable, "-m", "vietsign.cli", "dataset", "validate", "--config", str(config_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(inspect_result.stdout)["total_samples"] == 1
    stats_payload = json.loads(stats_result.stdout)
    assert stats_payload["total_samples"] == 1
    assert stats_payload["duration"]["count"] == 1
    assert stats_payload["fps"]["count"] == 1
    assert stats_payload["resolutions"]["32x24"] == 1
    assert json.loads(validate_result.stdout)["issues"] == []
