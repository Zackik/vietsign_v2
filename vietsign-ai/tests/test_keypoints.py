from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vietsign.data.schema import SignSample
from vietsign.preprocessing.cache import (
    CacheMismatchError,
    cache_paths,
    load_keypoint_cache,
    preprocessing_fingerprint,
    save_keypoint_cache,
)
from vietsign.preprocessing.config import PreprocessingConfig
from vietsign.vision.constants import (
    CANONICAL_JOINT_NAMES,
    HAND_JOINT_INDEX,
    HAND_JOINT_NAMES,
    POSE_JOINT_NAMES,
)
from vietsign.vision.extractors import ExtractorUnavailableError
from vietsign.vision.mediapipe import MediaPipeKeypointExtractor
from vietsign.vision.processing import (
    compute_velocity,
    interpolate_short_gaps,
    normalize_tensors,
)
from vietsign.vision.schema import (
    INTERPOLATED_SOURCE,
    FrameKeypoints,
    Keypoint,
    KeypointSequence,
)


def _sequence(frames: list[FrameKeypoints]) -> KeypointSequence:
    return KeypointSequence(
        frames=tuple(frames),
        fps=5.0,
        source_video=Path("synthetic.mp4"),
        extractor="test",
    )


def _video(path: Path) -> None:
    import cv2  # type: ignore

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (32, 24))
    assert writer.isOpened()
    try:
        for index in range(4):
            writer.write(np.full((24, 32, 3), index * 40, dtype=np.uint8))
    finally:
        writer.release()


def test_hand_joint_order_contract() -> None:
    assert len(HAND_JOINT_NAMES) == 21
    assert HAND_JOINT_INDEX["wrist"] == 0
    assert HAND_JOINT_INDEX["pinky_tip"] == 20
    assert len(POSE_JOINT_NAMES) == 13
    assert len(CANONICAL_JOINT_NAMES) == 55


def test_sequence_tensor_shape_and_missing_masks() -> None:
    frame = FrameKeypoints(
        frame_index=0,
        timestamp=0.0,
        left_hand=(Keypoint(0.1, 0.2, 0.3, confidence=None, valid=True),),
    )
    tensors = _sequence([frame]).to_tensors()

    assert tensors.coordinates.shape == (1, 55, 3)
    assert tensors.valid_mask.shape == (1, 55)
    assert tensors.confidence.shape == (1, 55)
    assert tensors.valid_mask[0, 0]
    assert not tensors.valid_mask[0, 1]
    assert np.allclose(tensors.coordinates[0, 1], 0.0)
    assert np.isnan(tensors.confidence[0, 0])


def test_body_relative_normalization_and_zero_scale_protection() -> None:
    pose = [Keypoint.missing() for _ in POSE_JOINT_NAMES]
    pose[5] = Keypoint(0.0, 0.0, 0.0, valid=True)
    pose[6] = Keypoint(2.0, 0.0, 0.0, valid=True)
    frame = FrameKeypoints(
        frame_index=0,
        timestamp=0.0,
        left_hand=(Keypoint(2.0, 0.0, 0.0, valid=True),),
        pose=tuple(pose),
    )
    normalized = normalize_tensors(_sequence([frame]).to_tensors(), method="body_relative")

    assert np.allclose(normalized.coordinates[0, 0], [0.5, 0.0, 0.0])

    pose[6] = Keypoint(0.0, 0.0, 0.0, valid=True)
    protected = normalize_tensors(
        _sequence([FrameKeypoints(frame_index=0, timestamp=0.0, pose=tuple(pose))]).to_tensors(),
        method="body_relative",
    )
    assert np.allclose(protected.coordinates[0, 42 + 5], [0.0, 0.0, 0.0])


def test_short_gap_interpolation_and_large_gap_behavior() -> None:
    frames = [
        FrameKeypoints(0, 0.0, left_hand=(Keypoint(0.0, 0.0, 0.0, valid=True),)),
        FrameKeypoints(1, 0.2),
        FrameKeypoints(2, 0.4, left_hand=(Keypoint(2.0, 0.0, 0.0, valid=True),)),
        FrameKeypoints(3, 0.6),
        FrameKeypoints(4, 0.8),
        FrameKeypoints(5, 1.0),
        FrameKeypoints(6, 1.2, left_hand=(Keypoint(6.0, 0.0, 0.0, valid=True),)),
    ]
    tensors = interpolate_short_gaps(_sequence(frames).to_tensors(), max_gap=1)

    assert tensors.valid_mask[1, 0]
    assert tensors.source_mask[1, 0] == INTERPOLATED_SOURCE
    assert np.allclose(tensors.coordinates[1, 0], [1.0, 0.0, 0.0])
    assert not tensors.valid_mask[3, 0]
    assert not tensors.valid_mask[4, 0]
    assert not tensors.valid_mask[5, 0]


def test_velocity_respects_validity_masks() -> None:
    frames = [
        FrameKeypoints(0, 0.0, left_hand=(Keypoint(1.0, 0.0, 0.0, valid=True),)),
        FrameKeypoints(1, 0.2, left_hand=(Keypoint(3.0, 0.0, 0.0, valid=True),)),
        FrameKeypoints(2, 0.4),
    ]
    tensors = _sequence(frames).to_tensors()
    velocity = compute_velocity(tensors)

    assert np.allclose(velocity[1, 0], [2.0, 0.0, 0.0])
    assert np.allclose(velocity[2, 0], [0.0, 0.0, 0.0])


def test_cache_fingerprint_save_load_and_mismatch(tmp_path: Path) -> None:
    config = PreprocessingConfig(dataset_config=Path("configs/data/vsl400.yaml"))
    sample = SignSample(
        sample_id="sample",
        video_path=Path("sample.mp4"),
        label="hello",
        label_id=0,
        split="train",
        dataset="unit",
    )
    tensors = _sequence([FrameKeypoints(0, 0.0)]).to_tensors()
    npz_path, json_path = cache_paths(tmp_path, sample)
    fingerprint = preprocessing_fingerprint(config)
    save_keypoint_cache(
        npz_path,
        json_path,
        tensors,
        sample=sample,
        metadata={"fingerprint": fingerprint, "coverage": {}},
    )

    arrays, metadata = load_keypoint_cache(npz_path, json_path, expected_fingerprint=fingerprint)

    assert arrays["coordinates"].shape == (1, 55, 3)
    assert metadata["fingerprint"] == fingerprint
    with pytest.raises(CacheMismatchError):
        load_keypoint_cache(npz_path, json_path, expected_fingerprint="stale")


def test_mediapipe_adapter_initializes() -> None:
    extractor = MediaPipeKeypointExtractor()
    try:
        if extractor.status == "legacy_solutions":
            frame = extractor.extract_frame(np.zeros((24, 32, 3), dtype=np.uint8), frame_index=0, timestamp=0.0)
            assert frame.metadata["backend"] == "mediapipe"
            assert len(frame.left_hand) == 21
            assert len(frame.right_hand) == 21
        else:
            with pytest.raises(ExtractorUnavailableError):
                extractor.extract_frame(np.zeros((24, 32, 3), dtype=np.uint8), frame_index=0, timestamp=0.0)
    finally:
        extractor.close()


def test_preprocess_inspect_and_visualize_cli(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    _video(video_path)
    config_path = tmp_path / "preprocess.yaml"
    cache_root = tmp_path / "cache"
    config_path.write_text(
        f"""
dataset_config: configs/data/vsl400.yaml
extractor:
  type: mediapipe
  hands: true
  pose: true
  face: false
video:
  target_fps: 5
  num_frames: 4
  sampling: uniform
normalization:
  type: body_relative
interpolation:
  enabled: true
  max_gap: 1
features:
  velocity: true
  acceleration: false
cache:
  enabled: true
  root: {cache_root}
  reuse: false
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vietsign.cli",
            "preprocess",
            "inspect-video",
            str(video_path),
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "unavailable" in result.stdout.lower() or "unavailable" in result.stderr.lower()

    sample = SignSample(
        sample_id="visual",
        video_path=video_path,
        label="synthetic",
        label_id=0,
        split="test",
        dataset="unit",
    )
    tensors = _sequence(
        [
            FrameKeypoints(0, 0.0, left_hand=(Keypoint(0.2, 0.2, 0.0, valid=True),)),
            FrameKeypoints(1, 0.2, left_hand=(Keypoint(0.4, 0.4, 0.0, valid=True),)),
        ]
    ).to_tensors()
    npz_path, json_path = cache_paths(tmp_path / "viz-cache", sample)
    save_keypoint_cache(npz_path, json_path, tensors, sample=sample, metadata={"fingerprint": "test"})

    arrays, _ = load_keypoint_cache(npz_path, json_path)
    assert arrays["valid_mask"].shape == (2, 55)

    vis_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vietsign.cli",
            "visualize",
            "keypoints",
            "--features",
            str(npz_path),
            "--output-dir",
            str(tmp_path / "viz"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    outputs = json.loads(vis_result.stdout)["outputs"]
    assert outputs
    assert Path(outputs[0]).exists()
