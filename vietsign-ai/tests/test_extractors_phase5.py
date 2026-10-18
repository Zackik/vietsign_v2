from __future__ import annotations

import numpy as np
import pytest

from vietsign.preprocessing.cache import preprocessing_fingerprint
from vietsign.preprocessing.config import PreprocessingConfig
from vietsign.preprocessing.pipeline import create_extractor
from vietsign.vision.cli import _real_extraction_gate
from vietsign.vision.doctor import extractor_doctor
from vietsign.vision.extractors import ExtractorUnavailableError
from vietsign.vision.mapping import (
    mediapipe_hand_to_keypoints,
    rtmpose_coco_to_pose_keypoints,
    rtmpose_hand_to_keypoints,
    rtmpose_openpose_to_pose_keypoints,
)
from vietsign.vision.mediapipe_tasks import (
    MediaPipeTasksConfig,
    MediaPipeTasksKeypointExtractor,
    bgr_to_rgb,
    canonical_handedness_label,
)
from vietsign.vision.quality import aggregate_quality, sequence_quality, smooth_ema
from vietsign.vision.rtmpose import select_primary_person
from vietsign.vision.schema import KeypointTensors


class Landmark:
    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 1.0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def _tensors() -> KeypointTensors:
    coordinates = np.zeros((3, 55, 3), dtype=np.float32)
    valid_mask = np.zeros((3, 55), dtype=bool)
    coordinates[:, 0, 0] = [0.0, 1.0, 2.0]
    valid_mask[:, 0] = True
    return KeypointTensors(
        coordinates=coordinates,
        valid_mask=valid_mask,
        confidence=np.full((3, 55), np.nan, dtype=np.float32),
        source_mask=valid_mask.astype(np.uint8),
        timestamps=np.arange(3, dtype=np.float32),
        frame_indices=np.arange(3, dtype=np.int64),
        joint_names=tuple(str(index) for index in range(55)),
    )


def test_extractor_doctor_reports_missing_assets() -> None:
    config = PreprocessingConfig.model_validate(
        {
            "dataset_config": "configs/data/vsl400.yaml",
            "extractor": {
                "type": "mediapipe_tasks",
                "hand_model": "models/mediapipe/missing_hand.task",
                "pose_model": "models/mediapipe/missing_pose.task",
            },
        }
    )

    report = extractor_doctor(config)

    assert report["mediapipe"]["tasks_api"] is True
    assert report["mediapipe"]["hand_model"]["exists"] is False
    assert report["ready_for_preprocessing"] is False


def test_mediapipe_tasks_missing_asset_error() -> None:
    config = PreprocessingConfig.model_validate(
        {
            "dataset_config": "configs/data/vsl400.yaml",
            "extractor": {
                "type": "mediapipe_tasks",
                "hand_model": "models/mediapipe/missing_hand.task",
                "pose_model": "models/mediapipe/missing_pose.task",
            },
        }
    )

    with pytest.raises(ExtractorUnavailableError, match="not ready"):
        create_extractor(config)


def test_rtmpose_missing_asset_error() -> None:
    config = PreprocessingConfig.model_validate(
        {
            "dataset_config": "configs/data/vsl400.yaml",
            "extractor": {"type": "rtmpose"},
            "rtmpose": {
                "det_model": "models/rtmpose/missing-det.onnx",
                "pose_model": "models/rtmpose/missing-pose.onnx",
            },
        }
    )

    with pytest.raises(ExtractorUnavailableError, match="model is missing"):
        create_extractor(config)


def test_canonical_mapping_utilities() -> None:
    hand = mediapipe_hand_to_keypoints([Landmark(float(index), 0.0) for index in range(21)])
    pose = rtmpose_coco_to_pose_keypoints([[float(index), 0.0, 0.0] for index in range(17)])

    assert len(hand) == 21
    assert hand[20].x == 20.0
    assert len(pose) == 13
    assert pose[0].valid

def test_rtmpose_openpose_mapping_uses_documented_slices() -> None:
    keypoints = np.asarray([[float(index), float(index + 1)] for index in range(134)])
    scores = np.ones((134,), dtype=np.float32)
    scores[92] = 0.1

    left_hand = rtmpose_hand_to_keypoints(keypoints[92:113], scores[92:113], threshold=0.3)
    pose = rtmpose_openpose_to_pose_keypoints(keypoints, scores, threshold=0.3)

    assert len(left_hand) == 21
    assert left_hand[0].x == 92.0
    assert left_hand[0].valid is False
    assert left_hand[1].valid is True
    assert len(pose) == 13
    assert pose[0].x == 0.0
    assert pose[5].x == 5.0
    assert pose[6].x == 2.0

def test_rtmpose_primary_person_prefers_higher_quality() -> None:
    keypoints = np.zeros((2, 134, 2), dtype=np.float32)
    scores = np.zeros((2, 134), dtype=np.float32)
    keypoints[0, :, 0] = np.linspace(0, 100, 134)
    keypoints[0, :, 1] = np.linspace(0, 100, 134)
    scores[0, :] = 0.2
    keypoints[1, :, 0] = np.linspace(0, 300, 134)
    keypoints[1, :, 1] = np.linspace(0, 300, 134)
    scores[1, :] = 0.8

    assert select_primary_person(keypoints, scores) == 1


def test_cache_fingerprint_includes_backend() -> None:
    mediapipe_config = PreprocessingConfig.model_validate(
        {"dataset_config": "configs/data/vsl400.yaml", "extractor": {"type": "mediapipe_tasks"}}
    )
    rtmpose_config = PreprocessingConfig.model_validate(
        {"dataset_config": "configs/data/vsl400.yaml", "extractor": {"type": "rtmpose"}}
    )

    assert preprocessing_fingerprint(mediapipe_config) != preprocessing_fingerprint(rtmpose_config)

def test_cache_fingerprint_includes_rtmpose_model_checksums(tmp_path) -> None:
    det_model = tmp_path / "det.onnx"
    pose_model = tmp_path / "pose.onnx"
    det_model.write_bytes(b"det-v1")
    pose_model.write_bytes(b"pose-v1")
    config = PreprocessingConfig.model_validate(
        {
            "dataset_config": "configs/data/vsl400.yaml",
            "extractor": {"type": "rtmpose"},
            "rtmpose": {"det_model": det_model, "pose_model": pose_model},
        }
    )
    first = preprocessing_fingerprint(config)

    det_model.write_bytes(b"det-v2")
    second = preprocessing_fingerprint(config)

    assert first != second

def test_acceptance_gate_rejects_empty_or_handless_outputs() -> None:
    arrays = {
        "coordinates": np.zeros((4, 55, 3), dtype=np.float32),
        "valid_mask": np.zeros((4, 55), dtype=bool),
    }

    assert _real_extraction_gate(arrays=arrays, coverage={}, extractor_ready=True)["ok"] is False
    arrays["valid_mask"][0, 42] = True
    assert (
        _real_extraction_gate(
            arrays=arrays,
            coverage={"pose_coverage": 0.5, "left_hand_coverage": 0.0, "right_hand_coverage": 0.0},
            extractor_ready=True,
        )["ok"]
        is False
    )
    assert (
        _real_extraction_gate(
            arrays=arrays,
            coverage={"pose_coverage": 0.5, "left_hand_coverage": 0.25, "right_hand_coverage": 0.0},
            extractor_ready=True,
        )["ok"]
        is True
    )


def test_quality_statistics_and_smoothing_respect_masks() -> None:
    tensors = _tensors()
    quality = sequence_quality(tensors)
    aggregate = aggregate_quality([quality])
    smoothed = smooth_ema(tensors, alpha=0.5)

    assert quality["overall_valid_joint_ratio"] > 0
    assert aggregate["overall_valid_joint_ratio"]["mean"] == quality["overall_valid_joint_ratio"]
    assert np.allclose(smoothed.coordinates[0, 0], tensors.coordinates[0, 0])
    assert smoothed.valid_mask.shape == tensors.valid_mask.shape


def test_bgr_to_rgb_conversion() -> None:
    bgr = np.asarray([[[10, 20, 30]]], dtype=np.uint8)
    rgb = bgr_to_rgb(bgr)

    assert rgb.tolist() == [[[30, 20, 10]]]


def test_mirrored_handedness_mapping() -> None:
    assert canonical_handedness_label("Left", mirrored=False) == "left"
    assert canonical_handedness_label("Left", mirrored=True) == "right"
    assert canonical_handedness_label("Right", mirrored=True) == "left"


def test_monotonic_timestamp_guard_without_model_init() -> None:
    extractor = object.__new__(MediaPipeTasksKeypointExtractor)
    extractor.config = MediaPipeTasksConfig()

    assert extractor._timestamp_ms(frame_index=10, timestamp=1.0) == 1000
    assert extractor._timestamp_ms(frame_index=5, timestamp=0.5) == 1001
