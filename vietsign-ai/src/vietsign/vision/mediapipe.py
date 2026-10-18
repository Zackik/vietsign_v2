"""MediaPipe keypoint extraction backend."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np

from vietsign.vision.constants import HAND_JOINT_NAMES, POSE_JOINT_NAMES
from vietsign.vision.extractors import ExtractorUnavailableError, KeypointExtractor
from vietsign.vision.schema import FrameKeypoints, Keypoint

POSE_LANDMARK_INDEX = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}


@dataclass(frozen=True)
class MediaPipeConfig:
    """MediaPipe extractor options."""

    hands: bool = True
    pose: bool = True
    face: bool = False
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class PathLikeTempdir:
    """Small helper to keep MediaPipe's matplotlib cache out of unwritable homes."""

    @staticmethod
    def mpl_config_dir() -> str:
        path = os.path.join(tempfile.gettempdir(), "vietsign-mplconfig")
        os.makedirs(path, exist_ok=True)
        return path


class MediaPipeKeypointExtractor(KeypointExtractor):
    """MediaPipe-backed extractor with canonical output.

    This adapter prefers the legacy `mp.solutions.holistic` API when available.
    The installed MediaPipe 1.0.1 package in this environment exposes the newer
    tasks namespace but does not bundle task model assets, so the extractor can
    initialize in metadata-only mode and will return missing keypoints with a
    clear frame metadata status instead of leaking MediaPipe-specific objects.
    """

    name = "mediapipe"
    version = "mediapipe"

    def __init__(self, config: MediaPipeConfig | None = None) -> None:
        self.config = config or MediaPipeConfig()
        self._holistic: Any | None = None
        self._mp_version = "unknown"
        self._status = "unavailable"
        try:
            os.environ.setdefault("MPLCONFIGDIR", str(PathLikeTempdir.mpl_config_dir()))
            import mediapipe as mp  # type: ignore

            self._mp_version = getattr(mp, "__version__", "unknown")
            self.version = self._mp_version
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
                self._holistic = mp.solutions.holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=self.config.model_complexity,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    refine_face_landmarks=False,
                    min_detection_confidence=self.config.min_detection_confidence,
                    min_tracking_confidence=self.config.min_tracking_confidence,
                )
                self._status = "legacy_solutions"
            else:
                self._status = "no_supported_runtime_without_task_model"
        except ImportError:
            self._status = "mediapipe_not_installed"

    @property
    def status(self) -> str:
        return self._status

    @property
    def mediapipe_version(self) -> str:
        return self._mp_version

    def close(self) -> None:
        if self._holistic is not None:
            self._holistic.close()

    def extract_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        frame_index: int,
        timestamp: float,
    ) -> FrameKeypoints:
        if self._holistic is None:
            raise ExtractorUnavailableError(
                "Legacy MediaPipe solutions runtime is unavailable. Use extractor.type=mediapipe_tasks "
                "with explicit task model assets, or install a MediaPipe package exposing mp.solutions."
            )

        import cv2  # type: ignore

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        result = self._holistic.process(frame_rgb)

        left_hand = _landmark_list_to_keypoints(result.left_hand_landmarks, len(HAND_JOINT_NAMES))
        right_hand = _landmark_list_to_keypoints(result.right_hand_landmarks, len(HAND_JOINT_NAMES))
        pose = _pose_to_keypoints(result.pose_landmarks)
        return FrameKeypoints(
            frame_index=frame_index,
            timestamp=timestamp,
            left_hand=left_hand,
            right_hand=right_hand,
            pose=pose,
            metadata={
                "backend": self.name,
                "backend_status": self._status,
                "mediapipe_version": self._mp_version,
                "confidence_policy": "MediaPipe holistic exposes no per-hand-landmark confidence; confidence is None.",
            },
        )


def _landmark_list_to_keypoints(landmarks: Any, expected_count: int) -> tuple[Keypoint, ...]:
    if landmarks is None:
        return tuple(Keypoint.missing() for _ in range(expected_count))
    values = []
    for landmark in landmarks.landmark[:expected_count]:
        values.append(
            Keypoint(
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
                confidence=None,
                valid=True,
            )
        )
    return tuple(values)


def _pose_to_keypoints(landmarks: Any) -> tuple[Keypoint, ...]:
    if landmarks is None:
        return tuple(Keypoint.missing() for _ in POSE_JOINT_NAMES)
    values = []
    for name in POSE_JOINT_NAMES:
        landmark = landmarks.landmark[POSE_LANDMARK_INDEX[name]]
        visibility = getattr(landmark, "visibility", None)
        values.append(
            Keypoint(
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
                confidence=float(visibility) if visibility is not None else None,
                valid=True,
            )
        )
    return tuple(values)
