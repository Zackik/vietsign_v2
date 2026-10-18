"""MediaPipe Tasks extractor backend."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vietsign.vision.assets import resolve_repo_path
from vietsign.vision.extractors import ExtractorUnavailableError, KeypointExtractor
from vietsign.vision.mapping import mediapipe_hand_to_keypoints, mediapipe_pose_to_keypoints
from vietsign.vision.schema import FrameKeypoints


@dataclass(frozen=True)
class MediaPipeTasksConfig:
    hands: bool = True
    pose: bool = True
    face: bool = False
    hand_model: Path | None = None
    pose_model: Path | None = None
    face_model: Path | None = None
    mirrored: bool = False


class MediaPipeTasksKeypointExtractor(KeypointExtractor):
    """MediaPipe Tasks backend requiring explicit task model assets."""

    name = "mediapipe_tasks"

    def __init__(self, config: MediaPipeTasksConfig) -> None:
        os.environ.setdefault("MPLCONFIGDIR", _mpl_config_dir())
        self.config = config
        self._mp_version = "unknown"
        self._hand_landmarker: Any | None = None
        self._pose_landmarker: Any | None = None
        self._status = "uninitialized"
        try:
            import mediapipe as mp  # type: ignore
            from mediapipe.tasks.python import BaseOptions  # type: ignore
            from mediapipe.tasks.python.vision import (  # type: ignore
                HandLandmarker,
                HandLandmarkerOptions,
                PoseLandmarker,
                PoseLandmarkerOptions,
                RunningMode,
            )

            self._mp = mp
            self._mp_version = getattr(mp, "__version__", "unknown")
            self.version = self._mp_version
            if config.hands:
                hand_model = resolve_repo_path(config.hand_model) if config.hand_model is not None else None
                if hand_model is None or not hand_model.exists() or hand_model.stat().st_size <= 0:
                    raise ExtractorUnavailableError(
                        "MediaPipe hand task model is missing. Set extractor.hand_model to a "
                        "valid hand_landmarker.task file."
                    )
                self._hand_landmarker = HandLandmarker.create_from_options(
                    HandLandmarkerOptions(
                        base_options=BaseOptions(
                            model_asset_path=str(hand_model),
                            delegate=BaseOptions.Delegate.CPU,
                        ),
                        running_mode=RunningMode.VIDEO,
                        num_hands=2,
                    )
                )
            if config.pose:
                pose_model = resolve_repo_path(config.pose_model) if config.pose_model is not None else None
                if pose_model is None or not pose_model.exists() or pose_model.stat().st_size <= 0:
                    raise ExtractorUnavailableError(
                        "MediaPipe pose task model is missing. Set extractor.pose_model to a "
                        "valid pose_landmarker.task file."
                    )
                self._pose_landmarker = PoseLandmarker.create_from_options(
                    PoseLandmarkerOptions(
                        base_options=BaseOptions(
                            model_asset_path=str(pose_model),
                            delegate=BaseOptions.Delegate.CPU,
                        ),
                        running_mode=RunningMode.VIDEO,
                        num_poses=1,
                    )
                )
            self._status = "ready"
        except ImportError as error:
            raise ExtractorUnavailableError(f"MediaPipe Tasks API is not importable: {error}") from error

    @property
    def status(self) -> str:
        return self._status

    @property
    def mediapipe_version(self) -> str:
        return self._mp_version

    def close(self) -> None:
        for landmarker in (self._hand_landmarker, self._pose_landmarker):
            if landmarker is not None:
                landmarker.close()

    def extract_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        frame_index: int,
        timestamp: float,
    ) -> FrameKeypoints:
        frame_rgb = bgr_to_rgb(frame_bgr)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = self._timestamp_ms(frame_index=frame_index, timestamp=timestamp)
        left_hand = None
        right_hand = None
        handedness_metadata: dict[str, float] = {}
        if self._hand_landmarker is not None:
            hand_result = self._hand_landmarker.detect_for_video(image, timestamp_ms)
            for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                category = handedness[0]
                label = canonical_handedness_label(category.category_name, mirrored=self.config.mirrored)
                score = float(category.score)
                if label == "left":
                    left_hand = landmarks
                    handedness_metadata["left_handedness_confidence"] = score
                elif label == "right":
                    right_hand = landmarks
                    handedness_metadata["right_handedness_confidence"] = score
        pose = None
        if self._pose_landmarker is not None:
            pose_result = self._pose_landmarker.detect_for_video(image, timestamp_ms)
            if pose_result.pose_landmarks:
                pose = pose_result.pose_landmarks[0]
        return FrameKeypoints(
            frame_index=frame_index,
            timestamp=timestamp,
            left_hand=mediapipe_hand_to_keypoints(left_hand),
            right_hand=mediapipe_hand_to_keypoints(right_hand),
            pose=mediapipe_pose_to_keypoints(pose),
            metadata={
                "backend": self.name,
                "backend_status": self._status,
                "mediapipe_version": self._mp_version,
                "timestamp_ms": timestamp_ms,
                "input_mirrored": self.config.mirrored,
                **handedness_metadata,
            },
        )

    def _timestamp_ms(self, *, frame_index: int, timestamp: float) -> int:
        candidate = round(timestamp * 1000.0)
        previous = getattr(self, "_last_timestamp_ms", None)
        if previous is not None and candidate <= previous:
            candidate = previous + 1
        self._last_timestamp_ms = candidate
        return candidate


def _mpl_config_dir() -> str:
    path = os.path.join(tempfile.gettempdir(), "vietsign-mplconfig")
    os.makedirs(path, exist_ok=True)
    return path


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert one OpenCV BGR frame to RGB for MediaPipe."""

    import cv2  # type: ignore

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def canonical_handedness_label(label: str, *, mirrored: bool) -> str:
    normalized = label.lower()
    if mirrored:
        if normalized == "left":
            return "right"
        if normalized == "right":
            return "left"
    return normalized
