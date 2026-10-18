"""RTMLib/RTMPose extractor backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vietsign.vision.assets import resolve_repo_path
from vietsign.vision.extractors import ExtractorUnavailableError, KeypointExtractor
from vietsign.vision.mapping import (
    RTMPOSE_LEFT_HAND_SLICE,
    RTMPOSE_RIGHT_HAND_SLICE,
    rtmpose_hand_to_keypoints,
    rtmpose_openpose_to_pose_keypoints,
)
from vietsign.vision.schema import FrameKeypoints

RTMPOSE_JOINT_COUNT = 134
RTMPOSE_MODE_INPUT_SIZES = {
    "performance": {"det": (640, 640), "pose": (288, 384)},
    "lightweight": {"det": (416, 416), "pose": (192, 256)},
    "balanced": {"det": (640, 640), "pose": (192, 256)},
}


@dataclass(frozen=True)
class RTMPoseConfig:
    runtime: str = "rtmlib"
    backend: str = "onnxruntime"
    device: str = "cpu"
    mode: str = "balanced"
    keypoint_threshold: float = 0.3
    det_model: Path | None = None
    pose_model: Path | None = None


def available_onnx_providers() -> tuple[str, ...]:
    """Return ONNX Runtime providers without requiring CUDA."""

    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        return ()
    return tuple(ort.get_available_providers())


def select_primary_person(keypoints: np.ndarray, scores: np.ndarray | None) -> int | None:
    """Select the most likely signer from RTMLib multi-person output."""

    if keypoints.size == 0:
        return None
    if keypoints.ndim == 2:
        return 0

    best_index: int | None = None
    best_score = -1.0
    for index, instance in enumerate(keypoints):
        instance_scores = scores[index] if scores is not None and index < len(scores) else None
        score = _instance_quality(instance, instance_scores)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


class RTMPoseKeypointExtractor(KeypointExtractor):
    """Real RTMLib Wholebody extractor mapped into VietSignAI's canonical tensors."""

    name = "rtmpose"
    version = "rtmlib"

    def __init__(self, config: RTMPoseConfig) -> None:
        self.config = config
        if config.runtime != "rtmlib":
            raise ExtractorUnavailableError(f"Unsupported RTMPose runtime: {config.runtime}")

        det_model = _resolve_existing_model(config.det_model, "RTMPose detector")
        pose_model = _resolve_existing_model(config.pose_model, "RTMPose pose")
        try:
            from rtmlib import Wholebody  # type: ignore
        except ImportError as error:
            raise ExtractorUnavailableError(
                "RTMLib is not installed. Install optional dependencies with "
                "`pip install -e '.[rtmpose]'`."
            ) from error

        try:
            input_sizes = rtmpose_input_sizes(config.mode)
            self._wholebody = Wholebody(
                det=str(det_model) if det_model is not None else None,
                det_input_size=input_sizes["det"],
                pose=str(pose_model) if pose_model is not None else None,
                pose_input_size=input_sizes["pose"],
                mode=config.mode,
                to_openpose=True,
                backend=config.backend,
                device=config.device,
            )
        except Exception as error:
            raise ExtractorUnavailableError(f"Failed to initialize RTMLib Wholebody: {error}") from error

        self._det_model = det_model
        self._pose_model = pose_model
        self._providers = available_onnx_providers()

    @property
    def status(self) -> str:
        return (
            f"runtime=rtmlib backend={self.config.backend} device={self.config.device} "
            f"mode={self.config.mode} providers={list(self._providers)}"
        )

    def close(self) -> None:
        self._wholebody = None

    def extract_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        frame_index: int,
        timestamp: float,
    ) -> FrameKeypoints:
        if self._wholebody is None:
            raise ExtractorUnavailableError("RTMPose extractor has been closed.")

        raw_keypoints, raw_scores = self._wholebody(frame_bgr)
        keypoints = np.asarray(raw_keypoints, dtype=np.float32)
        scores = np.asarray(raw_scores, dtype=np.float32) if raw_scores is not None else None

        if keypoints.size == 0:
            return self._missing_frame(frame_index=frame_index, timestamp=timestamp, person_count=0)
        if keypoints.ndim == 2:
            keypoints = keypoints[None, ...]
        if scores is not None and scores.ndim == 1:
            scores = scores[None, ...]

        selected_index = select_primary_person(keypoints, scores)
        if selected_index is None:
            return self._missing_frame(
                frame_index=frame_index,
                timestamp=timestamp,
                person_count=int(keypoints.shape[0]),
            )

        selected_keypoints = keypoints[selected_index]
        selected_scores = scores[selected_index] if scores is not None else None
        if selected_keypoints.shape[0] < RTMPOSE_JOINT_COUNT:
            raise ExtractorUnavailableError(
                f"RTMLib returned {selected_keypoints.shape[0]} joints; expected at least "
                f"{RTMPOSE_JOINT_COUNT} with to_openpose=True."
            )

        return FrameKeypoints(
            frame_index=frame_index,
            timestamp=timestamp,
            left_hand=rtmpose_hand_to_keypoints(
                selected_keypoints[RTMPOSE_LEFT_HAND_SLICE],
                _slice_scores(selected_scores, RTMPOSE_LEFT_HAND_SLICE),
                threshold=self.config.keypoint_threshold,
            ),
            right_hand=rtmpose_hand_to_keypoints(
                selected_keypoints[RTMPOSE_RIGHT_HAND_SLICE],
                _slice_scores(selected_scores, RTMPOSE_RIGHT_HAND_SLICE),
                threshold=self.config.keypoint_threshold,
            ),
            pose=rtmpose_openpose_to_pose_keypoints(
                selected_keypoints,
                selected_scores,
                threshold=self.config.keypoint_threshold,
            ),
            metadata={
                "backend": "rtmlib",
                "mode": self.config.mode,
                "provider": list(self._providers),
                "selected_person_index": selected_index,
                "person_count": int(keypoints.shape[0]),
                "keypoint_threshold": self.config.keypoint_threshold,
                "z_source": "unavailable_2d_backend",
                "det_model": str(self._det_model) if self._det_model else None,
                "pose_model": str(self._pose_model) if self._pose_model else None,
            },
        )

    def _missing_frame(self, *, frame_index: int, timestamp: float, person_count: int) -> FrameKeypoints:
        return FrameKeypoints(
            frame_index=frame_index,
            timestamp=timestamp,
            metadata={
                "backend": "rtmlib",
                "mode": self.config.mode,
                "provider": list(self._providers),
                "person_count": person_count,
                "z_source": "unavailable_2d_backend",
            },
        )


def _resolve_existing_model(path: Path | None, label: str) -> Path | None:
    if path is None:
        return None
    resolved = resolve_repo_path(path)
    if not resolved.exists() or resolved.stat().st_size <= 0:
        raise ExtractorUnavailableError(
            f"{label} model is missing at {resolved}. Run "
            "`python -m vietsign.cli extractor assets install rtmpose`."
        )
    return resolved


def rtmpose_input_sizes(mode: str) -> dict[str, tuple[int, int]]:
    try:
        return RTMPOSE_MODE_INPUT_SIZES[mode]
    except KeyError as error:
        choices = ", ".join(sorted(RTMPOSE_MODE_INPUT_SIZES))
        raise ExtractorUnavailableError(f"Unsupported RTMPose mode {mode!r}; choose one of: {choices}") from error


def _slice_scores(scores: np.ndarray | None, part: slice) -> np.ndarray | None:
    return None if scores is None else scores[part]


def _instance_quality(instance: np.ndarray, scores: np.ndarray | None) -> float:
    confidence = float(np.nanmean(scores)) if scores is not None and scores.size else 0.0
    valid_points = instance
    if scores is not None and scores.size:
        valid_points = instance[scores > 0.05]
    if len(valid_points) < 2:
        return confidence
    x_span = float(np.nanmax(valid_points[:, 0]) - np.nanmin(valid_points[:, 0]))
    y_span = float(np.nanmax(valid_points[:, 1]) - np.nanmin(valid_points[:, 1]))
    area_bonus = min((x_span * y_span) / 1_000_000.0, 1.0)
    return confidence + (0.1 * area_bonus)
