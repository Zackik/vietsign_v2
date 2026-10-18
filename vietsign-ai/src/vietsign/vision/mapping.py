"""Backend-specific landmark mapping into VietSignAI canonical joints."""

from __future__ import annotations

from typing import Any

from vietsign.vision.constants import HAND_JOINT_NAMES, POSE_JOINT_NAMES
from vietsign.vision.schema import Keypoint

MEDIAPIPE_POSE_INDEX: dict[str, int] = {
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

COCO_POSE_INDEX: dict[str, int] = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}

RTMPOSE_OPENPOSE_BODY_INDEX: dict[str, int] = {
    "nose": 0,
    "left_eye": 15,
    "right_eye": 14,
    "left_ear": 17,
    "right_ear": 16,
    "left_shoulder": 5,
    "right_shoulder": 2,
    "left_elbow": 6,
    "right_elbow": 3,
    "left_wrist": 7,
    "right_wrist": 4,
    "left_hip": 12,
    "right_hip": 9,
}
RTMPOSE_LEFT_HAND_SLICE = slice(92, 113)
RTMPOSE_RIGHT_HAND_SLICE = slice(113, 134)


def mediapipe_hand_to_keypoints(landmarks: list[Any] | None) -> tuple[Keypoint, ...]:
    """Map MediaPipe's 21 hand landmarks to canonical hand joints."""

    if not landmarks:
        return tuple(Keypoint.missing() for _ in HAND_JOINT_NAMES)
    return tuple(_landmark_to_keypoint(landmark, confidence=None) for landmark in landmarks[:21])


def mediapipe_pose_to_keypoints(landmarks: list[Any] | None) -> tuple[Keypoint, ...]:
    """Map MediaPipe pose landmarks to canonical upper-body pose joints."""

    if not landmarks:
        return tuple(Keypoint.missing() for _ in POSE_JOINT_NAMES)
    values = []
    for name in POSE_JOINT_NAMES:
        index = MEDIAPIPE_POSE_INDEX[name]
        if index >= len(landmarks):
            values.append(Keypoint.missing())
            continue
        landmark = landmarks[index]
        confidence = getattr(landmark, "visibility", None)
        if confidence is None:
            confidence = getattr(landmark, "presence", None)
        values.append(_landmark_to_keypoint(landmark, confidence=confidence))
    return tuple(values)


def rtmpose_coco_to_pose_keypoints(
    keypoints: list[list[float]] | None,
    scores: list[float] | None = None,
) -> tuple[Keypoint, ...]:
    """Map COCO-style RTMPose body landmarks to canonical pose joints."""

    if not keypoints:
        return tuple(Keypoint.missing() for _ in POSE_JOINT_NAMES)
    values = []
    for name in POSE_JOINT_NAMES:
        index = COCO_POSE_INDEX[name]
        if index >= len(keypoints):
            values.append(Keypoint.missing())
            continue
        point = keypoints[index]
        confidence = scores[index] if scores is not None and index < len(scores) else None
        values.append(
            Keypoint(
                x=float(point[0]),
                y=float(point[1]),
                z=float(point[2]) if len(point) > 2 else 0.0,
                confidence=float(confidence) if confidence is not None else None,
                valid=True,
            )
        )
    return tuple(values)

def rtmpose_hand_to_keypoints(
    keypoints: Any | None,
    scores: Any | None = None,
    *,
    threshold: float = 0.3,
) -> tuple[Keypoint, ...]:
    """Map one RTMLib/OpenPose hand part into canonical 21 hand joints."""

    if keypoints is None or len(keypoints) == 0:
        return tuple(Keypoint.missing() for _ in HAND_JOINT_NAMES)
    values = []
    for index in range(len(HAND_JOINT_NAMES)):
        if index >= len(keypoints):
            values.append(Keypoint.missing())
            continue
        point = keypoints[index]
        confidence = _score_at(scores, index)
        values.append(_point_to_keypoint(point, confidence=confidence, threshold=threshold))
    return tuple(values)

def rtmpose_openpose_to_pose_keypoints(
    keypoints: Any | None,
    scores: Any | None = None,
    *,
    threshold: float = 0.3,
) -> tuple[Keypoint, ...]:
    """Map RTMLib `to_openpose=True` body landmarks into canonical upper-body joints."""

    if keypoints is None or len(keypoints) == 0:
        return tuple(Keypoint.missing() for _ in POSE_JOINT_NAMES)
    values = []
    for name in POSE_JOINT_NAMES:
        index = RTMPOSE_OPENPOSE_BODY_INDEX[name]
        if index >= len(keypoints):
            values.append(Keypoint.missing())
            continue
        point = keypoints[index]
        confidence = _score_at(scores, index)
        values.append(_point_to_keypoint(point, confidence=confidence, threshold=threshold))
    return tuple(values)


def _landmark_to_keypoint(landmark: Any, *, confidence: float | None) -> Keypoint:
    return Keypoint(
        x=float(landmark.x),
        y=float(landmark.y),
        z=float(getattr(landmark, "z", 0.0)),
        confidence=float(confidence) if confidence is not None else None,
        valid=True,
    )

def _score_at(scores: Any | None, index: int) -> float | None:
    if scores is None or index >= len(scores):
        return None
    return float(scores[index])

def _point_to_keypoint(point: Any, *, confidence: float | None, threshold: float) -> Keypoint:
    valid = confidence is None or confidence >= threshold
    if not valid:
        return Keypoint(x=float(point[0]), y=float(point[1]), z=0.0, confidence=confidence, valid=False)
    return Keypoint(
        x=float(point[0]),
        y=float(point[1]),
        z=float(point[2]) if len(point) > 2 else 0.0,
        confidence=confidence,
        valid=True,
    )
