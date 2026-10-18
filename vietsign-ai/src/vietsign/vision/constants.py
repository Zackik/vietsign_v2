"""Canonical keypoint names and tensor layout."""

from __future__ import annotations

HAND_JOINT_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
HAND_JOINT_INDEX: dict[str, int] = {name: index for index, name in enumerate(HAND_JOINT_NAMES)}

POSE_JOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)
POSE_JOINT_INDEX: dict[str, int] = {name: index for index, name in enumerate(POSE_JOINT_NAMES)}

LEFT_HAND_PREFIX = "left_hand"
RIGHT_HAND_PREFIX = "right_hand"
POSE_PREFIX = "pose"

CANONICAL_JOINT_NAMES: tuple[str, ...] = (
    tuple(f"{LEFT_HAND_PREFIX}:{name}" for name in HAND_JOINT_NAMES)
    + tuple(f"{RIGHT_HAND_PREFIX}:{name}" for name in HAND_JOINT_NAMES)
    + tuple(f"{POSE_PREFIX}:{name}" for name in POSE_JOINT_NAMES)
)
CANONICAL_JOINT_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(CANONICAL_JOINT_NAMES)
}

LEFT_HAND_SLICE = slice(0, 21)
RIGHT_HAND_SLICE = slice(21, 42)
POSE_SLICE = slice(42, 55)

COORDINATE_CHANNELS: tuple[str, ...] = ("x", "y", "z")
SCHEMA_VERSION = "keypoints.v1"

