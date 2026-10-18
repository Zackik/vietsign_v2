"""Headless keypoint visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vietsign.vision.constants import LEFT_HAND_SLICE, POSE_SLICE, RIGHT_HAND_SLICE

HAND_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)
POSE_EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12))


def visualize_keypoint_npz(
    npz_path: str | Path,
    *,
    output_dir: str | Path = "outputs/visualizations",
    max_frames: int = 8,
    canvas_size: tuple[int, int] = (512, 512),
) -> list[Path]:
    """Render sampled keypoint frames to PNG files without requiring a GUI."""

    import cv2  # type: ignore

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path, allow_pickle=False) as data:
        coordinates = data["coordinates"]
        valid_mask = data["valid_mask"]

    frame_count = coordinates.shape[0]
    if frame_count == 0:
        return []
    indices = np.linspace(0, frame_count - 1, num=min(max_frames, frame_count), dtype=np.int64)
    written: list[Path] = []
    for frame_index in indices:
        canvas = np.full((canvas_size[1], canvas_size[0], 3), 255, dtype=np.uint8)
        _draw_region(canvas, coordinates[frame_index], valid_mask[frame_index], LEFT_HAND_SLICE, HAND_EDGES, (0, 120, 255))
        _draw_region(canvas, coordinates[frame_index], valid_mask[frame_index], RIGHT_HAND_SLICE, HAND_EDGES, (255, 80, 0))
        _draw_region(canvas, coordinates[frame_index], valid_mask[frame_index], POSE_SLICE, POSE_EDGES, (80, 180, 80))
        path = output / f"{Path(npz_path).stem}_frame_{int(frame_index):04d}.png"
        cv2.imwrite(str(path), canvas)
        written.append(path)
    return written


def _draw_region(
    canvas: np.ndarray,
    coordinates: np.ndarray,
    valid_mask: np.ndarray,
    region: slice,
    edges: tuple[tuple[int, int], ...],
    color: tuple[int, int, int],
) -> None:
    import cv2  # type: ignore

    height, width = canvas.shape[:2]
    points: dict[int, tuple[int, int]] = {}
    for local_index, joint_index in enumerate(range(region.start, region.stop)):
        if not valid_mask[joint_index]:
            continue
        x = float(coordinates[joint_index, 0])
        y = float(coordinates[joint_index, 1])
        if abs(x) <= 4 and abs(y) <= 4:
            px = int((x + 2.0) / 4.0 * width)
            py = int((y + 2.0) / 4.0 * height)
        else:
            px = int(x * width) if abs(x) <= 1.5 else int(x)
            py = int(y * height) if abs(y) <= 1.5 else int(y)
        points[local_index] = (max(0, min(width - 1, px)), max(0, min(height - 1, py)))
        cv2.circle(canvas, points[local_index], 3, color, -1)
    for start, end in edges:
        if start in points and end in points:
            cv2.line(canvas, points[start], points[end], color, 1)

