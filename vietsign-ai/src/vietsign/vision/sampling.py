"""Temporal frame sampling utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def sample_video_frames(
    video_path: str | Path,
    *,
    target_fps: float | None = None,
    num_frames: int | None = None,
    sampling: str = "uniform",
) -> list[tuple[int, float, np.ndarray]]:
    """Sample frames as (frame_index, timestamp_seconds, BGR frame)."""

    if sampling != "uniform":
        raise ValueError("Only uniform sampling is implemented in Phase 3.")

    import cv2  # type: ignore

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            raise ValueError(f"Video has no readable frames: {video_path}")

        indices = _select_indices(frame_count, fps=fps, target_fps=target_fps, num_frames=num_frames)
        frames: list[tuple[int, float, np.ndarray]] = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                continue
            timestamp = (float(frame_index) / fps) if fps > 0 else float(len(frames))
            frames.append((int(frame_index), timestamp, frame))
        return frames
    finally:
        capture.release()


def _select_indices(
    frame_count: int,
    *,
    fps: float,
    target_fps: float | None,
    num_frames: int | None,
) -> np.ndarray:
    if num_frames is not None and num_frames > 0:
        return np.linspace(0, frame_count - 1, num=min(num_frames, frame_count), dtype=np.int64)
    if target_fps is not None and target_fps > 0 and fps > target_fps:
        step = max(round(fps / target_fps), 1)
        return np.arange(0, frame_count, step, dtype=np.int64)
    return np.arange(frame_count, dtype=np.int64)
