"""Backend-independent keypoint extractor interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from vietsign.vision.schema import FrameKeypoints, KeypointSequence


class ExtractorUnavailableError(RuntimeError):
    """Raised when an extractor cannot run actual landmark inference."""


class KeypointExtractor(ABC):
    """Common interface for visual keypoint extractors."""

    name = "base"
    version = "0"

    @abstractmethod
    def extract_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        frame_index: int,
        timestamp: float,
    ) -> FrameKeypoints:
        """Extract canonical keypoints from one BGR frame."""

    def extract_video(
        self,
        frames: list[tuple[int, float, np.ndarray]],
        *,
        source_video: str | Path,
        fps: float | None,
    ) -> KeypointSequence:
        """Extract a canonical sequence from sampled frames."""

        return KeypointSequence(
            frames=tuple(
                self.extract_frame(frame, frame_index=frame_index, timestamp=timestamp)
                for frame_index, timestamp, frame in frames
            ),
            fps=fps,
            source_video=Path(source_video),
            extractor=self.name,
            metadata={"extractor_version": self.version},
        )
