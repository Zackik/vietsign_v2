"""Video metadata inspection utilities."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from vietsign.data.schema import SignSample


class VideoMetadata(BaseModel):
    """Metadata extracted from a video container."""

    path: Path
    readable: bool
    duration: float | None = Field(default=None, ge=0.0)
    fps: float | None = Field(default=None, ge=0.0)
    frame_count: int | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    error: str | None = None


def inspect_video(path: str | Path) -> VideoMetadata:
    """Inspect video metadata with OpenCV without decoding the full video."""

    video_path = Path(path)
    if not video_path.exists():
        return VideoMetadata(path=video_path, readable=False, error="missing file")

    try:
        import cv2  # type: ignore
    except ImportError as error:
        return VideoMetadata(path=video_path, readable=False, error=str(error))

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return VideoMetadata(path=video_path, readable=False, error="unreadable video")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count >= 0 else None

        if frame_count <= 0 or width <= 0 or height <= 0:
            return VideoMetadata(
                path=video_path,
                readable=False,
                fps=fps if fps > 0 else None,
                frame_count=frame_count if frame_count >= 0 else None,
                width=width if width > 0 else None,
                height=height if height > 0 else None,
                error="video has invalid metadata",
            )

        return VideoMetadata(
            path=video_path,
            readable=True,
            duration=duration,
            fps=fps if fps > 0 else None,
            frame_count=frame_count,
            width=width,
            height=height,
        )
    finally:
        capture.release()


def enrich_sample_with_video_metadata(sample: SignSample) -> SignSample:
    """Return a sample copy with video metadata filled when the video is readable."""

    metadata = inspect_video(sample.video_path)
    if not metadata.readable:
        return sample
    return sample.model_copy(
        update={
            "duration": metadata.duration,
            "fps": metadata.fps,
            "frame_count": metadata.frame_count,
            "width": metadata.width,
            "height": metadata.height,
        }
    )
