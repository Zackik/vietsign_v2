"""Upload and video validation helpers."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

import cv2  # type: ignore

from vietsign.inference.config import UploadLimits
from vietsign.inference.errors import InvalidVideoError, UnsupportedFormatError


def validate_extension(filename: str, limits: UploadLimits) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in {item.lower() for item in limits.allowed_extensions}:
        raise UnsupportedFormatError(f"Unsupported video extension: {suffix}")
    return suffix


@contextmanager
def safe_upload_file(fileobj: BinaryIO, filename: str, limits: UploadLimits) -> Iterator[Path]:
    suffix = validate_extension(filename, limits)
    limits.temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = limits.temp_dir / f"{uuid4().hex}{suffix}"
    max_bytes = limits.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with temp_path.open("wb") as output:
            while chunk := fileobj.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise InvalidVideoError("Uploaded file exceeds configured size limit.")
                output.write(chunk)
        validate_video_file(temp_path, limits)
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)


def validate_video_file(path: str | Path, limits: UploadLimits) -> dict:
    path = Path(path)
    if not path.exists():
        raise InvalidVideoError("Video file does not exist.")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise InvalidVideoError("Video decode failed.")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            ok, _frame = capture.read()
            if not ok:
                raise InvalidVideoError("Video contains no decodable frames.")
            frame_count = 1
        duration = frame_count / fps if fps > 0 else 0.0
        if duration > limits.max_duration_seconds:
            raise InvalidVideoError(
                "Video exceeds configured duration limit.",
                details={"duration_seconds": duration, "limit_seconds": limits.max_duration_seconds},
            )
        return {"frame_count": frame_count, "fps": fps, "duration_seconds": duration}
    finally:
        capture.release()


def copy_path_to_temp(path: str | Path, limits: UploadLimits) -> Path:
    source = Path(path)
    suffix = validate_extension(source.name, limits)
    limits.temp_dir.mkdir(parents=True, exist_ok=True)
    target = limits.temp_dir / f"{uuid4().hex}{suffix}"
    shutil.copyfile(source, target)
    return target
