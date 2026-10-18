"""Hugging Face VideoMAE inference for ViSignLanguage isolated signs."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Lock
from typing import Any

import cv2  # type: ignore
import numpy as np
import torch

from vietsign.inference.errors import InvalidVideoError, ModelUnavailableError

LABEL_FIXES = {
    "Ban ngÖy": "Ban ngày",
    "BÖn tay": "Bàn tay",
    "Ch£ng ta": "Chúng ta",
    "ChÉn": "Chân",
    "ChÖo": "Chào",
    "C†": "Cá",
    "C†ch ly": "Cách ly",
    "GhÇt": "Ghét",
    "Gi£p": "Giúp",
    "Hìm nay": "Hôm nay",
    "Khai b†o": "Khai báo",
    "Khu c†ch ly": "Khu cách ly",
    "Kh¢c": "Khóc",
    "Ng¢n tay": "Ngón tay",
    "NhÖ": "Nhà",
    "Nhçn": "Nhìn",
    "N¢i": "Nói",
    "Nìn ¢i": "Nôn ói",
    "Ph°a sau": "Phía sau",
    "Th°ch": "Thích",
    "Tìi": "Tôi",
    "Xe m†y": "Xe máy",
    "Xin phÇp": "Xin phép",
}


class ViSignVideoMAEPredictor:
    """Small lazy wrapper around the pretrained ViSignLanguage VideoMAE model."""

    def __init__(self, model_path: str | Path, *, device: str = "cpu") -> None:
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self._processor: Any | None = None
        self._model: Any | None = None
        self._lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def predict(self, video_path: str | Path, *, top_k: int = 5) -> dict[str, Any]:
        processor, model = self._load()
        started = time.perf_counter()
        frames = sample_video_frames(video_path, num_frames=int(getattr(model.config, "num_frames", 16)))
        inputs = processor(frames, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probabilities = logits.softmax(dim=-1)[0]
        top_values, top_indices = probabilities.topk(min(top_k, probabilities.shape[-1]))
        predictions = [
            {
                "label": clean_label(model.config.id2label[int(index)]),
                "raw_label": model.config.id2label[int(index)],
                "confidence": float(score),
                "class_id": int(index),
            }
            for score, index in zip(top_values.detach().cpu().tolist(), top_indices.detach().cpu().tolist())
        ]
        return {
            "label": predictions[0]["label"],
            "confidence": predictions[0]["confidence"],
            "top_k": predictions,
            "processing_ms": (time.perf_counter() - started) * 1000.0,
            "model": {
                "name": "star092304/vi-sign-language-videomae-base",
                "local_path": str(self.model_path),
                "device": self.device.type,
                "num_frames": int(getattr(model.config, "num_frames", 16)),
                "num_labels": int(model.config.num_labels),
            },
        }

    def _load(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        with self._lock:
            if self._processor is not None and self._model is not None:
                return self._processor, self._model
            if not self.model_path.exists():
                raise ModelUnavailableError(
                    "Pretrained VideoMAE model is not available locally.",
                    details={"path": str(self.model_path)},
                )
            try:
                from transformers import AutoImageProcessor, AutoModelForVideoClassification
            except ImportError as error:  # pragma: no cover
                raise ModelUnavailableError(
                    "Install the 'transformers' optional dependency to use pretrained VideoMAE inference."
                ) from error
            self._processor = AutoImageProcessor.from_pretrained(self.model_path, local_files_only=True)
            self._model = AutoModelForVideoClassification.from_pretrained(
                self.model_path,
                local_files_only=True,
            ).to(self.device)
            self._model.eval()
            return self._processor, self._model


def sample_video_frames(video_path: str | Path, *, num_frames: int) -> list[np.ndarray]:
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise InvalidVideoError("Video decode failed.", details={"path": str(path)})
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            frames = _read_all_frames(capture)
        else:
            frames = _read_indexed_frames(capture, frame_count, num_frames)
        if not frames:
            raise InvalidVideoError("Video contains no decodable frames.", details={"path": str(path)})
        while len(frames) < num_frames:
            frames.append(frames[-1])
        return frames[:num_frames]
    finally:
        capture.release()


def clean_label(label: str) -> str:
    return LABEL_FIXES.get(label, label)


def _read_indexed_frames(capture: cv2.VideoCapture, frame_count: int, num_frames: int) -> list[np.ndarray]:
    indices = np.linspace(0, max(frame_count - 1, 0), num_frames).astype(int)
    frames: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames


def _read_all_frames(capture: cv2.VideoCapture) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames
