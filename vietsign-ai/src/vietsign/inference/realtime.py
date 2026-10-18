"""Backend-independent realtime session primitives."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class RealtimeEvent:
    type: Literal["prediction_started", "prediction_updated", "prediction_committed", "prediction_cleared"]
    payload: dict[str, Any]


class RealtimeSession:
    def __init__(
        self,
        *,
        window_size: int = 32,
        stride: int = 8,
        minimum_frames: int = 16,
        confidence_threshold: float = 0.5,
        duplicate_cooldown_windows: int = 3,
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.minimum_frames = minimum_frames
        self.confidence_threshold = confidence_threshold
        self.duplicate_cooldown_windows = duplicate_cooldown_windows
        self.frames: deque[np.ndarray] = deque(maxlen=window_size)
        self.timestamps: deque[float] = deque(maxlen=window_size)
        self._since_prediction = 0
        self._cooldown = 0
        self.last_committed: str | None = None
        self.history: deque[tuple[str, float]] = deque(maxlen=5)

    def add_frame(self, frame: np.ndarray, *, timestamp: float) -> list[RealtimeEvent]:
        self.frames.append(frame)
        self.timestamps.append(timestamp)
        self._since_prediction += 1
        if len(self.frames) < self.minimum_frames or self._since_prediction < self.stride:
            return []
        self._since_prediction = 0
        return [RealtimeEvent("prediction_started", {"frames": len(self.frames)})]

    def update_prediction(self, label: str, confidence: float) -> list[RealtimeEvent]:
        events = [RealtimeEvent("prediction_updated", {"label": label, "confidence": confidence})]
        if confidence < self.confidence_threshold:
            return events
        self.history.append((label, confidence))
        smoothed, score = majority_vote(self.history)
        if self._cooldown > 0:
            self._cooldown -= 1
            return events
        if smoothed != self.last_committed:
            self.last_committed = smoothed
            self._cooldown = self.duplicate_cooldown_windows
            events.append(RealtimeEvent("prediction_committed", {"label": smoothed, "confidence": score}))
        return events

    def clear(self) -> RealtimeEvent:
        self.frames.clear()
        self.timestamps.clear()
        self.history.clear()
        self.last_committed = None
        self._since_prediction = 0
        self._cooldown = 0
        return RealtimeEvent("prediction_cleared", {})


def majority_vote(history: deque[tuple[str, float]]) -> tuple[str, float]:
    if not history:
        return "", 0.0
    counts = Counter(label for label, _confidence in history)
    label = counts.most_common(1)[0][0]
    confidences = [confidence for item, confidence in history if item == label]
    return label, sum(confidences) / max(len(confidences), 1)
