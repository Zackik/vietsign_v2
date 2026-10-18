"""PyTorch dataset wrappers for normalized metadata."""

from __future__ import annotations

from typing import Any

from vietsign.data.schema import SignSample


class SignVideoDataset:
    """A lightweight PyTorch-compatible dataset over normalized video references."""

    def __init__(self, samples: list[SignSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "sample_id": sample.sample_id,
            "video_path": str(sample.video_path),
            "label": sample.label,
            "label_id": sample.label_id,
            "split": sample.split,
            "dataset": sample.dataset,
            "signer_id": sample.signer_id,
            "metadata": sample.metadata,
        }

