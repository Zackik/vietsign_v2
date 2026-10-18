"""Normalized dataset sample schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SignSample(BaseModel):
    """Canonical metadata for one sign-language video sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    video_path: Path
    label: str
    label_id: int
    split: str
    dataset: str
    signer_id: str | None = None
    camera: str | None = None
    view_id: str | None = None
    domain_id: str | None = None
    duration: float | None = Field(default=None, ge=0.0)
    fps: float | None = Field(default=None, ge=0.0)
    frame_count: int | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_id", "label", "split", "dataset")
    @classmethod
    def _required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("label_id")
    @classmethod
    def _label_id_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("label_id must be non-negative")
        return value


class ContinuousSignSample(BaseModel):
    """Canonical metadata for one continuous sign-language video."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    video_path: Path
    glosses: list[str]
    gloss_ids: list[int] = Field(default_factory=list)
    split: str
    dataset: str
    signer_id: str | None = None
    sentence_id: str | None = None
    duration: float | None = Field(default=None, ge=0.0)
    fps: float | None = Field(default=None, ge=0.0)
    frame_count: int | None = Field(default=None, ge=0)
    camera: str | None = None
    view_id: str | None = None
    domain_id: str | None = None
    translation: str | None = None
    boundaries: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_id", "split", "dataset")
    @classmethod
    def _required_continuous_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("glosses")
    @classmethod
    def _glosses_not_empty(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        if not normalized:
            raise ValueError("continuous sample must contain at least one gloss")
        return normalized


class TranslationSample(BaseModel):
    """Canonical gloss-to-Vietnamese paired sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    glosses: list[str]
    source_text: str
    target_text: str
    split: str
    dataset: str
    signer_id: str | None = None
    source_video: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_id", "source_text", "target_text", "split", "dataset")
    @classmethod
    def _required_translation_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("glosses")
    @classmethod
    def _translation_glosses_not_empty(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        if not normalized:
            raise ValueError("translation sample must contain at least one gloss")
        return normalized
