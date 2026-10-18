"""API and inference runtime configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class UploadLimits(BaseModel):
    max_upload_mb: int = Field(default=100, gt=0)
    max_duration_seconds: float = Field(default=30.0, gt=0.0)
    allowed_extensions: list[str] = Field(default_factory=lambda: [".mp4", ".mov", ".avi", ".mkv", ".webm"])
    temp_dir: Path = Path("outputs/tmp/uploads")


class CorsConfig(BaseModel):
    allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"])
    allow_credentials: bool = False


class RealtimeConfig(BaseModel):
    window_size: int = Field(default=32, gt=0)
    stride: int = Field(default=8, gt=0)
    minimum_frames: int = Field(default=16, gt=0)
    smoothing: Literal["majority_vote", "ema"] = "majority_vote"
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    duplicate_cooldown_windows: int = Field(default=3, ge=0)


class InferenceModelConfig(BaseModel):
    name: str
    task: Literal[
        "isolated_recognition",
        "continuous_recognition",
        "gloss_translation",
        "rgb_recognition",
        "multimodal_recognition",
    ]
    modality: str
    config_path: Path | None = None
    checkpoint_path: Path | None = None
    development_only: bool = False
    maturity: Literal["development", "experimental_real_data", "validated_real_data", "production"] = "development"
    load_on_start: bool = False
    enabled: bool = True


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, gt=0)
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    default_pipeline: str = "continuous_translate"
    upload: UploadLimits = Field(default_factory=UploadLimits)
    cors: CorsConfig = Field(default_factory=CorsConfig)
    realtime: RealtimeConfig = Field(default_factory=RealtimeConfig)
    models: list[InferenceModelConfig] = Field(default_factory=list)


def load_api_config(path: str | Path) -> ApiConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    return ApiConfig.model_validate(payload)
