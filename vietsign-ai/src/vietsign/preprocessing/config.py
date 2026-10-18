"""Preprocessing configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ExtractorConfig(BaseModel):
    type: Literal["mediapipe", "mediapipe_tasks", "rtmpose"] = "mediapipe_tasks"
    hands: bool = True
    pose: bool = True
    face: bool = False
    hand_model: Path | None = None
    pose_model: Path | None = None
    face_model: Path | None = None
    rtmpose_config: Path | None = None
    checkpoint: Path | None = None


class RTMPoseRuntimeConfig(BaseModel):
    runtime: Literal["rtmlib"] = "rtmlib"
    backend: Literal["onnxruntime", "opencv", "openvino"] = "onnxruntime"
    device: str = "cpu"
    mode: Literal["lightweight", "balanced", "performance"] = "balanced"
    keypoint_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    det_model: Path | None = None
    pose_model: Path | None = None


class VideoSamplingConfig(BaseModel):
    target_fps: float | None = 25
    num_frames: int | None = Field(default=64, gt=0)
    sampling: Literal["uniform"] = "uniform"


class NormalizationConfig(BaseModel):
    type: Literal["none", "image", "body_relative"] = "body_relative"


class InputVideoConfig(BaseModel):
    mirrored: bool = False


class InterpolationConfig(BaseModel):
    enabled: bool = True
    max_gap: int = Field(default=3, ge=0)


class SmoothingConfig(BaseModel):
    enabled: bool = False
    type: Literal["ema"] = "ema"
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)


class FeatureConfig(BaseModel):
    velocity: bool = True
    acceleration: bool = False


class CacheConfig(BaseModel):
    enabled: bool = True
    root: Path = Path("data/cache/keypoints")
    reuse: bool = True


class WarningConfig(BaseModel):
    hand_coverage_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PreprocessingConfig(BaseModel):
    dataset_config: Path
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    rtmpose: RTMPoseRuntimeConfig = Field(default_factory=RTMPoseRuntimeConfig)
    input: InputVideoConfig = Field(default_factory=InputVideoConfig)
    video: VideoSamplingConfig = Field(default_factory=VideoSamplingConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    interpolation: InterpolationConfig = Field(default_factory=InterpolationConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    warnings: WarningConfig = Field(default_factory=WarningConfig)


def load_preprocessing_config(path: str | Path) -> PreprocessingConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}
    return PreprocessingConfig.model_validate(raw_config)
