"""YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    """Dataset and manifest locations."""

    root_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    train_manifest: Path = Path("data/train.csv")
    val_manifest: Path = Path("data/val.csv")
    test_manifest: Path = Path("data/test.csv")


class PreprocessingConfig(BaseModel):
    """Video preprocessing parameters."""

    frame_rate: int = Field(default=25, gt=0)
    image_size: tuple[int, int] = (224, 224)
    max_frames: int = Field(default=256, gt=0)
    normalize: bool = True


class FeaturesConfig(BaseModel):
    """Feature extraction options."""

    use_pose: bool = True
    use_rgb: bool = True
    pose_backend: str = "mediapipe"
    rgb_backend: str = "none"


class ModelConfig(BaseModel):
    """Model hyperparameters."""

    name: str = "baseline_temporal"
    num_classes: int | None = None
    hidden_size: int = Field(default=256, gt=0)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    batch_size: int = Field(default=8, gt=0)
    epochs: int = Field(default=20, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    num_workers: int = Field(default=2, ge=0)
    checkpoint_dir: Path = Path("outputs/checkpoints")
    log_interval: int = Field(default=20, gt=0)


class DeviceConfig(BaseModel):
    """Runtime device preferences."""

    preference: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    allow_mps: bool = True
    allow_cuda: bool = True


class ProjectConfig(BaseModel):
    """Top-level project configuration."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)


def load_config(path: str | Path) -> ProjectConfig:
    """Load a YAML config file into a validated ProjectConfig."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}
    return ProjectConfig.model_validate(raw_config)

