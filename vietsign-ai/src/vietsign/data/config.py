"""Dataset configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DatasetSourceConfig(BaseModel):
    """Configuration for one dataset source."""

    name: str
    adapter: Literal["custom_csv", "multiview_csv", "visignlanguage_video", "vsl400", "vsl1000"] = "custom_csv"
    root: Path = Path("data")
    metadata_csv: Path | None = None


class DatasetValidationConfig(BaseModel):
    """Dataset validation options."""

    inspect_videos: bool = True


class DatasetConfig(BaseModel):
    """Top-level dataset config."""

    dataset: DatasetSourceConfig
    validation: DatasetValidationConfig = Field(default_factory=DatasetValidationConfig)


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Load a dataset YAML configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}
    return DatasetConfig.model_validate(raw_config)
