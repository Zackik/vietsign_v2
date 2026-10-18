"""External real-data source registry."""

from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataSource(BaseModel):
    name: str
    task: str
    source_url: str
    access_type: str
    license: str = "unknown"
    expected_format: str
    local_target_directory: Path
    status: str


class DataSourceRegistry(BaseModel):
    sources: list[DataSource] = Field(default_factory=list)

    def get(self, name: str) -> DataSource:
        normalized = _normalize_name(name)
        for source in self.sources:
            if _normalize_name(source.name) == normalized:
                return source
        raise KeyError(f"Unknown data source: {name}")


def load_source_registry(path: str | Path = "configs/data/sources.yaml") -> DataSourceRegistry:
    with Path(path).open("r", encoding="utf-8") as file:
        return DataSourceRegistry.model_validate(yaml.safe_load(file) or {})


def source_status_matrix(registry: DataSourceRegistry | None = None) -> list[dict[str, Any]]:
    registry = registry or load_source_registry()
    rows = []
    for source in registry.sources:
        local = Path(source.local_target_directory)
        rows.append(
            {
                "dataset": source.name,
                "task": source.task,
                "real": True,
                "source_url": source.source_url,
                "license_access": f"{source.license} / {source.access_type}",
                "local_target_directory": str(local),
                "local_status": "present" if local.exists() and any(local.iterdir()) else "missing",
                "registry_status": source.status,
                **_capability_flags(source.name),
            }
        )
    return rows


def fetch_huggingface_metadata(repo_id: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo_id}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_visignlanguage_video(
    target: str | Path = "data/raw/visignlanguage_video",
    *,
    metadata_only: bool = True,
    extract: bool = False,
) -> dict[str, Any]:
    """Fetch ViSignLanguage-Video metadata or full public dataset archive."""

    target_path = Path(target)
    target_path.mkdir(parents=True, exist_ok=True)
    repo_base = "https://huggingface.co/datasets/star092304/ViSignLanguage-Video/resolve/main"
    metadata_path = target_path / "dataset_metadata.csv"
    _download_file(f"{repo_base}/dataset_metadata.csv", metadata_path)

    report: dict[str, Any] = {
        "dataset": "ViSignLanguage-Video",
        "target": str(target_path),
        "metadata_csv": str(metadata_path),
        "metadata_only": metadata_only,
    }
    if metadata_only:
        return report

    archive_path = target_path / "dataset.zip"
    _download_file(f"{repo_base}/dataset.zip", archive_path)
    report["archive_path"] = str(archive_path)
    report["archive_size_bytes"] = archive_path.stat().st_size
    if extract:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target_path)
        report["extracted"] = True
    else:
        report["extracted"] = False
        report["next_step"] = f"Unzip {archive_path} under {target_path} before training."
    return report


def _download_file(url: str, output: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response, output.open("wb") as file:
        shutil.copyfileobj(response, file)


def _capability_flags(name: str) -> dict[str, bool | str]:
    key = _normalize_name(name)
    if key == "voya-vsl":
        return {
            "video": False,
            "keypoints": True,
            "isolated_labels": True,
            "gloss_sequence": False,
            "Vietnamese_text": False,
            "signer_ids": False,
            "usable_for_ISLR": True,
            "usable_for_CSLR": False,
            "usable_for_SLT": False,
        }
    if key == "vsl400":
        return {
            "video": True,
            "keypoints": False,
            "isolated_labels": True,
            "gloss_sequence": False,
            "Vietnamese_text": False,
            "signer_ids": True,
            "usable_for_ISLR": "pending_authorized_access",
            "usable_for_CSLR": False,
            "usable_for_SLT": False,
        }
    if key == "parallel-corpus-vie-vsl":
        return {
            "video": False,
            "keypoints": False,
            "isolated_labels": False,
            "gloss_sequence": True,
            "Vietnamese_text": True,
            "signer_ids": False,
            "usable_for_ISLR": False,
            "usable_for_CSLR": False,
            "usable_for_SLT": True,
        }
    if key == "visl-news":
        return {
            "video": "metadata_only_until_media_resolved",
            "keypoints": False,
            "isolated_labels": False,
            "gloss_sequence": False,
            "Vietnamese_text": True,
            "signer_ids": True,
            "usable_for_ISLR": False,
            "usable_for_CSLR": False,
            "usable_for_SLT": "gloss_free_sentence_level_candidate",
        }
    if key == "visignlanguage-video":
        return {
            "video": True,
            "keypoints": False,
            "isolated_labels": True,
            "gloss_sequence": False,
            "Vietnamese_text": "class_label_text",
            "signer_ids": False,
            "usable_for_ISLR": True,
            "usable_for_CSLR": False,
            "usable_for_SLT": False,
        }
    return {}


def _normalize_name(name: str) -> str:
    return name.lower().replace("_", "-")
