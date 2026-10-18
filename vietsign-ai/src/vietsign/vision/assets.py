"""Extractor model asset management."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MEDIAPIPE_ASSETS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
}

RTMPOSE_ASSETS = {
    "performance": {
        "det": (
            "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx",
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        ),
        "pose": (
            "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx",
            "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
        ),
    },
    "lightweight": {
        "det": (
            "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx",
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        ),
        "pose": (
            "rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx",
            "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip",
        ),
    },
    "balanced": {
        "det": (
            "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx",
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        ),
        "pose": (
            "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx",
            "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
        ),
    },
}


@dataclass(frozen=True)
class AssetStatus:
    name: str
    path: Path
    exists: bool
    size_bytes: int
    sha256: str | None
    source_url: str

    @property
    def valid(self) -> bool:
        return self.exists and self.size_bytes > 0


class AssetDownloadError(RuntimeError):
    """Raised when official model asset download fails."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def mediapipe_asset_dir() -> Path:
    return repo_root() / "models" / "mediapipe"

def rtmpose_asset_dir() -> Path:
    return repo_root() / "models" / "rtmpose"


def asset_status(asset_dir: str | Path | None = None) -> dict[str, AssetStatus]:
    root = Path(asset_dir) if asset_dir is not None else mediapipe_asset_dir()
    return {
        name: _status_for(name, root / name, url)
        for name, url in MEDIAPIPE_ASSETS.items()
    }


def install_mediapipe_assets(
    *,
    asset_dir: str | Path | None = None,
    force: bool = False,
    timeout: int = 60,
) -> dict[str, AssetStatus]:
    root = Path(asset_dir) if asset_dir is not None else mediapipe_asset_dir()
    root.mkdir(parents=True, exist_ok=True)
    installed = {}
    for name, url in MEDIAPIPE_ASSETS.items():
        target = root / name
        if target.exists() and target.stat().st_size > 0 and not force:
            installed[name] = _status_for(name, target, url)
            continue
        _download_atomic(url, target, timeout=timeout)
        installed[name] = _status_for(name, target, url)
    _write_manifest(root, installed)
    return installed

def rtmpose_asset_status(
    *,
    mode: str = "balanced",
    asset_dir: str | Path | None = None,
) -> dict[str, AssetStatus]:
    root = Path(asset_dir) if asset_dir is not None else rtmpose_asset_dir()
    assets = _rtmpose_assets_for_mode(mode)
    return {name: _status_for(filename, root / filename, url) for name, (filename, url) in assets.items()}

def install_rtmpose_assets(
    *,
    mode: str = "balanced",
    asset_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> dict[str, AssetStatus]:
    root = Path(asset_dir) if asset_dir is not None else rtmpose_asset_dir()
    root.mkdir(parents=True, exist_ok=True)
    assets = _rtmpose_assets_for_mode(mode)
    installed = {}
    for name, (filename, url) in assets.items():
        target = root / filename
        if target.exists() and target.stat().st_size > 0 and not force:
            installed[name] = _status_for(filename, target, url)
            continue
        if force:
            target.unlink(missing_ok=True)
        try:
            from rtmlib.tools.base import download_checkpoint  # type: ignore

            downloaded = Path(download_checkpoint(url, dst_dir=str(root), progress=progress))
        except Exception as error:
            raise AssetDownloadError(f"Failed to download RTMPose asset {url}: {error}") from error
        if downloaded != target and downloaded.exists():
            downloaded.replace(target)
        installed[name] = _status_for(filename, target, url)
    _write_manifest(root, installed)
    return installed


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status_for(name: str, path: Path, url: str) -> AssetStatus:
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    checksum = sha256_file(path) if exists and size > 0 else None
    return AssetStatus(name=name, path=path, exists=exists, size_bytes=size, sha256=checksum, source_url=url)


def _download_atomic(url: str, target: Path, *, timeout: int) -> None:
    if not url.startswith("https://"):
        raise AssetDownloadError(f"Refusing non-HTTPS model URL: {url}")
    part = target.with_suffix(target.suffix + f".{int(time.time())}.part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "VietSignAI/asset-installer"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status >= 400:
                raise AssetDownloadError(f"HTTP {status} while downloading {url}")
            with part.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        if not part.exists() or part.stat().st_size <= 0:
            raise AssetDownloadError(f"Downloaded zero-byte asset from {url}")
        part.replace(target)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        part.unlink(missing_ok=True)
        raise AssetDownloadError(f"Failed to download {url}: {error}") from error


def _write_manifest(root: Path, statuses: dict[str, AssetStatus]) -> None:
    payload = {
        name: {
            "path": str(status.path),
            "size_bytes": status.size_bytes,
            "sha256": status.sha256,
            "source_url": status.source_url,
        }
        for name, status in statuses.items()
    }
    (root / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def _rtmpose_assets_for_mode(mode: str) -> dict[str, tuple[str, str]]:
    try:
        return RTMPOSE_ASSETS[mode]
    except KeyError as error:
        choices = ", ".join(sorted(RTMPOSE_ASSETS))
        raise AssetDownloadError(f"Unsupported RTMPose mode {mode!r}; choose one of: {choices}") from error
