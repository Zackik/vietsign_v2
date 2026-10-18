"""Real-data discovery utilities."""

from __future__ import annotations

from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def discover_videos(root: str | Path, *, max_depth: int = 3) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    videos = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            try:
                path.relative_to(root)
            except ValueError:
                continue
            if len(path.relative_to(root).parts) <= max_depth:
                videos.append(path)
    return sorted(videos)


def discover_real_datasets(data_root: str | Path = "data") -> dict:
    data_root = Path(data_root)
    candidates = {
        "vsl400": data_root / "raw" / "vsl400",
        "vsl1000": data_root / "raw" / "vsl1000",
        "custom": data_root / "raw" / "custom",
    }
    result = {}
    for name, path in candidates.items():
        videos = discover_videos(path, max_depth=6)
        metadata = sorted((data_root / "metadata").glob(f"{name}*.csv")) if (data_root / "metadata").exists() else []
        result[name] = {
            "root": str(path),
            "exists": path.exists(),
            "video_count": len(videos),
            "metadata_files": [str(item) for item in metadata],
            "available": path.exists() and len(videos) > 0,
        }
    return result
