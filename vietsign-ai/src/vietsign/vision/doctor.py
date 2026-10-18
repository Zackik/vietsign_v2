"""Extractor readiness diagnostics."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from vietsign.preprocessing.config import PreprocessingConfig
from vietsign.utils.device import select_device
from vietsign.vision.assets import resolve_repo_path, rtmpose_asset_status, sha256_file
from vietsign.vision.rtmpose import rtmpose_input_sizes


def extractor_doctor(config: PreprocessingConfig | None = None) -> dict[str, Any]:
    """Report extractor backend readiness without pretending missing assets work."""

    mediapipe_info = _mediapipe_info(config)
    rtmpose_info = _rtmpose_info(config)
    if config is not None and config.extractor.type == "rtmpose":
        ready = bool(rtmpose_info["ready"])
    elif config is not None and config.extractor.type == "mediapipe_tasks":
        ready = bool(mediapipe_info["ready"])
    else:
        ready = bool(mediapipe_info["ready"] or rtmpose_info["ready"])
    return {
        "device": str(select_device()),
        "mediapipe": mediapipe_info,
        "rtmpose": rtmpose_info,
        "ready_for_preprocessing": ready,
    }


def _mediapipe_info(config: PreprocessingConfig | None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "installed": False,
        "version": None,
        "tasks_api": False,
        "legacy_solutions": False,
        "hand_model": "not_configured",
        "pose_model": "not_configured",
        "ready": False,
    }
    try:
        os.environ.setdefault("MPLCONFIGDIR", _mpl_config_dir())
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks.python.vision import (  # type: ignore  # noqa: F401
            HandLandmarker,
            PoseLandmarker,
        )

        info["installed"] = True
        info["version"] = getattr(mp, "__version__", "unknown")
        info["tasks_api"] = True
        info["legacy_solutions"] = hasattr(mp, "solutions")
    except Exception as error:  # noqa: BLE001
        info["error"] = str(error)
        return info

    if config is not None and config.extractor.type == "mediapipe_tasks":
        hand_report = _model_asset_report(config.extractor.hand_model)
        pose_report = _model_asset_report(config.extractor.pose_model)
        info["hand_model"] = hand_report
        info["pose_model"] = pose_report
        if config.extractor.hands and hand_report["exists"]:
            hand_report["initialization"] = _try_init_hand_landmarker(hand_report["path"])
        if config.extractor.pose and pose_report["exists"]:
            pose_report["initialization"] = _try_init_pose_landmarker(pose_report["path"])
        info["ready"] = bool(
            info["tasks_api"]
            and (not config.extractor.hands or hand_report.get("initialization", {}).get("ok") is True)
            and (not config.extractor.pose or pose_report.get("initialization", {}).get("ok") is True)
        )
    return info


def _rtmpose_info(config: PreprocessingConfig | None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "installed": False,
        "version": None,
        "onnxruntime_installed": False,
        "onnxruntime_providers": [],
        "assets": {},
        "initialization": None,
        "ready": False,
        "status": "not_installed",
    }
    try:
        import rtmlib  # type: ignore

        info["installed"] = True
        info["version"] = getattr(rtmlib, "__version__", "unknown")
        info["status"] = "installed"
    except ImportError:
        return info

    try:
        import onnxruntime as ort  # type: ignore

        info["onnxruntime_installed"] = True
        info["onnxruntime_providers"] = list(ort.get_available_providers())
    except ImportError as error:
        info["status"] = f"onnxruntime_missing: {error}"
        return info

    if config is not None and config.extractor.type == "rtmpose":
        det_report = _model_asset_report(config.rtmpose.det_model)
        pose_report = _model_asset_report(config.rtmpose.pose_model)
        info["assets"] = {
            "det_model": det_report,
            "pose_model": pose_report,
            "mode_defaults": {
                name: {
                    "path": str(status.path),
                    "exists": status.exists,
                    "size_bytes": status.size_bytes,
                    "sha256": status.sha256,
                    "source_url": status.source_url,
                }
                for name, status in rtmpose_asset_status(mode=config.rtmpose.mode).items()
            },
        }
        if det_report["exists"] and pose_report["exists"]:
            info["initialization"] = _try_init_rtmpose(
                det_path=det_report["path"],
                pose_path=pose_report["path"],
                mode=config.rtmpose.mode,
                backend=config.rtmpose.backend,
                device=config.rtmpose.device,
            )
            info["ready"] = bool(info["initialization"].get("ok"))
            info["status"] = "ready" if info["ready"] else "initialization_failed"
        else:
            info["status"] = "missing_model_assets"
    return info


def _model_asset_report(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"configured": False, "path": None, "exists": False, "size_bytes": 0, "sha256": None}
    resolved = resolve_repo_path(path)
    exists = resolved.exists() and resolved.stat().st_size > 0
    return {
        "configured": True,
        "path": str(resolved),
        "exists": exists,
        "size_bytes": resolved.stat().st_size if resolved.exists() else 0,
        "sha256": sha256_file(resolved) if exists else None,
    }


def _try_init_hand_landmarker(path: str | None) -> dict[str, Any]:
    code = f"""
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
landmarker = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path={str(path)!r}, delegate=BaseOptions.Delegate.CPU),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    )
)
landmarker.close()
"""
    return _run_init_subprocess(code)


def _try_init_pose_landmarker(path: str | None) -> dict[str, Any]:
    code = f"""
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
landmarker = PoseLandmarker.create_from_options(
    PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path={str(path)!r}, delegate=BaseOptions.Delegate.CPU),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )
)
landmarker.close()
"""
    return _run_init_subprocess(code)


def _try_init_rtmpose(*, det_path: str | None, pose_path: str | None, mode: str, backend: str, device: str) -> dict[str, Any]:
    input_sizes = rtmpose_input_sizes(mode)
    code = f"""
from rtmlib import Wholebody
model = Wholebody(
    det={str(det_path)!r},
    det_input_size={input_sizes["det"]!r},
    pose={str(pose_path)!r},
    pose_input_size={input_sizes["pose"]!r},
    mode={mode!r},
    to_openpose=True,
    backend={backend!r},
    device={device!r},
)
assert model is not None
"""
    return _run_init_subprocess(code)


def _run_init_subprocess(code: str) -> dict[str, Any]:
    env = {**os.environ, "MPLCONFIGDIR": _mpl_config_dir()}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "error": (result.stderr or result.stdout).strip() or None,
    }


def _mpl_config_dir() -> str:
    path = os.path.join(tempfile.gettempdir(), "vietsign-mplconfig")
    os.makedirs(path, exist_ok=True)
    return path
