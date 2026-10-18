"""Extractor diagnostics and real-video verification CLI."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vietsign.data.schema import SignSample
from vietsign.models.temporal_transformer import TemporalTransformerClassifier
from vietsign.preprocessing.cache import load_keypoint_cache
from vietsign.preprocessing.config import load_preprocessing_config
from vietsign.preprocessing.pipeline import preprocess_sample
from vietsign.training.config import InputFeatureConfig, SequenceConfig
from vietsign.training.input import prepare_keypoint_arrays
from vietsign.vision.assets import (
    AssetDownloadError,
    install_mediapipe_assets,
    install_rtmpose_assets,
)
from vietsign.vision.doctor import extractor_doctor
from vietsign.vision.visualization import visualize_keypoint_npz


def add_extractor_subcommands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("extractor", help="Extractor diagnostics")
    extractor_subparsers = parser.add_subparsers(dest="extractor_command")

    doctor = extractor_subparsers.add_parser("doctor", help="Report extractor readiness")
    doctor.add_argument("--config", default=None, help="Optional preprocessing config")
    doctor.set_defaults(handler=handle_doctor)

    assets = extractor_subparsers.add_parser("assets", help="Manage extractor model assets")
    assets_subparsers = assets.add_subparsers(dest="assets_command")
    install = assets_subparsers.add_parser("install", help="Install extractor assets")
    install.add_argument("backend", choices=["mediapipe", "rtmpose"])
    install.add_argument("--mode", choices=["lightweight", "balanced", "performance"], default="balanced")
    install.add_argument("--force", action="store_true")
    install.set_defaults(handler=handle_assets_install)

    test = extractor_subparsers.add_parser("test", help="Run one real-video extractor test")
    test.add_argument("--video", type=Path, required=True)
    test.add_argument("--config", required=True)
    test.add_argument("--visualize", action="store_true")
    test.set_defaults(handler=handle_test)

    acceptance = extractor_subparsers.add_parser("acceptance-test", help="Run Phase 05B real extraction gate")
    acceptance.add_argument("--video", type=Path, required=True)
    acceptance.add_argument("--config", required=True)
    acceptance.set_defaults(handler=handle_acceptance_test)

    compare = extractor_subparsers.add_parser("compare", help="Compare extractor configs on one video")
    compare.add_argument("--video", type=Path, required=True)
    compare.add_argument("--configs", nargs="+", required=True)
    compare.set_defaults(handler=handle_compare)

    benchmark = extractor_subparsers.add_parser("benchmark", help="Benchmark RTMPose modes on one video")
    benchmark.add_argument("--video", type=Path, required=True)
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument(
        "--modes",
        nargs="+",
        choices=["lightweight", "balanced", "performance"],
        default=["lightweight", "balanced", "performance"],
    )
    benchmark.set_defaults(handler=handle_benchmark)


def handle_doctor(args: argparse.Namespace) -> int:
    config = load_preprocessing_config(args.config) if args.config else None
    payload = extractor_doctor(config)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ready_for_preprocessing"] else 1


def handle_assets_install(args: argparse.Namespace) -> int:
    try:
        if args.backend == "mediapipe":
            statuses = install_mediapipe_assets(force=args.force)
            default_config = Path("configs/preprocessing/mediapipe_tasks.yaml")
        else:
            statuses = install_rtmpose_assets(mode=args.mode, force=args.force)
            default_config = Path("configs/preprocessing/rtmpose.yaml")
    except AssetDownloadError as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2, ensure_ascii=False))
        return 1
    payload = {
        "backend": args.backend,
        "assets": {
            name: {
                "path": str(status.path),
                "size_bytes": status.size_bytes,
                "sha256": status.sha256,
                "source_url": status.source_url,
                "valid": status.valid,
            }
            for name, status in statuses.items()
        },
    }
    if default_config.exists():
        payload["doctor"] = extractor_doctor(load_preprocessing_config(default_config))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if all(status.valid for status in statuses.values()) and payload.get("doctor", {}).get("ready_for_preprocessing", True) else 1


def handle_test(args: argparse.Namespace) -> int:
    config = load_preprocessing_config(args.config)
    started = time.perf_counter()
    result = _preprocess_single(args.video, config)
    payload = {
        "video": str(args.video),
        "result": result.__dict__,
        "runtime_seconds": time.perf_counter() - started,
        "frames_per_second": result.frame_count / max(time.perf_counter() - started, 1e-9),
    }
    if result.cache_npz and args.visualize:
        payload["visualizations"] = [
            str(path) for path in visualize_keypoint_npz(result.cache_npz, output_dir="outputs/quality_review")
        ]
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if result.status != "processed":
        return 1
    if max(result.coverage.values(), default=0.0) <= 0.0:
        return 1
    return 0


def handle_acceptance_test(args: argparse.Namespace) -> int:
    config = load_preprocessing_config(args.config)
    doctor_payload = extractor_doctor(config)
    started = time.perf_counter()
    result = _preprocess_single(args.video, config)
    payload = {
        "doctor": doctor_payload,
        "video": str(args.video),
        "preprocess": result.__dict__,
        "runtime_seconds": time.perf_counter() - started,
        "status": "FAIL",
    }
    if result.cache_npz and result.cache_metadata:
        arrays, _metadata = load_keypoint_cache(Path(result.cache_npz), Path(result.cache_metadata))
        valid_sum = int(arrays["valid_mask"].sum())
        payload["coordinates_shape"] = list(arrays["coordinates"].shape)
        payload["valid_mask_shape"] = list(arrays["valid_mask"].shape)
        payload["valid_mask_sum"] = valid_sum
        payload["acceptance_gate"] = _real_extraction_gate(
            arrays=arrays,
            coverage=result.coverage,
            extractor_ready=doctor_payload["ready_for_preprocessing"],
        )
        payload["visualizations"] = [
            str(path)
            for path in visualize_keypoint_npz(
                result.cache_npz,
                output_dir=Path("outputs/quality_review") / Path(args.video).stem,
            )
        ]
        if payload["acceptance_gate"]["ok"]:
            model_report = _phase04_compatibility(arrays)
            payload["phase04_compatibility"] = model_report
            if model_report["ok"]:
                payload["status"] = "PASS"
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


def handle_compare(args: argparse.Namespace) -> int:
    rows = []
    for config_path in args.configs:
        config = load_preprocessing_config(config_path)
        started = time.perf_counter()
        result = _preprocess_single(args.video, config)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "config": config_path,
                "status": result.status,
                "error": result.error,
                "coverage": result.coverage,
                "runtime_seconds": elapsed,
                "milliseconds_per_frame": (elapsed * 1000 / result.frame_count) if result.frame_count else None,
                "frames_per_second": result.frame_count / elapsed if elapsed > 0 and result.frame_count else 0.0,
            }
        )
    print(json.dumps({"video": str(args.video), "results": rows}, indent=2, ensure_ascii=False))
    return 0 if any(row["status"] == "processed" for row in rows) else 1

def handle_benchmark(args: argparse.Namespace) -> int:
    rows = []
    base_config = load_preprocessing_config(args.config)
    for mode in args.modes:
        config = base_config.model_copy(deep=True)
        config.rtmpose.mode = mode
        started = time.perf_counter()
        result = _preprocess_single(args.video, config)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "mode": mode,
                "status": result.status,
                "error": result.error,
                "coverage": result.coverage,
                "runtime_seconds": elapsed,
                "milliseconds_per_frame": (elapsed * 1000 / result.frame_count) if result.frame_count else None,
                "frames_per_second": result.frame_count / elapsed if elapsed > 0 and result.frame_count else 0.0,
            }
        )
    print(json.dumps({"video": str(args.video), "results": rows}, indent=2, ensure_ascii=False))
    return 0 if any(row["status"] == "processed" for row in rows) else 1


def _preprocess_single(video: Path, config):
    sample = SignSample(
        sample_id=video.stem,
        video_path=video,
        label="unknown",
        label_id=0,
        split="inspect",
        dataset="single_video",
    )
    return preprocess_sample(sample, config)


def _phase04_compatibility(arrays: dict) -> dict:
    try:
        inputs, valid_mask, temporal_mask, _ratio = prepare_keypoint_arrays(
            arrays,
            feature_config=InputFeatureConfig(validity=True, velocity=False),
            sequence_config=SequenceConfig(mode="variable"),
            minimum_valid_joint_ratio=0.0,
            reject_empty=True,
        )
        batch = {
            "coordinates": torch.from_numpy(inputs).unsqueeze(0),
            "valid_mask": torch.from_numpy(valid_mask).unsqueeze(0),
            "temporal_mask": torch.from_numpy(temporal_mask).unsqueeze(0),
        }
        model = TemporalTransformerClassifier(
            joint_count=batch["coordinates"].shape[2],
            input_channels=batch["coordinates"].shape[3],
            num_classes=2,
            embedding_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        with torch.no_grad():
            logits = model(batch["coordinates"], batch["valid_mask"], batch["temporal_mask"])
        finite = bool(torch.isfinite(logits).all().item())
        return {
            "label": "MODEL COMPATIBILITY TEST ONLY",
            "ok": finite and list(logits.shape) == [1, 2],
            "model_input_shape": list(batch["coordinates"].shape),
            "logits_shape": list(logits.shape),
            "finite": finite,
        }
    except Exception as error:  # noqa: BLE001
        return {"label": "MODEL COMPATIBILITY TEST ONLY", "ok": False, "error": str(error)}

def _real_extraction_gate(*, arrays: dict, coverage: dict[str, float], extractor_ready: bool) -> dict:
    coordinates_shape = list(arrays["coordinates"].shape)
    valid_mask_shape = list(arrays["valid_mask"].shape)
    valid_sum = int(arrays["valid_mask"].sum())
    pose_coverage = float(coverage.get("pose_coverage", 0.0))
    left_hand_coverage = float(coverage.get("left_hand_coverage", 0.0))
    right_hand_coverage = float(coverage.get("right_hand_coverage", 0.0))
    ok = bool(
        extractor_ready
        and len(coordinates_shape) == 3
        and coordinates_shape[1:] == [55, 3]
        and valid_mask_shape == coordinates_shape[:2]
        and valid_sum > 0
        and pose_coverage > 0.0
        and (left_hand_coverage > 0.0 or right_hand_coverage > 0.0)
    )
    return {
        "ok": ok,
        "requires": "ready extractor, coordinates=[T,55,3], valid_mask=[T,55], valid_sum>0, pose_coverage>0, and at least one hand coverage>0",
        "pose_coverage": pose_coverage,
        "left_hand_coverage": left_hand_coverage,
        "right_hand_coverage": right_hand_coverage,
    }
