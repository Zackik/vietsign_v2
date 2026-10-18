"""CLI handlers for visual preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vietsign.data.schema import SignSample
from vietsign.data.video import inspect_video
from vietsign.preprocessing.config import load_preprocessing_config
from vietsign.preprocessing.pipeline import preprocess_dataset, preprocess_sample
from vietsign.training.config import load_training_config
from vietsign.training.runner import extract_rgb_embeddings
from vietsign.vision.processing import detection_coverage
from vietsign.vision.visualization import visualize_keypoint_npz


def add_preprocess_subcommands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("preprocess", help="Visual preprocessing workflows")
    preprocess_subparsers = parser.add_subparsers(dest="preprocess_command")

    keypoints = preprocess_subparsers.add_parser("keypoints", help="Extract and cache keypoints")
    keypoints.add_argument("--config", required=True, help="Path to preprocessing YAML config")
    keypoints.set_defaults(handler=handle_keypoints)

    rgb = preprocess_subparsers.add_parser("rgb", help="Extract and cache RGB embeddings")
    rgb.add_argument("--config", required=True, help="Path to RGB training/preprocessing YAML config")
    rgb.set_defaults(handler=handle_rgb)

    inspect = preprocess_subparsers.add_parser("inspect-video", help="Inspect one video through pipeline")
    inspect.add_argument("video", type=Path, help="Video path")
    inspect.add_argument(
        "--config",
        default="configs/preprocessing/mediapipe.yaml",
        help="Path to preprocessing YAML config",
    )
    inspect.set_defaults(handler=handle_inspect_video)


def add_visualize_subcommands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("visualize", help="Visualization tools")
    visualize_subparsers = parser.add_subparsers(dest="visualize_command")

    keypoints = visualize_subparsers.add_parser("keypoints", help="Render keypoint cache frames")
    keypoints.add_argument("--features", type=Path, required=True, help="Path to cached NPZ file")
    keypoints.add_argument("--output-dir", type=Path, default=Path("outputs/visualizations"))
    keypoints.add_argument("--max-frames", type=int, default=8)
    keypoints.set_defaults(handler=handle_visualize_keypoints)


def handle_keypoints(args: argparse.Namespace) -> int:
    config = load_preprocessing_config(args.config)
    summary = preprocess_dataset(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if summary["failed"] else 0

def handle_rgb(args: argparse.Namespace) -> int:
    config = load_training_config(args.config)
    summary = extract_rgb_embeddings(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if summary["failed"] else 0


def handle_inspect_video(args: argparse.Namespace) -> int:
    config = load_preprocessing_config(args.config)
    sample = SignSample(
        sample_id=Path(args.video).stem,
        video_path=args.video,
        label="unknown",
        label_id=0,
        split="inspect",
        dataset="single_video",
    )
    video_metadata = inspect_video(args.video)
    result = preprocess_sample(sample, config)
    payload = {
        "video_metadata": video_metadata.model_dump(mode="json"),
        "preprocess": result.__dict__,
    }
    if result.cache_npz:
        import numpy as np

        with np.load(result.cache_npz, allow_pickle=False) as data:
            payload["tensor_shape"] = {
                "coordinates": list(data["coordinates"].shape),
                "valid_mask": list(data["valid_mask"].shape),
                "confidence": list(data["confidence"].shape),
            }
            payload["coverage"] = detection_coverage(
                type(
                    "LoadedTensors",
                    (),
                    {
                        "valid_mask": data["valid_mask"],
                    },
                )()
            )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if result.status != "failed" else 1


def handle_visualize_keypoints(args: argparse.Namespace) -> int:
    outputs = visualize_keypoint_npz(
        args.features,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
    )
    print(json.dumps({"outputs": [str(path) for path in outputs]}, indent=2, ensure_ascii=False))
    return 0
