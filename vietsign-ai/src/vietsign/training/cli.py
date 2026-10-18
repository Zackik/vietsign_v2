"""CLI handlers for recognition training/evaluation/prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vietsign.training.benchmark import model_info, run_synthetic_smoke_benchmark
from vietsign.training.config import load_training_config
from vietsign.training.runner import (
    evaluate_checkpoint,
    predict_cached_features,
    predict_continuous,
    predict_multimodal,
    predict_video,
    recognize_and_translate,
    train_model,
    translate_file,
    translate_gloss,
)
from vietsign.training.voya_direct import run_voya_direct_experiment
from vietsign.training.voya_sprint04 import run_voya_sprint04_benchmark


def add_recognition_subcommands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    train = subparsers.add_parser("train", help="Train isolated sign recognition model")
    train.add_argument("--config", required=True, help="Path to training YAML config")
    train.add_argument("--resume", default=None, help="Optional checkpoint to resume")
    train.set_defaults(handler=handle_train)

    train_voya = subparsers.add_parser("train-voya-direct", help="Train real VOYA native [T,1605] baseline")
    train_voya.add_argument("--config", required=True, help="Path to VOYA direct training YAML config")
    train_voya.set_defaults(handler=handle_train_voya_direct)

    voya_scaled = subparsers.add_parser("benchmark-voya-scaled", help="Run Sprint 04 scaled VOYA benchmark")
    voya_scaled.add_argument("--config", required=True, help="Path to Sprint 04 YAML config")
    voya_scaled.set_defaults(handler=handle_benchmark_voya_scaled)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate recognition checkpoint")
    evaluate.add_argument("--config", required=True, help="Path to training YAML config")
    evaluate.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    evaluate.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    evaluate.set_defaults(handler=handle_evaluate)

    evaluate_continuous = subparsers.add_parser("evaluate-continuous", help="Evaluate CSLR CTC checkpoint")
    evaluate_continuous.add_argument("--config", required=True)
    evaluate_continuous.add_argument("--checkpoint", required=True)
    evaluate_continuous.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    evaluate_continuous.set_defaults(handler=handle_evaluate)

    predict = subparsers.add_parser("predict", help="Predict from cached keypoint NPZ")
    predict.add_argument("--config", required=True, help="Path to training YAML config")
    predict.add_argument("--features", type=Path, required=True, help="Path to cached NPZ")
    predict.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    predict.add_argument("--top-k", type=int, default=5)
    predict.set_defaults(handler=handle_predict)

    predict_video_parser = subparsers.add_parser("predict-video", help="Predict from an RGB video")
    predict_video_parser.add_argument("--config", required=True, help="Path to RGB training YAML config")
    predict_video_parser.add_argument("--video", type=Path, required=True, help="Path to video file")
    predict_video_parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    predict_video_parser.add_argument("--top-k", type=int, default=5)
    predict_video_parser.set_defaults(handler=handle_predict_video)

    predict_multimodal_parser = subparsers.add_parser(
        "predict-multimodal",
        help="Predict from RGB video plus cached keypoints",
    )
    predict_multimodal_parser.add_argument("--config", required=True)
    predict_multimodal_parser.add_argument("--video", type=Path, required=True)
    predict_multimodal_parser.add_argument("--keypoints", type=Path, required=True)
    predict_multimodal_parser.add_argument("--metadata", type=Path, default=None)
    predict_multimodal_parser.add_argument("--checkpoint", required=True)
    predict_multimodal_parser.add_argument("--top-k", type=int, default=5)
    predict_multimodal_parser.set_defaults(handler=handle_predict_multimodal)

    predict_continuous_parser = subparsers.add_parser("predict-continuous", help="Predict glosses from continuous keypoints")
    predict_continuous_parser.add_argument("--features", type=Path, required=True)
    predict_continuous_parser.add_argument("--checkpoint", required=True)
    predict_continuous_parser.add_argument("--config", default=None)
    predict_continuous_parser.set_defaults(handler=handle_predict_continuous)

    translate_parser = subparsers.add_parser("translate", help="Translate gloss sequence to Vietnamese")
    translate_parser.add_argument("--gloss", default=None, help="Gloss sequence, e.g. 'TOI MUON NUOC'")
    translate_parser.add_argument("--input", type=Path, default=None, help="CSV containing predicted_glosses")
    translate_parser.add_argument("--output", type=Path, default=None, help="Output CSV for file translation")
    translate_parser.add_argument("--config", default=None, help="Path to gloss translation config")
    translate_parser.add_argument("--checkpoint", required=True, help="Path to gloss translation checkpoint")
    translate_parser.set_defaults(handler=handle_translate)

    recognize_translate_parser = subparsers.add_parser(
        "recognize-and-translate",
        help="Run Phase 09 CSLR checkpoint then Phase 10 translator on cached features",
    )
    recognize_translate_parser.add_argument("--features", type=Path, required=True)
    recognize_translate_parser.add_argument("--cslr-checkpoint", required=True)
    recognize_translate_parser.add_argument("--translation-checkpoint", required=True)
    recognize_translate_parser.add_argument("--cslr-config", default=None)
    recognize_translate_parser.add_argument("--translation-config", default=None)
    recognize_translate_parser.set_defaults(handler=handle_recognize_and_translate)

    model = subparsers.add_parser("model", help="Model introspection")
    model_subparsers = model.add_subparsers(dest="model_command")
    info = model_subparsers.add_parser("info", help="Show model architecture info")
    info.add_argument("--config", required=True, help="Path to training YAML config")
    info.add_argument("--num-classes", type=int, default=2)
    info.set_defaults(handler=handle_model_info)

    benchmark = subparsers.add_parser("benchmark", help="Benchmark and comparison utilities")
    benchmark_subparsers = benchmark.add_subparsers(dest="benchmark_command")
    models = benchmark_subparsers.add_parser("models", help="Compare model configs")
    models.add_argument("--configs", nargs="+", required=True)
    models.add_argument("--output-dir", default="outputs/benchmarks/phase06-smoke")
    models.add_argument("--synthetic-smoke", action="store_true")
    models.set_defaults(handler=handle_benchmark_models)


def handle_train(args: argparse.Namespace) -> int:
    config = load_training_config(args.config)
    summary = train_model(config, resume=args.resume)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def handle_train_voya_direct(args: argparse.Namespace) -> int:
    summary = run_voya_direct_experiment(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def handle_benchmark_voya_scaled(args: argparse.Namespace) -> int:
    summary = run_voya_sprint04_benchmark(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary.get("status") == "complete" else 1


def handle_evaluate(args: argparse.Namespace) -> int:
    config = load_training_config(args.config)
    report = evaluate_checkpoint(config, checkpoint_path=args.checkpoint, split=args.split)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def handle_predict(args: argparse.Namespace) -> int:
    config = load_training_config(args.config)
    prediction = predict_cached_features(
        config,
        features_path=args.features,
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
    )
    print(json.dumps(prediction, indent=2, ensure_ascii=False))
    return 0

def handle_predict_video(args: argparse.Namespace) -> int:
    config = load_training_config(args.config)
    prediction = predict_video(
        config,
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
    )
    print(json.dumps(prediction, indent=2, ensure_ascii=False))
    return 0

def handle_predict_multimodal(args: argparse.Namespace) -> int:
    config = load_training_config(args.config)
    prediction = predict_multimodal(
        config,
        video_path=args.video,
        keypoints_path=args.keypoints,
        metadata_path=args.metadata,
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
    )
    print(json.dumps(prediction, indent=2, ensure_ascii=False))
    return 0

def handle_predict_continuous(args: argparse.Namespace) -> int:
    config = load_training_config(args.config) if args.config else None
    prediction = predict_continuous(
        features_path=args.features,
        checkpoint_path=args.checkpoint,
        config=config,
    )
    print(json.dumps(prediction, indent=2, ensure_ascii=False))
    return 0


def handle_translate(args: argparse.Namespace) -> int:
    config = load_training_config(args.config) if args.config else None
    if args.input:
        if args.output is None:
            raise ValueError("--output is required when --input is provided.")
        report = translate_file(
            input_path=args.input,
            output_path=args.output,
            checkpoint_path=args.checkpoint,
            config=config,
        )
    else:
        if not args.gloss:
            raise ValueError("Provide --gloss or --input.")
        report = translate_gloss(gloss=args.gloss, checkpoint_path=args.checkpoint, config=config)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def handle_recognize_and_translate(args: argparse.Namespace) -> int:
    cslr_config = load_training_config(args.cslr_config) if args.cslr_config else None
    translation_config = load_training_config(args.translation_config) if args.translation_config else None
    report = recognize_and_translate(
        features_path=args.features,
        cslr_checkpoint=args.cslr_checkpoint,
        translation_checkpoint=args.translation_checkpoint,
        cslr_config=cslr_config,
        translation_config=translation_config,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def handle_model_info(args: argparse.Namespace) -> int:
    config = load_training_config(args.config)
    print(json.dumps(model_info(config, num_classes=args.num_classes), indent=2, ensure_ascii=False))
    return 0

def handle_benchmark_models(args: argparse.Namespace) -> int:
    if not args.synthetic_smoke:
        print(
            json.dumps(
                {
                    "status": "not_run",
                    "message": "Provide --synthetic-smoke, or use trained-checkpoint evaluation in a later run.",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 2
    report = run_synthetic_smoke_benchmark(args.configs, output_dir=args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0
