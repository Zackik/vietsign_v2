"""Validation CLI commands."""

from __future__ import annotations

import argparse
import json

from vietsign.validation.reports import build_phase12_manifest, write_validation_docs


def add_validation_subcommands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("validation", help="Validation and audit utilities")
    validation_subparsers = parser.add_subparsers(dest="validation_command")
    phase12 = validation_subparsers.add_parser("phase12", help="Generate Phase 12 validation manifest and docs")
    phase12.add_argument("--acceptance-dir", default="data/acceptance")
    phase12.add_argument("--preprocessing-config", default="configs/preprocessing/rtmpose.yaml")
    phase12.add_argument("--api-config", default="configs/api/default.yaml")
    phase12.add_argument("--output", default="outputs/validation/phase12_manifest.json")
    phase12.set_defaults(handler=handle_phase12)


def handle_phase12(args: argparse.Namespace) -> int:
    manifest = build_phase12_manifest(
        acceptance_dir=args.acceptance_dir,
        preprocessing_config=args.preprocessing_config,
        api_config_path=args.api_config,
        output_path=args.output,
    )
    write_validation_docs(manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0
