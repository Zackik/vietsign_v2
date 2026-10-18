"""Command-line interface for VietSignAI."""

from __future__ import annotations

import argparse
from pathlib import Path

from vietsign.config import load_config
from vietsign.data.adapters import DatasetLoadError
from vietsign.data.cli import add_dataset_subcommands
from vietsign.preprocessing.cli import add_preprocess_subcommands, add_visualize_subcommands
from vietsign.training.cli import add_recognition_subcommands
from vietsign.utils.device import select_device
from vietsign.utils.logging import get_logger
from vietsign.validation.cli import add_validation_subcommands
from vietsign.vision.cli import add_extractor_subcommands

LOGGER = get_logger(__name__)


def _not_implemented(command: str) -> int:
    LOGGER.info("%s is not implemented in the foundation phase.", command)
    return 2


def _handle_config(args: argparse.Namespace) -> int:
    config = load_config(args.path)
    LOGGER.info("Loaded config from %s with seed=%s", args.path, config.seed)
    return 0


def _handle_device(_: argparse.Namespace) -> int:
    device = select_device()
    LOGGER.info("Selected device: %s", device)
    return 0


def _handle_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("Install the 'api' optional dependency to run the API server.") from error
    from vietsign.api.app import create_app
    from vietsign.inference.config import load_api_config

    config = load_api_config(args.config)
    uvicorn.run(create_app(config), host=config.host, port=config.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the root CLI parser."""

    parser = argparse.ArgumentParser(prog="vietsign", description="VietSignAI command line")
    subparsers = parser.add_subparsers(dest="command")

    config_parser = subparsers.add_parser("config", help="Load and validate a YAML config")
    config_parser.add_argument("path", type=Path, help="Path to a YAML config file")
    config_parser.set_defaults(handler=_handle_config)

    device_parser = subparsers.add_parser("device", help="Show selected runtime device")
    device_parser.set_defaults(handler=_handle_device)

    serve_parser = subparsers.add_parser("serve", help="Run VietSignAI FastAPI server")
    serve_parser.add_argument("--config", default="configs/api/default.yaml", help="Path to API YAML config")
    serve_parser.add_argument("--reload", action="store_true", help="Enable Uvicorn reload for development")
    serve_parser.set_defaults(handler=_handle_serve)

    add_dataset_subcommands(subparsers)
    add_preprocess_subcommands(subparsers)
    add_visualize_subcommands(subparsers)
    add_extractor_subcommands(subparsers)
    add_recognition_subcommands(subparsers)
    add_validation_subcommands(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 1
    try:
        return int(args.handler(args))
    except DatasetLoadError as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
