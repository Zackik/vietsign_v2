"""CLI handlers for dataset workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vietsign.data.adapters import DatasetLoadError, create_adapter
from vietsign.data.config import load_dataset_config
from vietsign.data.sources import (
    fetch_huggingface_metadata,
    fetch_visignlanguage_video,
    load_source_registry,
    source_status_matrix,
)
from vietsign.data.stats import compute_dataset_statistics
from vietsign.data.validation import validate_samples
from vietsign.data.video import enrich_sample_with_video_metadata
from vietsign.data.voya import (
    DEFAULT_PROBE_CLASS,
    VOYA_RESOLVE_BASE_URL,
    audit_voya_directory,
    build_voya_probe_report,
    fetch_voya_probe_files,
    normalize_class_name,
    remote_file_size,
    voya_feature_layout_report,
    write_voya_probe_outputs,
    write_voya_structure_diagnostic,
)
from vietsign.utils.logging import get_logger

LOGGER = get_logger(__name__)


def add_dataset_subcommands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register dataset subcommands."""

    dataset_parser = subparsers.add_parser("dataset", help="Dataset inspection and validation")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command")

    for command, help_text, handler in (
        ("inspect", "Inspect normalized dataset samples", handle_inspect),
        ("validate", "Validate normalized dataset samples", handle_validate),
        ("stats", "Compute dataset statistics", handle_stats),
    ):
        command_parser = dataset_subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("--config", required=True, help="Path to dataset YAML config")
        command_parser.set_defaults(handler=handler)

    sources = dataset_subparsers.add_parser("sources", help="List external real-data source registry")
    sources.add_argument("--config", default="configs/data/sources.yaml")
    sources.set_defaults(handler=handle_sources)

    fetch = dataset_subparsers.add_parser("fetch", help="Fetch or inspect external data source metadata")
    fetch.add_argument("source", choices=["voya-vsl", "visignlanguage-video"])
    fetch.add_argument("--metadata-only", action="store_true", help="Download only small metadata files")
    fetch.add_argument("--dry-run", action="store_true", help="Print official fetch command without downloading")
    fetch.add_argument("--target", default=None)
    fetch.add_argument("--extract", action="store_true", help="Extract archive after download when supported")
    fetch.add_argument("--classes", nargs="*", default=None, help="Selected VOYA class IDs, e.g. 1 class_0001")
    fetch.add_argument("--max-class-size-mb", type=float, default=1024.0)
    fetch.add_argument("--max-total-size-gb", type=float, default=None)
    fetch.set_defaults(handler=handle_fetch)

    verify = dataset_subparsers.add_parser("verify", help="Verify external data source local availability")
    verify.add_argument(
        "source",
        choices=[
            "voya-vsl",
            "vsl400",
            "parallel-corpus-vie-vsl",
            "visl-news",
            "visignlanguage-video",
        ],
    )
    verify.add_argument("--root", default=None)
    verify.set_defaults(handler=handle_verify)

    audit_voya = dataset_subparsers.add_parser("audit-voya", help="Audit local VOYA_VSL NPZ structure")
    audit_voya.add_argument("--root", default="data/raw/voya_vsl")
    audit_voya.set_defaults(handler=handle_audit_voya)

    probe_voya = dataset_subparsers.add_parser("probe-voya", help="Probe one local VOYA class NPZ")
    probe_voya.add_argument("--root", default="data/raw/voya_vsl_probe")
    probe_voya.add_argument("--class-name", "--class", dest="class_name", default=DEFAULT_PROBE_CLASS)
    probe_voya.add_argument("--sequence-index", type=int, default=0)
    probe_voya.add_argument("--output", default="outputs/validation/voya_probe.json")
    probe_voya.add_argument("--markdown", default="docs/VOYA_PROBE_REPORT.md")
    probe_voya.add_argument("--visualize", action="store_true")
    probe_voya.add_argument("--visualization-output", default="outputs/validation/voya_probe_structure.png")
    probe_voya.set_defaults(handler=handle_probe_voya)

    matrix = dataset_subparsers.add_parser("real-data-matrix", help="Write real-data usability matrix")
    matrix.add_argument("--output", default="outputs/validation/real_data_matrix.json")
    matrix.set_defaults(handler=handle_real_data_matrix)


def handle_inspect(args: argparse.Namespace) -> int:
    samples = _load_samples(args.config)
    preview = [sample.model_dump(mode="json") for sample in samples[:5]]
    print(json.dumps({"total_samples": len(samples), "preview": preview}, indent=2, ensure_ascii=False))
    return 0


def handle_validate(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config)
    samples = create_adapter(config).load_samples()
    report = validate_samples(samples, inspect_videos=config.validation.inspect_videos)
    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
    if report.is_valid:
        LOGGER.info("Dataset validation passed.")
        return 0
    LOGGER.error("Dataset validation failed with %s issue(s).", len(report.issues))
    return 1


def handle_stats(args: argparse.Namespace) -> int:
    config = load_dataset_config(args.config)
    samples = create_adapter(config).load_samples()
    if config.validation.inspect_videos:
        samples = [enrich_sample_with_video_metadata(sample) for sample in samples]
    stats = compute_dataset_statistics(samples)
    print(json.dumps(stats.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


def handle_sources(args: argparse.Namespace) -> int:
    registry = load_source_registry(args.config)
    print(json.dumps(registry.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


def handle_fetch(args: argparse.Namespace) -> int:
    if args.source == "voya-vsl":
        registry = load_source_registry()
        source = registry.get("VOYA_VSL")
        target = Path(args.target) if args.target else source.local_target_directory
        metadata = fetch_huggingface_metadata("Kateht/VOYA_VSL")
        official_command = (
            "huggingface-cli download Kateht/VOYA_VSL --repo-type dataset "
            f"--local-dir {target} --local-dir-use-symlinks False"
        )
        payload = {
            "source": source.model_dump(mode="json"),
            "remote": {
                "sha": metadata.get("sha"),
                "license": metadata.get("cardData", {}).get("license"),
                "used_storage_bytes": metadata.get("usedStorage"),
                "siblings": len(metadata.get("siblings", [])),
            },
            "official_download_command": official_command,
        }
        classes = [normalize_class_name(value) for value in args.classes] if args.classes else [DEFAULT_PROBE_CLASS]
        payload["selected_classes"] = classes
        payload["selected_class_sizes"] = [
            {
                "class": class_name,
                "url": f"{VOYA_RESOLVE_BASE_URL}/Merged/{class_name}.npz",
                "size_bytes": remote_file_size(f"{VOYA_RESOLVE_BASE_URL}/Merged/{class_name}.npz"),
            }
            for class_name in classes
        ]
        if args.dry_run:
            payload["status"] = "dry_run" if args.dry_run else "metadata_only_required_for_safe_default"
            payload["note"] = "No files downloaded. Use --classes to fetch only selected class NPZ files."
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0
        fetch_report = fetch_voya_probe_files(
            target=target,
            classes=classes,
            metadata_only=args.metadata_only,
            max_class_size_mb=args.max_class_size_mb,
            max_total_size_gb=args.max_total_size_gb,
        )
        payload["status"] = "metadata_downloaded" if args.metadata_only else "probe_subset_downloaded"
        payload["target"] = str(target)
        payload["fetch_report"] = fetch_report
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if args.source == "visignlanguage-video":
        registry = load_source_registry()
        source = registry.get("ViSignLanguage-Video")
        target = Path(args.target) if args.target else source.local_target_directory
        metadata = fetch_huggingface_metadata("star092304/ViSignLanguage-Video")
        official_command = (
            "huggingface-cli download star092304/ViSignLanguage-Video --repo-type dataset "
            f"--local-dir {target} --local-dir-use-symlinks False"
        )
        payload = {
            "source": source.model_dump(mode="json"),
            "remote": {
                "sha": metadata.get("sha"),
                "license": metadata.get("cardData", {}).get("license"),
                "used_storage_bytes": metadata.get("usedStorage"),
                "siblings": len(metadata.get("siblings", [])),
            },
            "official_download_command": official_command,
            "dataset_zip_size_bytes": 863_939_416,
            "dataset_zip_size_mb": 824.0,
        }
        if args.dry_run:
            payload["status"] = "dry_run"
            payload["note"] = "No files downloaded."
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0
        fetch_report = fetch_visignlanguage_video(
            target=target,
            metadata_only=args.metadata_only,
            extract=args.extract,
        )
        payload["status"] = "metadata_downloaded" if args.metadata_only else "dataset_downloaded"
        payload["fetch_report"] = fetch_report
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    raise ValueError(args.source)


def handle_verify(args: argparse.Namespace) -> int:
    registry = load_source_registry()
    source = registry.get(args.source)
    root = Path(args.root) if args.root else source.local_target_directory
    if args.source == "voya-vsl":
        report = audit_voya_directory(root)
    elif args.source == "vsl400":
        report = {
            "dataset": "VSL400",
            "root": str(root),
            "status": "WAITING_FOR_AUTHORIZED_DATA" if not root.exists() else "AUTHORIZED_DATA_PRESENT_NEEDS_INSPECTION",
            "exists": root.exists(),
            "expected_next_step": "Place authorized files locally after DUA approval; then inspect actual structure before parsing.",
        }
    elif args.source == "parallel-corpus-vie-vsl":
        report = {
            "dataset": "Parallel-Corpus-Vie-VSL",
            "root": str(root),
            "exists": root.exists(),
            "Vie10k_txt": (root / "Vie10k.txt").exists(),
            "VSL10k_txt": (root / "VSL10k.txt").exists(),
            "license_status": "unknown",
        }
    else:
        if args.source == "visignlanguage-video":
            dataset_root = root / "dataset" if (root / "dataset").exists() else root
            report = {
                "dataset": "ViSignLanguage-Video",
                "root": str(root),
                "dataset_root": str(dataset_root),
                "exists": root.exists(),
                "metadata_csv": (root / "dataset_metadata.csv").exists(),
                "archive_zip": (root / "dataset.zip").exists(),
                "train_dir": (dataset_root / "train").exists(),
                "public_test_dir": (dataset_root / "public_test").exists(),
                "private_test_dir": (dataset_root / "private_test").exists(),
                "status": "ready_for_video_training"
                if (dataset_root / "train").exists()
                else "metadata_or_archive_present_needs_extraction",
            }
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0
        report = {
            "dataset": "ViSL-News",
            "root": str(root),
            "exists": root.exists(),
            "status": "metadata_only_until_media_resolved" if not root.exists() else "local_files_present_needs_inspection",
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def handle_audit_voya(args: argparse.Namespace) -> int:
    report = audit_voya_directory(args.root)
    report["feature_layout"] = voya_feature_layout_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def handle_probe_voya(args: argparse.Namespace) -> int:
    report = build_voya_probe_report(
        root=args.root,
        class_name=args.class_name,
        sequence_index=args.sequence_index,
    )
    outputs: dict[str, str] = {}
    if args.visualize:
        npz_path = Path(args.root) / f"{normalize_class_name(args.class_name)}.npz"
        outputs["visualization"] = write_voya_structure_diagnostic(
            npz_path,
            output_path=args.visualization_output,
            sequence_index=args.sequence_index,
        )
    outputs.update(write_voya_probe_outputs(report, json_path=args.output, markdown_path=args.markdown))
    report["outputs"] = outputs
    write_voya_probe_outputs(report, json_path=args.output, markdown_path=args.markdown)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def handle_real_data_matrix(args: argparse.Namespace) -> int:
    rows = source_status_matrix()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": rows}, indent=2, ensure_ascii=False))
    return 0


def _load_samples(config_path: str) -> list:
    try:
        config = load_dataset_config(config_path)
        return create_adapter(config).load_samples()
    except DatasetLoadError:
        raise
    except OSError as error:
        raise DatasetLoadError(str(error)) from error
