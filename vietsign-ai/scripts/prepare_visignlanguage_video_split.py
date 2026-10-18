"""Create a train/validation manifest for ViSignLanguage-Video."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from vietsign.data.adapters import create_adapter
from vietsign.data.config import load_dataset_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/visignlanguage_video.yaml")
    parser.add_argument("--output", default="data/metadata/visignlanguage_video_split.csv")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap per class for smoke/debug manifests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = create_adapter(load_dataset_config(args.config)).load_samples()
    rows = split_samples(
        samples,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        max_per_class=args.max_per_class,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(output, rows)
    labels = {row["label"] for row in rows}
    print(
        {
            "output": str(output),
            "samples": len(rows),
            "labels": len(labels),
            "train": sum(row["split"] == "train" for row in rows),
            "validation": sum(row["split"] == "validation" for row in rows),
        }
    )
    return 0


def split_samples(samples, *, validation_ratio: float, seed: int, max_per_class: int | None):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample)

    rows = []
    for label in sorted(by_label):
        class_samples = sorted(by_label[label], key=lambda item: item.sample_id)
        rng.shuffle(class_samples)
        if max_per_class is not None:
            class_samples = class_samples[:max_per_class]
        validation_count = max(1, round(len(class_samples) * validation_ratio))
        if len(class_samples) > 1:
            validation_count = min(validation_count, len(class_samples) - 1)
        else:
            validation_count = 0
        validation_ids = {sample.sample_id for sample in class_samples[:validation_count]}
        for sample in class_samples:
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "video_path": str(sample.video_path),
                    "label": sample.label,
                    "label_id": sample.label_id,
                    "signer_id": sample.signer_id or "",
                    "split": "validation" if sample.sample_id in validation_ids else "train",
                    "dataset": sample.dataset,
                    "camera": sample.camera or "",
                    "duration": sample.duration or "",
                    "fps": sample.fps or "",
                    "frame_count": sample.frame_count or "",
                    "width": sample.width or "",
                    "height": sample.height or "",
                }
            )
    return sorted(rows, key=lambda row: (row["split"], row["label"], row["sample_id"]))


def write_manifest(output: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sample_id",
        "video_path",
        "label",
        "label_id",
        "signer_id",
        "split",
        "dataset",
        "camera",
        "duration",
        "fps",
        "frame_count",
        "width",
        "height",
    ]
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
