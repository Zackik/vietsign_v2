"""Adapters for real Vietnamese Sign Language translation metadata sources."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from vietsign.translation.normalization import normalize_gloss_sequence
from vietsign.translation.parallel_corpus import load_parallel_vie_vsl_samples


def write_parallel_corpus_csv(
    root: str | Path = "data/raw/parallel_corpus_vie_vsl",
    output_csv: str | Path = "data/metadata/parallel_vie_vsl_translation.csv",
) -> dict[str, Any]:
    samples = load_parallel_vie_vsl_samples(root)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "gloss_sequence", "vietnamese_text", "split", "dataset"],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "gloss_sequence": " ".join(sample.glosses),
                    "vietnamese_text": sample.target_text,
                    "split": _deterministic_split(sample.sample_id),
                    "dataset": "Parallel-Corpus-Vie-VSL",
                }
            )
    return {
        "dataset": "Parallel-Corpus-Vie-VSL",
        "samples": len(samples),
        "output_csv": str(output_csv),
        "usable_for": "gloss_to_vietnamese_translation",
    }


def write_vsl_gh_translation_csv(
    dataset_json: str | Path = "data/raw/vsl_gh/dataset.json",
    output_csv: str | Path = "data/metadata/vsl_gh_translation.csv",
) -> dict[str, Any]:
    dataset_json = Path(dataset_json)
    data = json.loads(dataset_json.read_text(encoding="utf-8"))
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "gloss_sequence",
                "vietnamese_text",
                "split",
                "dataset",
                "signer_id",
                "source_video",
            ],
        )
        writer.writeheader()
        written = 0
        for row in data:
            glosses = row.get("gloss_sequence") or []
            translation = (row.get("translation") or "").strip()
            normalized_glosses = normalize_gloss_sequence(" ".join(str(gloss) for gloss in glosses), uppercase=True)
            if not normalized_glosses or not translation:
                continue
            split = row.get("split") or "train"
            writer.writerow(
                {
                    "sample_id": row.get("id") or f"vsl_gh_{written:06d}",
                    "gloss_sequence": " ".join(normalized_glosses),
                    "vietnamese_text": translation,
                    "split": "validation" if split == "val" else split,
                    "dataset": "VSL-GH",
                    "signer_id": row.get("signer_id") or "",
                    "source_video": row.get("video_file") or "",
                }
            )
            written += 1
    return {
        "dataset": "VSL-GH",
        "samples": written,
        "output_csv": str(output_csv),
        "usable_for": "gloss_to_vietnamese_translation",
        "media_status": "metadata_only_video_files_not_downloaded",
    }


def summarize_visl_news_metadata(root: str | Path = "data/raw/visl_news") -> dict[str, Any]:
    root = Path(root)
    summary: dict[str, Any] = {
        "dataset": "ViSL-News",
        "usable_for": "sentence_level_video_to_text_when_source_videos_are_available",
        "media_status": "metadata_only_original_videos_not_redistributed",
        "splits": {},
    }
    for split, filename in (("train", "train_info.csv"), ("validation", "val_info.csv"), ("test", "test_info.csv")):
        path = root / filename
        if not path.exists():
            summary["splits"][split] = {"exists": False}
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        summary["splits"][split] = {
            "exists": True,
            "rows": len(rows),
            "columns": reader.fieldnames or [],
            "first_sample": rows[0] if rows else None,
        }
    return summary


def prepare_real_translation_sources(
    *,
    parallel_root: str | Path = "data/raw/parallel_corpus_vie_vsl",
    vsl_gh_root: str | Path = "data/raw/vsl_gh",
    visl_news_root: str | Path = "data/raw/visl_news",
    output_json: str | Path = "outputs/validation/vsl_translation_sources.json",
) -> dict[str, Any]:
    reports = []
    parallel_root = Path(parallel_root)
    if (parallel_root / "Vie10k.txt").exists() and (parallel_root / "VSL10k.txt").exists():
        reports.append(write_parallel_corpus_csv(parallel_root))
    vsl_gh_dataset = Path(vsl_gh_root) / "dataset.json"
    if vsl_gh_dataset.exists():
        reports.append(write_vsl_gh_translation_csv(vsl_gh_dataset))
    reports.append(summarize_visl_news_metadata(visl_news_root))
    payload = {"sources": reports}
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _deterministic_split(sample_id: str) -> str:
    index = int(sample_id.rsplit("_", 1)[-1])
    bucket = index % 10
    if bucket == 8:
        return "validation"
    if bucket == 9:
        return "test"
    return "train"
