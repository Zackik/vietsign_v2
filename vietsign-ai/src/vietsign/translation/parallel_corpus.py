"""Adapter for local Parallel-Corpus-Vie-VSL files."""

from __future__ import annotations

from pathlib import Path

from vietsign.data.schema import TranslationSample
from vietsign.translation.normalization import normalize_gloss_sequence, normalize_vietnamese_text


def load_parallel_vie_vsl_samples(
    root: str | Path = "data/raw/parallel_corpus_vie_vsl",
    *,
    max_samples: int | None = None,
    split: str = "train",
) -> list[TranslationSample]:
    root = Path(root)
    vie_path = root / "Vie10k.txt"
    vsl_path = root / "VSL10k.txt"
    if not vie_path.exists() or not vsl_path.exists():
        raise FileNotFoundError(
            "Parallel-Corpus-Vie-VSL local text files not found. Expected Vie10k.txt and VSL10k.txt."
        )
    vie_lines = vie_path.read_text(encoding="utf-8").splitlines()
    vsl_lines = vsl_path.read_text(encoding="utf-8").splitlines()
    if len(vie_lines) != len(vsl_lines):
        raise ValueError(f"Parallel corpus line count mismatch: Vie={len(vie_lines)} VSL={len(vsl_lines)}")
    samples = []
    for index, (vie, vsl) in enumerate(zip(vie_lines, vsl_lines, strict=True)):
        if max_samples is not None and index >= max_samples:
            break
        target = normalize_vietnamese_text(vie)
        glosses = normalize_gloss_sequence(vsl, uppercase=True)
        if not target or not glosses:
            continue
        samples.append(
            TranslationSample(
                sample_id=f"parallel_vie_vsl_{index:06d}",
                glosses=glosses,
                source_text=" ".join(glosses),
                target_text=target,
                split=split,
                dataset="Parallel-Corpus-Vie-VSL",
                metadata={
                    "source_dataset": "Parallel-Corpus-Vie-VSL",
                    "original_line_index": index,
                    "original_vietnamese": vie,
                    "original_vsl": vsl,
                    "license_status": "unknown",
                },
            )
        )
    return samples
