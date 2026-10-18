"""Dataset loading and batching for gloss-to-Vietnamese translation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from vietsign.data.schema import TranslationSample
from vietsign.translation.normalization import (
    glosses_to_text,
    normalize_gloss_sequence,
    normalize_vietnamese_text,
)
from vietsign.translation.vocabulary import TextVocabulary, tokenize_vietnamese_whitespace

VALID_SPLITS = {"train", "validation", "val", "test"}


class TranslationDatasetError(ValueError):
    """Raised for malformed translation metadata."""


def load_translation_samples(
    metadata_csv: str | Path,
    *,
    dataset: str = "translation",
    lowercase_vietnamese: bool = False,
    uppercase_glosses: bool = True,
) -> list[TranslationSample]:
    path = Path(metadata_csv)
    seen_ids: set[str] = set()
    samples: list[TranslationSample] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"sample_id", "gloss_sequence", "vietnamese_text", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise TranslationDatasetError(f"Missing translation CSV columns: {sorted(missing)}")
        for row in reader:
            sample_id = (row.get("sample_id") or "").strip()
            if sample_id in seen_ids:
                raise TranslationDatasetError(f"Duplicate translation sample_id: {sample_id}")
            seen_ids.add(sample_id)
            split = (row.get("split") or "").strip()
            if split not in VALID_SPLITS:
                raise TranslationDatasetError(f"Invalid split for {sample_id}: {split}")
            glosses = normalize_gloss_sequence(row.get("gloss_sequence") or "", uppercase=uppercase_glosses)
            target_text = normalize_vietnamese_text(row.get("vietnamese_text") or "", lowercase=lowercase_vietnamese)
            if not glosses:
                raise TranslationDatasetError(f"Empty gloss sequence for {sample_id}")
            if not target_text:
                raise TranslationDatasetError(f"Empty Vietnamese target for {sample_id}")
            source_text = glosses_to_text(glosses)
            samples.append(
                TranslationSample(
                    sample_id=sample_id,
                    glosses=glosses,
                    source_text=source_text,
                    target_text=target_text,
                    split="validation" if split == "val" else split,
                    dataset=(row.get("dataset") or dataset).strip() or dataset,
                    signer_id=(row.get("signer_id") or "").strip() or None,
                    source_video=Path(row["source_video"]) if (row.get("source_video") or "").strip() else None,
                )
            )
    return samples


class GlossTranslationDataset(Dataset):
    """Translation samples encoded for teacher-forced seq2seq training."""

    def __init__(
        self,
        samples: list[TranslationSample],
        *,
        source_vocabulary: TextVocabulary,
        target_vocabulary: TextVocabulary,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.samples = samples
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        if len(sample.glosses) > self.max_source_length:
            raise TranslationDatasetError(f"Gloss sequence too long for {sample.sample_id}")
        target_tokens = tokenize_vietnamese_whitespace(sample.target_text)
        if len(target_tokens) + 1 > self.max_target_length:
            raise TranslationDatasetError(f"Target sentence too long for {sample.sample_id}")
        return {
            "sample_id": sample.sample_id,
            "source_ids": torch.tensor(self.source_vocabulary.encode(sample.glosses), dtype=torch.long),
            "target_ids": torch.tensor(
                self.target_vocabulary.encode(target_tokens, add_bos=True, add_eos=True),
                dtype=torch.long,
            ),
            "source_gloss": sample.source_text,
            "target_text": sample.target_text,
            "glosses": sample.glosses,
        }


def build_translation_vocabularies(samples: list[TranslationSample]) -> tuple[TextVocabulary, TextVocabulary]:
    source_vocab = TextVocabulary.build([sample.glosses for sample in samples])
    target_vocab = TextVocabulary.build([tokenize_vietnamese_whitespace(sample.target_text) for sample in samples])
    return source_vocab, target_vocab


def collate_translation_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    source_lengths = torch.tensor([len(item["source_ids"]) for item in items], dtype=torch.long)
    target_lengths = torch.tensor([len(item["target_ids"]) for item in items], dtype=torch.long)
    max_source = int(source_lengths.max().item())
    max_target = int(target_lengths.max().item())
    source = torch.zeros((len(items), max_source), dtype=torch.long)
    target = torch.zeros((len(items), max_target), dtype=torch.long)
    for index, item in enumerate(items):
        source[index, : len(item["source_ids"])] = item["source_ids"]
        target[index, : len(item["target_ids"])] = item["target_ids"]
    return {
        "source_input_ids": source,
        "source_attention_mask": source.ne(0),
        "target_input_ids": target[:, :-1],
        "target_labels": target[:, 1:],
        "target_attention_mask": target[:, :-1].ne(0),
        "sample_ids": [item["sample_id"] for item in items],
        "source_glosses": [item["source_gloss"] for item in items],
        "target_texts": [item["target_text"] for item in items],
        "glosses": [item["glosses"] for item in items],
    }
