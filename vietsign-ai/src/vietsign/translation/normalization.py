"""Conservative text normalization for gloss translation."""

from __future__ import annotations

import re
import unicodedata

_SPACE_RE = re.compile(r"\s+")


def normalize_vietnamese_text(text: str, *, lowercase: bool = False) -> str:
    """Normalize Vietnamese text without stripping diacritics."""

    normalized = unicodedata.normalize("NFC", text)
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    normalized = normalized.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    return normalized.lower() if lowercase else normalized


def normalize_gloss_sequence(glosses: list[str] | str, *, uppercase: bool = True) -> list[str]:
    """Normalize gloss whitespace/casing without translating gloss tokens."""

    tokens = glosses.split() if isinstance(glosses, str) else glosses
    normalized = []
    for token in tokens:
        value = _SPACE_RE.sub(" ", token.strip())
        if not value:
            continue
        normalized.append(value.upper() if uppercase else value)
    return normalized


def glosses_to_text(glosses: list[str]) -> str:
    return " ".join(glosses)
