"""Gloss vocabulary for continuous sign recognition."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GlossVocabulary:
    id_to_token: tuple[str, ...]
    blank_token: str = "<blank>"
    unk_token: str | None = None

    @classmethod
    def build(cls, gloss_sequences: list[list[str]], *, blank_token: str = "<blank>") -> GlossVocabulary:
        tokens = sorted({token for sequence in gloss_sequences for token in sequence})
        return cls(id_to_token=(blank_token, *tokens), blank_token=blank_token)

    @property
    def blank_index(self) -> int:
        return self.id_to_token.index(self.blank_token)

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.id_to_token)}

    def encode(self, glosses: list[str]) -> list[int]:
        mapping = self.token_to_id
        ids = []
        for gloss in glosses:
            if gloss in mapping:
                ids.append(mapping[gloss])
            elif self.unk_token is not None and self.unk_token in mapping:
                ids.append(mapping[self.unk_token])
            else:
                raise ValueError(f"Unknown gloss token: {gloss}")
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[index] for index in ids]

    def to_dict(self) -> dict:
        return {
            "id_to_token": list(self.id_to_token),
            "blank_token": self.blank_token,
            "blank_index": self.blank_index,
            "unk_token": self.unk_token,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> GlossVocabulary:
        return cls(
            id_to_token=tuple(payload["id_to_token"]),
            blank_token=payload.get("blank_token", "<blank>"),
            unk_token=payload.get("unk_token"),
        )
