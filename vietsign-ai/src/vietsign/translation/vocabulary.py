"""Token vocabularies for gloss-to-Vietnamese translation."""

from __future__ import annotations

from dataclasses import dataclass

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"


@dataclass(frozen=True)
class TextVocabulary:
    id_to_token: tuple[str, ...]
    pad_token: str = PAD
    bos_token: str = BOS
    eos_token: str = EOS
    unk_token: str = UNK

    @classmethod
    def build(cls, token_sequences: list[list[str]]) -> TextVocabulary:
        specials = (PAD, BOS, EOS, UNK)
        tokens = sorted({token for sequence in token_sequences for token in sequence if token not in specials})
        return cls(id_to_token=(*specials, *tokens))

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.id_to_token)}

    @property
    def pad_index(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_index(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_index(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def unk_index(self) -> int:
        return self.token_to_id[self.unk_token]

    def encode(self, tokens: list[str], *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        mapping = self.token_to_id
        ids = [mapping.get(token, self.unk_index) for token in tokens]
        if add_bos:
            ids = [self.bos_index, *ids]
        if add_eos:
            ids = [*ids, self.eos_index]
        return ids

    def decode(self, ids: list[int], *, skip_special: bool = True) -> list[str]:
        specials = {self.pad_token, self.bos_token, self.eos_token, self.unk_token}
        tokens = [self.id_to_token[index] for index in ids]
        return [token for token in tokens if not skip_special or token not in specials]

    def to_dict(self) -> dict:
        return {
            "id_to_token": list(self.id_to_token),
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> TextVocabulary:
        return cls(
            id_to_token=tuple(payload["id_to_token"]),
            pad_token=payload.get("pad_token", PAD),
            bos_token=payload.get("bos_token", BOS),
            eos_token=payload.get("eos_token", EOS),
            unk_token=payload.get("unk_token", UNK),
        )


def tokenize_vietnamese_whitespace(text: str) -> list[str]:
    return [token for token in text.split(" ") if token]
