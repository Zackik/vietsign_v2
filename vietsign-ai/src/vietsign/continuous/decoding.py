"""CTC decoding utilities."""

from __future__ import annotations

import torch

from vietsign.continuous.vocabulary import GlossVocabulary


class CTCGreedyDecoder:
    """Greedy CTC decoder: argmax, collapse repeats, remove blank."""

    def __init__(self, vocabulary: GlossVocabulary) -> None:
        self.vocabulary = vocabulary

    def decode_ids(self, logits_or_log_probs: torch.Tensor, *, input_lengths: torch.Tensor | None = None) -> list[list[int]]:
        if logits_or_log_probs.ndim != 3:
            raise ValueError("CTC decoder expects [B,T,V].")
        predictions = logits_or_log_probs.argmax(dim=-1).detach().cpu()
        lengths = input_lengths.detach().cpu().tolist() if input_lengths is not None else [predictions.shape[1]] * predictions.shape[0]
        decoded = []
        for row, length in zip(predictions, lengths):
            collapsed = []
            previous = None
            for token_id in row[: int(length)].tolist():
                if token_id != previous and token_id != self.vocabulary.blank_index:
                    collapsed.append(int(token_id))
                previous = token_id
            decoded.append(collapsed)
        return decoded

    def decode(self, logits_or_log_probs: torch.Tensor, *, input_lengths: torch.Tensor | None = None) -> list[list[str]]:
        return [self.vocabulary.decode(ids) for ids in self.decode_ids(logits_or_log_probs, input_lengths=input_lengths)]


class CTCBeamDecoder:
    """Interface placeholder for future beam search / language-model decoding."""

    def decode(self, *_args, **_kwargs):
        raise NotImplementedError("CTC beam search is reserved for a future phase.")
