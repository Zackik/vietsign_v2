"""Gloss-to-Vietnamese Transformer baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TranslationModelIntrospection:
    model_name: str
    task_type: str
    modality: str
    source_vocab_size: int
    target_vocab_size: int
    d_model: int
    encoder_layers: int
    decoder_layers: int
    num_heads: int
    parameter_count: int
    trainable_parameter_count: int


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 512) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_terms = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        encoding = torch.zeros((max_length, d_model), dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_terms)
        encoding[:, 1::2] = torch.cos(positions * div_terms[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:, : values.shape[1]].to(values.device)


class GlossToVietnameseTransformer(nn.Module):
    """Small seq2seq Transformer for software validation and controlled baselines."""

    def __init__(
        self,
        *,
        source_vocab_size: int,
        target_vocab_size: int,
        pad_index: int,
        bos_index: int,
        eos_index: int,
        d_model: int = 128,
        num_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_length: int = 128,
    ) -> None:
        super().__init__()
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.pad_index = pad_index
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_heads = num_heads
        self.source_embedding = nn.Embedding(source_vocab_size, d_model, padding_idx=pad_index)
        self.target_embedding = nn.Embedding(target_vocab_size, d_model, padding_idx=pad_index)
        self.position = SinusoidalPositionEncoding(d_model, max_length=max_length)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Linear(d_model, target_vocab_size)

    def forward(
        self,
        source_input_ids: torch.Tensor,
        target_input_ids: torch.Tensor,
        *,
        source_attention_mask: torch.Tensor | None = None,
        target_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source_padding_mask = _padding_mask(source_input_ids, source_attention_mask, self.pad_index)
        target_padding_mask = _padding_mask(target_input_ids, target_attention_mask, self.pad_index)
        target_mask = self.causal_mask(target_input_ids.shape[1], target_input_ids.device)
        source = self.position(self.source_embedding(source_input_ids) * (self.d_model**0.5))
        target = self.position(self.target_embedding(target_input_ids) * (self.d_model**0.5))
        hidden = self.transformer(
            source,
            target,
            tgt_mask=target_mask,
            src_key_padding_mask=source_padding_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=source_padding_mask,
        )
        return self.output(hidden)

    def loss(
        self,
        source_input_ids: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_labels: torch.Tensor,
        *,
        source_attention_mask: torch.Tensor | None = None,
        target_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self(
            source_input_ids,
            target_input_ids,
            source_attention_mask=source_attention_mask,
            target_attention_mask=target_attention_mask,
        )
        return nn.CrossEntropyLoss(ignore_index=self.pad_index)(
            logits.reshape(-1, logits.shape[-1]),
            target_labels.reshape(-1),
        )

    def greedy_decode(
        self,
        source_input_ids: torch.Tensor,
        *,
        source_attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 64,
    ) -> torch.Tensor:
        batch_size = source_input_ids.shape[0]
        generated = torch.full((batch_size, 1), self.bos_index, dtype=torch.long, device=source_input_ids.device)
        finished = torch.zeros((batch_size,), dtype=torch.bool, device=source_input_ids.device)
        for _ in range(max_new_tokens):
            logits = self(source_input_ids, generated, source_attention_mask=source_attention_mask)
            next_ids = logits[:, -1].argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, self.pad_index), next_ids)
            generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)
            finished |= next_ids.eq(self.eos_index)
            if bool(finished.all()):
                break
        return generated

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)

    def introspect(self) -> TranslationModelIntrospection:
        return TranslationModelIntrospection(
            model_name="gloss_transformer",
            task_type="gloss_translation",
            modality="text",
            source_vocab_size=self.source_vocab_size,
            target_vocab_size=self.target_vocab_size,
            d_model=self.d_model,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            num_heads=self.num_heads,
            parameter_count=sum(parameter.numel() for parameter in self.parameters()),
            trainable_parameter_count=sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        )


def _padding_mask(
    ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    pad_index: int,
) -> torch.Tensor:
    if attention_mask is not None:
        return ~attention_mask.bool()
    return ids.eq(pad_index)
