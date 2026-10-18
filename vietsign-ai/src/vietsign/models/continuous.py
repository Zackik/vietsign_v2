"""Continuous sign recognition models with CTC heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from vietsign.models.temporal_transformer import TemporalTransformerClassifier


@dataclass(frozen=True)
class ContinuousModelIntrospection:
    model_name: str
    task_type: str
    encoder_type: str
    input_joints: int
    input_channels: int
    embedding_dim: int
    vocab_size: int
    blank_index: int
    parameter_count: int
    trainable_parameter_count: int


class ContinuousSignRecognizer(nn.Module):
    """Temporal keypoint encoder plus CTC projection head."""

    def __init__(
        self,
        *,
        joint_count: int,
        input_channels: int,
        vocab_size: int,
        blank_index: int = 0,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.blank_index = blank_index
        self.encoder = TemporalTransformerClassifier(
            joint_count=joint_count,
            input_channels=input_channels,
            num_classes=vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ctc_head = nn.Linear(embedding_dim, vocab_size)

    def forward_tokens(
        self,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder.forward_tokens(coordinates, valid_mask, temporal_mask)

    def forward(
        self,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.ctc_head(self.forward_tokens(coordinates, valid_mask, temporal_mask))

    def log_probs(
        self,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> torch.Tensor:
        return torch.log_softmax(self.forward(coordinates, valid_mask, temporal_mask), dim=-1)

    def ctc_loss(
        self,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
        temporal_mask: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        *,
        zero_infinity: bool = True,
    ) -> torch.Tensor:
        input_lengths = temporal_mask.long().sum(dim=1)
        validate_ctc_lengths(input_lengths, target_lengths)
        log_probs = self.log_probs(coordinates, valid_mask, temporal_mask).transpose(0, 1)
        loss = nn.CTCLoss(blank=self.blank_index, zero_infinity=zero_infinity)
        return loss(log_probs, targets.long(), input_lengths.long(), target_lengths.long())

    def introspect(self) -> ContinuousModelIntrospection:
        return ContinuousModelIntrospection(
            model_name="ctc_recognizer",
            task_type="continuous_recognition",
            encoder_type="temporal_transformer",
            input_joints=self.encoder.joint_count,
            input_channels=self.encoder.input_channels,
            embedding_dim=self.encoder.embedding_dim,
            vocab_size=self.vocab_size,
            blank_index=self.blank_index,
            parameter_count=sum(parameter.numel() for parameter in self.parameters()),
            trainable_parameter_count=sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        )


def validate_ctc_lengths(input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> None:
    invalid = input_lengths < target_lengths
    if bool(invalid.any().item()):
        bad = invalid.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"CTC impossible length: input_length < target_length for batch indices {bad}")
