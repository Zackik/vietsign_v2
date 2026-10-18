"""Temporal Transformer baseline for isolated sign recognition."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelIntrospection:
    model_name: str
    input_joints: int
    input_channels: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    parameter_count: int
    class_count: int


class FrameEncoder(nn.Module):
    """Encode joint features for one frame into a temporal embedding."""

    def __init__(self, joint_count: int, input_channels: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.joint_projection = nn.Linear(input_channels, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.frame_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        projected = self.joint_projection(x)
        joint_weights = valid_mask.unsqueeze(-1).to(projected.dtype)
        weighted = projected * joint_weights
        denom = joint_weights.sum(dim=2).clamp_min(1.0)
        pooled = weighted.sum(dim=2) / denom
        return self.frame_projection(self.dropout(self.norm(pooled)))


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal temporal positional encoding."""

    def __init__(self, embedding_dim: int, max_length: int = 4096) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embedding_dim)
        )
        encoding = torch.zeros((max_length, embedding_dim), dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.shape[1], :]


class TemporalTransformerClassifier(nn.Module):
    """Temporal Transformer baseline over cached keypoint tensors."""

    def __init__(
        self,
        *,
        joint_count: int,
        input_channels: int,
        num_classes: int,
        embedding_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.joint_count = joint_count
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.frame_encoder = FrameEncoder(joint_count, input_channels, embedding_dim, dropout)
        self.position = SinusoidalPositionalEncoding(embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits [B,num_classes] from inputs [B,T,J,C]."""

        return self.classifier(self.forward_features(coordinates, valid_mask, temporal_mask))

    def forward_tokens(
        self,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return temporal pose tokens [B,T,D] before pooling."""

        x = torch.nan_to_num(coordinates.float(), nan=0.0, posinf=0.0, neginf=0.0)
        valid_mask = valid_mask.bool()
        temporal_mask = temporal_mask.bool()
        encoded = self.position(self.frame_encoder(x, valid_mask))
        encoded = self.encoder(encoded, src_key_padding_mask=~temporal_mask)
        return self.norm(encoded)

    def forward_features(
        self,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return pooled pose embedding [B,D]."""

        return masked_mean_pooling(self.forward_tokens(coordinates, valid_mask, temporal_mask), temporal_mask)

    def introspect(self) -> ModelIntrospection:
        return ModelIntrospection(
            model_name="temporal_transformer",
            input_joints=self.joint_count,
            input_channels=self.input_channels,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            parameter_count=sum(parameter.numel() for parameter in self.parameters()),
            class_count=self.num_classes,
        )


def masked_mean_pooling(values: torch.Tensor, temporal_mask: torch.Tensor) -> torch.Tensor:
    weights = temporal_mask.unsqueeze(-1).to(values.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denom
