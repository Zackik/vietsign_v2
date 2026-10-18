"""Flat temporal feature adapters for source-layout sequence baselines."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from vietsign.models.temporal_transformer import (
    SinusoidalPositionalEncoding,
    masked_mean_pooling,
)


@dataclass(frozen=True)
class FlatFeatureModelIntrospection:
    model_name: str
    input_mode: str
    input_dim: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    parameter_count: int
    class_count: int
    representation: str


class FlatTemporalFeatureProjector(nn.Module):
    """Project native per-frame source features to Transformer token embeddings."""

    def __init__(
        self,
        *,
        input_dim: int,
        embedding_dim: int,
        dropout: float = 0.1,
        layer_norm_input: bool = True,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim) if layer_norm_input else nn.Identity()
        self.projection = nn.Linear(input_dim, embedding_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return self.output_norm(self.dropout(self.activation(self.projection(self.input_norm(x)))))


class FlatFeatureTemporalTransformerClassifier(nn.Module):
    """Temporal Transformer classifier for native flat features like VOYA [T,1605]."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.projector = FlatTemporalFeatureProjector(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
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

    def forward(self, features: torch.Tensor, temporal_mask: torch.Tensor) -> torch.Tensor:
        """Return logits [B,num_classes] from native flat inputs [B,T,F]."""

        return self.classifier(self.forward_features(features, temporal_mask))

    def forward_tokens(self, features: torch.Tensor, temporal_mask: torch.Tensor) -> torch.Tensor:
        temporal_mask = temporal_mask.bool()
        encoded = self.position(self.projector(features))
        encoded = self.encoder(encoded, src_key_padding_mask=~temporal_mask)
        return self.norm(encoded)

    def forward_features(self, features: torch.Tensor, temporal_mask: torch.Tensor) -> torch.Tensor:
        return masked_mean_pooling(self.forward_tokens(features, temporal_mask), temporal_mask)

    def introspect(self) -> FlatFeatureModelIntrospection:
        return FlatFeatureModelIntrospection(
            model_name="flat_feature_temporal_transformer",
            input_mode="flat_features",
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            parameter_count=sum(parameter.numel() for parameter in self.parameters()),
            class_count=self.num_classes,
            representation="native_1605",
        )
