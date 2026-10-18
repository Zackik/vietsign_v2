"""Multimodal RGB + keypoint fusion models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from vietsign.models.temporal_transformer import TemporalTransformerClassifier, masked_mean_pooling
from vietsign.models.videomae import VideoMAEClassifier


@dataclass(frozen=True)
class MultimodalIntrospection:
    model_name: str
    modality: str
    fusion_type: str
    rgb_encoder_type: str
    pose_encoder_type: str
    shared_dim: int
    rgb_embedding_dim: int
    pose_embedding_dim: int
    class_count: int
    rgb_parameter_count: int
    rgb_trainable_parameter_count: int
    pose_parameter_count: int
    pose_trainable_parameter_count: int
    fusion_parameter_count: int
    parameter_count: int
    trainable_parameter_count: int


class MultimodalFusionClassifier(nn.Module):
    """RGB + pose fusion classifier.

    Contract:
    - RGB: `pixel_values=[B,T_rgb,C,H,W]`
    - Pose: `coordinates=[B,T_pose,J,C_pose]`, masks `[B,T_pose,J]`, `[B,T_pose]`
    """

    def __init__(
        self,
        *,
        rgb_encoder: VideoMAEClassifier,
        pose_encoder: TemporalTransformerClassifier,
        num_classes: int,
        fusion_type: str = "concat",
        shared_dim: int = 128,
        dropout: float = 0.1,
        freeze_rgb: bool = True,
        freeze_pose: bool = False,
        modality_dropout: float = 0.0,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if fusion_type not in {"concat", "gated", "cross_attention"}:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        self.rgb_encoder = rgb_encoder
        self.pose_encoder = pose_encoder
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        self.shared_dim = shared_dim
        self.modality_dropout = modality_dropout
        _set_requires_grad(self.rgb_encoder, not freeze_rgb)
        _set_requires_grad(self.pose_encoder, not freeze_pose)
        self.rgb_projection = nn.Linear(rgb_encoder.embedding_dim, shared_dim)
        self.pose_projection = nn.Linear(pose_encoder.embedding_dim, shared_dim)
        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.LayerNorm(shared_dim * 2),
                nn.Linear(shared_dim * 2, shared_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif fusion_type == "gated":
            self.gate = nn.Linear(shared_dim * 2, shared_dim)
            self.fusion = nn.Sequential(nn.LayerNorm(shared_dim), nn.Dropout(dropout))
        else:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=shared_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.fusion = nn.Sequential(nn.LayerNorm(shared_dim), nn.Dropout(dropout))
        self.classifier = nn.Linear(shared_dim, num_classes)

    def encode_rgb(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.rgb_encoder.encode(pixel_values)

    def encode_pose(
        self,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.pose_encoder.forward_features(coordinates, valid_mask, temporal_mask)

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
        temporal_mask: torch.Tensor,
        modality_mask: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        fused = self.forward_features(
            pixel_values=pixel_values,
            coordinates=coordinates,
            valid_mask=valid_mask,
            temporal_mask=temporal_mask,
            modality_mask=modality_mask,
        )
        return self.classifier(fused)

    def forward_features(
        self,
        *,
        pixel_values: torch.Tensor,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
        temporal_mask: torch.Tensor,
        modality_mask: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        rgb_embedding = self.rgb_projection(self.encode_rgb(pixel_values))
        pose_embedding = self.pose_projection(self.encode_pose(coordinates, valid_mask, temporal_mask))
        rgb_embedding, pose_embedding = self._apply_modality_controls(
            rgb_embedding,
            pose_embedding,
            modality_mask=modality_mask,
        )
        if self.fusion_type == "concat":
            return self.fusion(torch.cat([rgb_embedding, pose_embedding], dim=-1))
        if self.fusion_type == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([rgb_embedding, pose_embedding], dim=-1)))
            return self.fusion(gate * rgb_embedding + (1.0 - gate) * pose_embedding)

        rgb_tokens = self.rgb_projection(self.rgb_encoder.encode_tokens(pixel_values))
        pose_tokens = self.pose_projection(
            self.pose_encoder.forward_tokens(coordinates, valid_mask, temporal_mask)
        )
        attended, _ = self.cross_attention(
            query=pose_tokens,
            key=rgb_tokens,
            value=rgb_tokens,
            need_weights=False,
        )
        return self.fusion(masked_mean_pooling(attended + pose_tokens, temporal_mask.bool()))

    def embedding_report(
        self,
        *,
        pixel_values: torch.Tensor,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> dict[str, list[int]]:
        with torch.no_grad():
            rgb = self.encode_rgb(pixel_values)
            pose = self.encode_pose(coordinates, valid_mask, temporal_mask)
            projected_rgb = self.rgb_projection(rgb)
            projected_pose = self.pose_projection(pose)
            fused = self.forward_features(
                pixel_values=pixel_values,
                coordinates=coordinates,
                valid_mask=valid_mask,
                temporal_mask=temporal_mask,
            )
            logits = self.classifier(fused)
        return {
            "rgb_embedding": list(rgb.shape),
            "pose_embedding": list(pose.shape),
            "projected_rgb": list(projected_rgb.shape),
            "projected_pose": list(projected_pose.shape),
            "fused": list(fused.shape),
            "logits": list(logits.shape),
        }

    def introspect(self) -> MultimodalIntrospection:
        fusion_params = sum(parameter.numel() for name, parameter in self.named_parameters() if _is_fusion(name))
        return MultimodalIntrospection(
            model_name="multimodal",
            modality="multimodal",
            fusion_type=self.fusion_type,
            rgb_encoder_type="videomae",
            pose_encoder_type="temporal_transformer",
            shared_dim=self.shared_dim,
            rgb_embedding_dim=self.rgb_encoder.embedding_dim,
            pose_embedding_dim=self.pose_encoder.embedding_dim,
            class_count=self.num_classes,
            rgb_parameter_count=sum(parameter.numel() for parameter in self.rgb_encoder.parameters()),
            rgb_trainable_parameter_count=sum(
                parameter.numel() for parameter in self.rgb_encoder.parameters() if parameter.requires_grad
            ),
            pose_parameter_count=sum(parameter.numel() for parameter in self.pose_encoder.parameters()),
            pose_trainable_parameter_count=sum(
                parameter.numel() for parameter in self.pose_encoder.parameters() if parameter.requires_grad
            ),
            fusion_parameter_count=fusion_params,
            parameter_count=sum(parameter.numel() for parameter in self.parameters()),
            trainable_parameter_count=sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        )

    def _apply_modality_controls(
        self,
        rgb_embedding: torch.Tensor,
        pose_embedding: torch.Tensor,
        *,
        modality_mask: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if modality_mask is not None:
            if "rgb" in modality_mask:
                rgb_embedding = rgb_embedding * modality_mask["rgb"].to(rgb_embedding.device).float().unsqueeze(-1)
            if "pose" in modality_mask:
                pose_embedding = pose_embedding * modality_mask["pose"].to(pose_embedding.device).float().unsqueeze(-1)
        if self.training and self.modality_dropout > 0:
            keep_rgb = (torch.rand((rgb_embedding.shape[0], 1), device=rgb_embedding.device) > self.modality_dropout).float()
            keep_pose = (torch.rand((pose_embedding.shape[0], 1), device=pose_embedding.device) > self.modality_dropout).float()
            both_dropped = (keep_rgb + keep_pose) == 0
            keep_pose = torch.where(both_dropped, torch.ones_like(keep_pose), keep_pose)
            rgb_embedding = rgb_embedding * keep_rgb
            pose_embedding = pose_embedding * keep_pose
        return rgb_embedding, pose_embedding


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def _is_fusion(name: str) -> bool:
    return not name.startswith(("rgb_encoder.", "pose_encoder."))
