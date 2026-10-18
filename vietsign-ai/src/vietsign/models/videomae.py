"""VideoMAE RGB adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RGBModelIntrospection:
    model_name: str
    modality: str
    pretrained_model: str
    pretrained: bool
    freeze_backbone: bool
    embedding_dim: int
    class_count: int
    parameter_count: int
    trainable_parameter_count: int


class VideoMAEClassifier(nn.Module):
    """VideoMAE-style classifier for RGB clips.

    Input contract is project-level RGB layout `[B,T,C,H,W]`. The Hugging Face
    VideoMAE backbone receives `[B,T,C,H,W]`; the tiny local fallback also uses
    this layout and exists for offline software validation only.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        pretrained_model: str,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_model = pretrained_model
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        if pretrained:
            self.backbone, embedding_dim = _load_hf_videomae(pretrained_model)
            if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()
        else:
            self.backbone = TinyVideoMAEBackbone(embedding_dim=hidden_dim)
            embedding_dim = hidden_dim
        self.embedding_dim = int(embedding_dim)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, num_classes),
        )

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = pixel_values.float()
        if self.freeze_backbone:
            with torch.no_grad():
                return self._encode_backbone(x)
        return self._encode_backbone(x)

    def encode_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return RGB tokens [B,N,D] for fusion.

        The pretrained Hugging Face backbone exposes tubelet tokens. The tiny
        offline backbone exposes one pooled token, which is enough for software
        validation without pretending to be pretrained VideoMAE.
        """

        x = pixel_values.float()
        if self.freeze_backbone:
            with torch.no_grad():
                return self._encode_backbone_tokens(x)
        return self._encode_backbone_tokens(x)

    def forward(self, pixel_values: torch.Tensor, temporal_mask: torch.Tensor | None = None) -> torch.Tensor:
        del temporal_mask
        return self.classifier(self.encode(pixel_values))

    def introspect(self) -> RGBModelIntrospection:
        return RGBModelIntrospection(
            model_name="videomae",
            modality="rgb",
            pretrained_model=self.pretrained_model,
            pretrained=self.pretrained,
            freeze_backbone=self.freeze_backbone,
            embedding_dim=self.embedding_dim,
            class_count=self.num_classes,
            parameter_count=sum(parameter.numel() for parameter in self.parameters()),
            trainable_parameter_count=sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        )

    def _encode_backbone(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values)
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state.mean(dim=1)
        raise TypeError("Unsupported VideoMAE backbone output.")

    def _encode_backbone_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values)
        if isinstance(output, torch.Tensor):
            return output.unsqueeze(1)
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        raise TypeError("Unsupported VideoMAE backbone output.")


class RGBEmbeddingClassifier(nn.Module):
    """Lightweight classifier for cached RGB embeddings."""

    def __init__(self, *, embedding_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(embedding.float())


class TinyVideoMAEBackbone(nn.Module):
    """Tiny offline VideoMAE-compatible backbone for software validation only."""

    def __init__(self, *, embedding_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.projection = nn.Linear(32, embedding_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
        x = self.net(x).flatten(1)
        return self.projection(x)


def _load_hf_videomae(model_id: str):
    try:
        from transformers import VideoMAEModel  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "VideoMAE pretrained mode requires transformers. Install with "
            "`python -m pip install -e '.[transformers]'`, or set model.pretrained=false "
            "for offline software validation."
        ) from error
    try:
        model = VideoMAEModel.from_pretrained(model_id)
    except Exception as error:
        raise RuntimeError(
            f"Unable to load pretrained VideoMAE model {model_id!r}. Check network/cache, "
            "or explicitly set model.pretrained=false for non-pretrained smoke tests."
        ) from error
    hidden = int(getattr(model.config, "hidden_size", 768))
    return model, hidden
