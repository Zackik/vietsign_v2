"""Model registry and factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vietsign.models.continuous import ContinuousSignRecognizer
from vietsign.models.flat_features import FlatFeatureTemporalTransformerClassifier
from vietsign.models.graph import graph_for_features
from vietsign.models.graph_models import CTRGCNClassifier, STGCNClassifier
from vietsign.models.multimodal import MultimodalFusionClassifier
from vietsign.models.temporal_transformer import TemporalTransformerClassifier
from vietsign.models.translation import GlossToVietnameseTransformer
from vietsign.models.videomae import VideoMAEClassifier
from vietsign.training.config import RecognitionTrainingConfig


@dataclass(frozen=True)
class InputSpec:
    joint_count: int
    channel_count: int


def build_model(
    config: RecognitionTrainingConfig,
    *,
    num_classes: int,
    input_spec: InputSpec,
) -> torch.nn.Module:
    model_type = config.model.type
    vocab_size = getattr(config, "_runtime_vocab_size", None)
    blank_index = getattr(config, "_runtime_blank_index", 0)
    if model_type == "ctc_recognizer":
        if vocab_size is None:
            raise ValueError("CTC recognizer requires runtime vocab size.")
        return ContinuousSignRecognizer(
            joint_count=input_spec.joint_count,
            input_channels=input_spec.channel_count,
            vocab_size=int(vocab_size),
            blank_index=int(blank_index),
            embedding_dim=config.model.embedding_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
        )
    if model_type == "gloss_transformer":
        source_vocab_size = getattr(config, "_runtime_source_vocab_size", None)
        target_vocab_size = getattr(config, "_runtime_target_vocab_size", None)
        if source_vocab_size is None or target_vocab_size is None:
            raise ValueError("Gloss transformer requires runtime source and target vocabulary sizes.")
        return GlossToVietnameseTransformer(
            source_vocab_size=int(source_vocab_size),
            target_vocab_size=int(target_vocab_size),
            pad_index=0,
            bos_index=1,
            eos_index=2,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            encoder_layers=config.model.encoder_layers,
            decoder_layers=config.model.decoder_layers,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            max_length=max(config.translation.max_source_length, config.translation.max_target_length) + 4,
        )
    if model_type == "temporal_transformer":
        return TemporalTransformerClassifier(
            joint_count=input_spec.joint_count,
            input_channels=input_spec.channel_count,
            num_classes=num_classes,
            embedding_dim=config.model.embedding_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
        )
    if model_type == "flat_feature_transformer":
        input_dim = config.model.input_dim or input_spec.channel_count
        return FlatFeatureTemporalTransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            embedding_dim=config.model.embedding_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
        )
    if model_type == "stgcn":
        return STGCNClassifier(
            graph=graph_for_features(config.features),
            input_channels=input_spec.channel_count,
            num_classes=num_classes,
            channels=tuple(config.model.channels),
            dropout=config.model.dropout,
        )
    if model_type == "ctrgcn":
        return CTRGCNClassifier(
            graph=graph_for_features(config.features),
            input_channels=input_spec.channel_count,
            num_classes=num_classes,
            channels=tuple(config.model.channels),
            dropout=config.model.dropout,
            relation_reduction=config.model.relation_reduction,
        )
    if model_type == "videomae":
        return VideoMAEClassifier(
            num_classes=num_classes,
            pretrained_model=config.model.pretrained_model,
            pretrained=config.model.pretrained,
            freeze_backbone=config.model.freeze_backbone,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
        )
    if model_type == "multimodal":
        rgb_encoder = VideoMAEClassifier(
            num_classes=num_classes,
            pretrained_model=config.model.pretrained_model,
            pretrained=config.model.pretrained,
            freeze_backbone=config.model.freeze_rgb,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
        )
        pose_encoder = TemporalTransformerClassifier(
            joint_count=input_spec.joint_count,
            input_channels=input_spec.channel_count,
            num_classes=num_classes,
            embedding_dim=config.model.embedding_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
        )
        return MultimodalFusionClassifier(
            rgb_encoder=rgb_encoder,
            pose_encoder=pose_encoder,
            num_classes=num_classes,
            fusion_type=config.model.fusion_type,
            shared_dim=config.model.shared_dim,
            dropout=config.model.dropout,
            freeze_rgb=config.model.freeze_rgb,
            freeze_pose=config.model.freeze_pose,
            modality_dropout=config.model.modality_dropout,
            num_heads=config.model.num_heads,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def model_modality(model_type: str) -> str:
    if model_type in {"temporal_transformer", "stgcn", "ctrgcn"}:
        return "keypoints"
    if model_type == "flat_feature_transformer":
        return "flat_features"
    if model_type == "videomae":
        return "rgb"
    if model_type == "multimodal":
        return "multimodal"
    if model_type == "ctc_recognizer":
        return "continuous"
    if model_type in {"gloss_transformer", "pretrained_seq2seq"}:
        return "text"
    raise ValueError(f"Unsupported model type: {model_type}")


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_metadata(model: torch.nn.Module) -> dict[str, Any]:
    if hasattr(model, "introspect"):
        return model.introspect().__dict__
    return {
        "model_name": model.__class__.__name__,
        "parameter_count": parameter_count(model),
    }
