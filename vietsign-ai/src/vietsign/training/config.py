"""Training configuration for keypoint recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    name: str = "transformer_baseline"
    output_dir: Path = Path("outputs/runs")

class TaskConfig(BaseModel):
    type: Literal["isolated_recognition", "continuous_recognition", "gloss_translation"] = "isolated_recognition"


class TrainingDatasetConfig(BaseModel):
    config: Path
    cache_root: Path = Path("data/cache/keypoints")
    expected_fingerprint: str | None = None
    split_strategy: str = "signer_independent"


class SequenceConfig(BaseModel):
    mode: Literal["variable", "fixed"] = "fixed"
    length: int | None = Field(default=64, gt=0)


class InputFeatureConfig(BaseModel):
    left_hand: bool = True
    right_hand: bool = True
    pose: bool = True
    validity: bool = True
    confidence: bool = False
    source_mask: bool = False
    velocity: bool = False

class RGBVideoConfig(BaseModel):
    num_frames: int = Field(default=16, gt=0)
    sampling: Literal["uniform", "random"] = "uniform"

class RGBPreprocessingConfig(BaseModel):
    resize: int = Field(default=224, gt=0)
    crop_size: int = Field(default=224, gt=0)
    mean: list[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225])
    horizontal_flip: bool = False
    random_crop: bool = False
    cache_root: Path = Path("data/cache/rgb/videomae")


class DataQualityConfig(BaseModel):
    minimum_valid_joint_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    reject_empty: bool = True


class ModelConfig(BaseModel):
    type: Literal[
        "temporal_transformer",
        "flat_feature_transformer",
        "stgcn",
        "ctrgcn",
        "videomae",
        "multimodal",
        "ctc_recognizer",
        "gloss_transformer",
        "pretrained_seq2seq",
    ] = "temporal_transformer"
    input_mode: Literal["canonical_keypoints", "flat_features"] = "canonical_keypoints"
    input_dim: int | None = Field(default=None, gt=0)
    embedding_dim: int = Field(default=256, gt=0)
    num_layers: int = Field(default=4, gt=0)
    num_heads: int = Field(default=8, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    positional_encoding: Literal["sinusoidal", "learned"] = "sinusoidal"
    channels: list[int] = Field(default_factory=lambda: [64, 64, 128, 256])
    graph: dict[str, str] = Field(default_factory=lambda: {"type": "vietsign55"})
    relation_reduction: int = Field(default=8, gt=0)
    pretrained_model: str = "MCG-NJU/videomae-base"
    pretrained: bool = True
    freeze_backbone: bool = True
    hidden_dim: int = Field(default=128, gt=0)
    fusion_type: Literal["concat", "gated", "cross_attention"] = "concat"
    shared_dim: int = Field(default=128, gt=0)
    freeze_rgb: bool = True
    freeze_pose: bool = False
    modality_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    rgb_checkpoint: Path | None = None
    pose_checkpoint: Path | None = None
    load_pretrained_encoders: bool = False
    d_model: int = Field(default=128, gt=0)
    encoder_layers: int = Field(default=2, gt=0)
    decoder_layers: int = Field(default=2, gt=0)
    dim_feedforward: int = Field(default=256, gt=0)
    task_prefix: str = ""


class LoopConfig(BaseModel):
    epochs: int = Field(default=50, gt=0)
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    gradient_clip_norm: float | None = Field(default=1.0, gt=0.0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    mixed_precision: bool = False
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"


class OptimizerConfig(BaseModel):
    type: Literal["adamw"] = "adamw"
    rgb_lr: float | None = Field(default=None, gt=0.0)
    pose_lr: float | None = Field(default=None, gt=0.0)
    fusion_lr: float | None = Field(default=None, gt=0.0)


class SchedulerConfig(BaseModel):
    type: Literal["none", "cosine", "plateau"] = "none"


class LossConfig(BaseModel):
    class_weighting: Literal["none", "inverse_frequency"] = "none"


class EarlyStoppingConfig(BaseModel):
    enabled: bool = True
    patience: int = Field(default=10, ge=1)
    metric: Literal["macro_f1", "top1_accuracy", "loss", "bleu", "chrf", "exact_match"] = "macro_f1"


class EvaluationConfig(BaseModel):
    fail_on_signer_overlap: bool = False
    top_k: int = Field(default=5, gt=0)

class CTCConfig(BaseModel):
    blank_token: str = "<blank>"
    zero_infinity: bool = True


class TranslationConfig(BaseModel):
    lowercase_vietnamese: bool = False
    uppercase_glosses: bool = True
    max_source_length: int = Field(default=64, gt=0)
    max_target_length: int = Field(default=64, gt=0)
    reject_over_length: bool = True


class DecodingConfig(BaseModel):
    type: Literal["greedy", "beam"] = "greedy"
    beam_size: int = Field(default=4, gt=0)
    max_new_tokens: int = Field(default=64, gt=0)
    length_penalty: float = Field(default=1.0, gt=0.0)


class RecognitionTrainingConfig(BaseModel):
    seed: int = 42
    task: TaskConfig = Field(default_factory=TaskConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    dataset: TrainingDatasetConfig
    sequence: SequenceConfig = Field(default_factory=SequenceConfig)
    features: InputFeatureConfig = Field(default_factory=InputFeatureConfig)
    video: RGBVideoConfig = Field(default_factory=RGBVideoConfig)
    rgb: RGBPreprocessingConfig = Field(default_factory=RGBPreprocessingConfig)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: LoopConfig = Field(default_factory=LoopConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    ctc: CTCConfig = Field(default_factory=CTCConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    decoding: DecodingConfig = Field(default_factory=DecodingConfig)


def load_training_config(path: str | Path) -> RecognitionTrainingConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}
    return RecognitionTrainingConfig.model_validate(raw_config)
