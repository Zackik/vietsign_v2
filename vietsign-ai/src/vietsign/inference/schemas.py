"""Pydantic schemas shared by the API and inference service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProcessingMetadata(BaseModel):
    total_ms: float
    stages: dict[str, float] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    request_id: str
    error: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    name: str
    task: str
    modality: str
    checkpoint: str | None = None
    loaded: bool = False
    parameter_count: int | None = None
    device: str | None = None
    development_only: bool = False
    maturity: str = "development"
    enabled: bool = True
    warnings: list[str] = Field(default_factory=list)


class PipelineInfo(BaseModel):
    name: str
    task: str
    required_inputs: list[str]
    ready: bool
    reason: str | None = None
    development_only: bool = False


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    version: str
    device: str
    available_pipelines: list[str]
    extractor_status: str
    loaded_models_count: int
    real_vsl_ready: bool
    readiness: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class TranslationRequest(BaseModel):
    glosses: list[str] | None = None
    gloss: str | None = None
    model: str | None = None


class FeatureRequest(BaseModel):
    features_path: str
    model: str | None = None


class RecognizeTranslateRequest(BaseModel):
    features_path: str
    cslr_model: str | None = None
    translation_model: str | None = None


class TranslationResult(BaseModel):
    request_id: str
    task: Literal["gloss_translation"] = "gloss_translation"
    source_gloss: str
    translation: str
    processing: ProcessingMetadata
    model: ModelInfo | None = None
    development_only: bool = False
    warnings: list[str] = Field(default_factory=list)


class ContinuousRecognitionResult(BaseModel):
    request_id: str
    task: Literal["continuous_recognition"] = "continuous_recognition"
    glosses: list[str]
    decoder: str
    processing: ProcessingMetadata
    model: ModelInfo | None = None
    development_only: bool = False
    warnings: list[str] = Field(default_factory=list)


class IsolatedRecognitionResult(BaseModel):
    request_id: str
    task: Literal["isolated_recognition"] = "isolated_recognition"
    label: str
    confidence: float
    top_k: list[dict[str, Any]] = Field(default_factory=list)
    processing: ProcessingMetadata
    model: ModelInfo | None = None
    development_only: bool = False
    warnings: list[str] = Field(default_factory=list)


class RecognitionTranslationResult(BaseModel):
    request_id: str
    task: Literal["continuous_translation"] = "continuous_translation"
    glosses: list[str]
    translation: str
    processing: ProcessingMetadata
    recognition_model: ModelInfo | None = None
    translation_model: ModelInfo | None = None
    development_only: bool = False
    warnings: list[str] = Field(default_factory=list)
