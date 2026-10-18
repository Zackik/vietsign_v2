"""Structured inference errors."""

from __future__ import annotations


class InferenceError(RuntimeError):
    code = "internal_inference_failure"
    status_code = 500

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidVideoError(InferenceError):
    code = "invalid_video"
    status_code = 400


class UnsupportedFormatError(InferenceError):
    code = "unsupported_format"
    status_code = 415


class MissingCheckpointError(InferenceError):
    code = "missing_checkpoint"
    status_code = 503


class ModelUnavailableError(InferenceError):
    code = "model_unavailable"
    status_code = 503


class PipelineUnavailableError(InferenceError):
    code = "pipeline_unavailable"
    status_code = 503
