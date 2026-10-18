"""High-level inference orchestration independent of API transport."""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from vietsign.inference.errors import PipelineUnavailableError
from vietsign.inference.model_manager import ModelManager
from vietsign.inference.schemas import (
    ContinuousRecognitionResult,
    IsolatedRecognitionResult,
    ProcessingMetadata,
    RecognitionTranslationResult,
    TranslationResult,
)
from vietsign.training.runner import (
    predict_continuous,
    predict_video,
    recognize_and_translate,
    translate_gloss,
)


class VietSignPipeline:
    def __init__(self, manager: ModelManager) -> None:
        self.manager = manager

    def translate_glosses(self, gloss: str | list[str], *, request_id: str | None = None, model_name: str | None = None) -> TranslationResult:
        request_id = request_id or _request_id()
        started = time.perf_counter()
        model = self.manager.get(model_name, task="gloss_translation")
        result = translate_gloss(
            gloss=gloss,
            checkpoint_path=self.manager.checkpoint_path(model),
            config=self.manager.training_config(model),
        )
        return TranslationResult(
            request_id=request_id,
            source_gloss=result["source_gloss"],
            translation=result["translation"],
            processing=_processing(started),
            model=self.manager.info(model),
            development_only=model.development_only,
            warnings=[result["note"]],
        )

    def recognize_continuous(
        self,
        features_path: str | Path,
        *,
        request_id: str | None = None,
        model_name: str | None = None,
    ) -> ContinuousRecognitionResult:
        request_id = request_id or _request_id()
        started = time.perf_counter()
        model = self.manager.get(model_name, task="continuous_recognition")
        result = predict_continuous(
            features_path=features_path,
            checkpoint_path=self.manager.checkpoint_path(model),
            config=self.manager.training_config(model),
        )
        return ContinuousRecognitionResult(
            request_id=request_id,
            glosses=result["gloss_sequence"],
            decoder=result["decoder"],
            processing=_processing(started),
            model=self.manager.info(model),
            development_only=model.development_only,
            warnings=[result["note"]],
        )

    def recognize_and_translate(
        self,
        features_path: str | Path,
        *,
        request_id: str | None = None,
        cslr_model_name: str | None = None,
        translation_model_name: str | None = None,
    ) -> RecognitionTranslationResult:
        request_id = request_id or _request_id()
        started = time.perf_counter()
        cslr_model = self.manager.get(cslr_model_name, task="continuous_recognition")
        translation_model = self.manager.get(translation_model_name, task="gloss_translation")
        result = recognize_and_translate(
            features_path=features_path,
            cslr_checkpoint=self.manager.checkpoint_path(cslr_model),
            translation_checkpoint=self.manager.checkpoint_path(translation_model),
            cslr_config=self.manager.training_config(cslr_model),
            translation_config=self.manager.training_config(translation_model),
        )
        return RecognitionTranslationResult(
            request_id=request_id,
            glosses=result["recognized_glosses"],
            translation=result["translation"],
            processing=_processing(started),
            recognition_model=self.manager.info(cslr_model),
            translation_model=self.manager.info(translation_model),
            development_only=cslr_model.development_only or translation_model.development_only,
            warnings=[result["note"]],
        )

    def recognize_isolated_rgb(
        self,
        video_path: str | Path,
        *,
        request_id: str | None = None,
        model_name: str | None = None,
        top_k: int = 5,
    ) -> IsolatedRecognitionResult:
        request_id = request_id or _request_id()
        started = time.perf_counter()
        model = self.manager.get(model_name, task="rgb_recognition")
        result = predict_video(
            config=self.manager.training_config(model),
            video_path=video_path,
            checkpoint_path=self.manager.checkpoint_path(model),
            top_k=top_k,
        )
        return IsolatedRecognitionResult(
            request_id=request_id,
            label=result["label"],
            confidence=result["confidence"],
            top_k=result["top_k"],
            processing=_processing(started),
            model=self.manager.info(model),
            development_only=model.development_only,
            warnings=[result["note"]],
        )

    def recognize_isolated_unavailable(self) -> None:
        raise PipelineUnavailableError("Isolated raw-video prediction requires real extractor/checkpoint readiness.")


def _request_id() -> str:
    return uuid4().hex


def _processing(started: float) -> ProcessingMetadata:
    return ProcessingMetadata(total_ms=(time.perf_counter() - started) * 1000.0)
