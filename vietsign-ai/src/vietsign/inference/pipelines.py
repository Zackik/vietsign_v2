"""Inference pipeline registry."""

from __future__ import annotations

from dataclasses import dataclass

from vietsign.inference.model_manager import ModelManager
from vietsign.inference.schemas import PipelineInfo


@dataclass(frozen=True)
class PipelineDefinition:
    name: str
    task: str
    required_inputs: tuple[str, ...]
    model_task: str | None
    real_video_required: bool = False


PIPELINES = (
    PipelineDefinition("isolated_keypoint", "isolated_recognition", ("video", "checkpoint"), "isolated_recognition", True),
    PipelineDefinition("isolated_rgb", "isolated_recognition", ("video", "checkpoint"), "rgb_recognition", False),
    PipelineDefinition(
        "isolated_multimodal",
        "isolated_recognition",
        ("video", "keypoint_features", "checkpoint"),
        "multimodal_recognition",
        False,
    ),
    PipelineDefinition("continuous_keypoint", "continuous_recognition", ("features_npz", "checkpoint"), "continuous_recognition"),
    PipelineDefinition(
        "continuous_translate",
        "continuous_translation",
        ("features_npz", "cslr_checkpoint", "translation_checkpoint"),
        None,
    ),
    PipelineDefinition("gloss_translation", "gloss_translation", ("gloss_sequence", "checkpoint"), "gloss_translation"),
)


def pipeline_infos(manager: ModelManager) -> list[PipelineInfo]:
    infos = []
    model_tasks = {model.task for model in manager.config.models if model.enabled and model.checkpoint_path}
    for definition in PIPELINES:
        ready = True
        reason = None
        development_only = False
        if definition.model_task is not None and definition.model_task not in model_tasks:
            ready = False
            reason = f"no enabled checkpoint for {definition.model_task}"
        if definition.name == "continuous_translate":
            ready = {"continuous_recognition", "gloss_translation"}.issubset(model_tasks)
            reason = None if ready else "requires both CSLR and translation checkpoints"
        if definition.real_video_required:
            ready = False
            reason = "real extractor acceptance deferred"
        for model in manager.config.models:
            if model.enabled and model.development_only and (
                definition.model_task == model.task or definition.name == "continuous_translate"
            ):
                development_only = True
        infos.append(
            PipelineInfo(
                name=definition.name,
                task=definition.task,
                required_inputs=list(definition.required_inputs),
                ready=ready,
                reason=reason,
                development_only=development_only,
            )
        )
    return infos
