"""Lazy model/checkpoint manager for product inference."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from vietsign.inference.config import ApiConfig, InferenceModelConfig
from vietsign.inference.errors import MissingCheckpointError, ModelUnavailableError
from vietsign.inference.schemas import ModelInfo
from vietsign.training.checkpoint import load_checkpoint
from vietsign.training.config import RecognitionTrainingConfig, load_training_config
from vietsign.utils.device import select_device


class ModelManager:
    """Registry-backed lazy manager.

    The Phase 11 API keeps heavy model construction inside the existing Phase
    04/09/10 prediction helpers. This manager caches configuration and
    checkpoint metadata so routes do not touch filesystem details directly.
    """

    def __init__(self, config: ApiConfig) -> None:
        self.config = config
        self._models = {model.name: model for model in config.models}
        self._metadata_cache: dict[str, dict[str, Any]] = {}
        self._loaded: set[str] = set()
        self._lock = Lock()
        self.device = select_device(config.device)

    def names(self) -> list[str]:
        return sorted(self._models)

    def get(self, name: str | None, *, task: str | None = None) -> InferenceModelConfig:
        candidates = [model for model in self._models.values() if model.enabled and (task is None or model.task == task)]
        if name is not None:
            if name not in self._models or not self._models[name].enabled:
                raise ModelUnavailableError(f"Model is not registered or enabled: {name}")
            model = self._models[name]
            if task is not None and model.task != task:
                raise ModelUnavailableError(f"Model '{name}' is task={model.task}, expected {task}.")
            return model
        if not candidates:
            raise ModelUnavailableError(f"No enabled model is registered for task={task}.")
        return candidates[0]

    def training_config(self, model: InferenceModelConfig) -> RecognitionTrainingConfig:
        if model.config_path is None:
            raise ModelUnavailableError(f"Model '{model.name}' has no config_path.")
        return load_training_config(model.config_path)

    def checkpoint_path(self, model: InferenceModelConfig) -> Path:
        if model.checkpoint_path is None:
            raise MissingCheckpointError(f"Model '{model.name}' has no checkpoint_path.")
        path = Path(model.checkpoint_path)
        if not path.exists():
            raise MissingCheckpointError(f"Checkpoint does not exist for model '{model.name}'.", details={"path": str(path)})
        return path

    def checkpoint_metadata(self, model: InferenceModelConfig) -> dict[str, Any]:
        with self._lock:
            if model.name not in self._metadata_cache:
                checkpoint = load_checkpoint(self.checkpoint_path(model), map_location="cpu")
                self._metadata_cache[model.name] = checkpoint.get("model_metadata", {})
            self._loaded.add(model.name)
            return self._metadata_cache[model.name]

    def info(self, model: InferenceModelConfig) -> ModelInfo:
        warnings = []
        parameter_count = None
        loaded = model.name in self._loaded
        if model.checkpoint_path and Path(model.checkpoint_path).exists():
            try:
                metadata = self.checkpoint_metadata(model)
                parameter_count = metadata.get("parameter_count")
                loaded = model.name in self._loaded
            except Exception as error:  # noqa: BLE001
                warnings.append(str(error))
        elif model.checkpoint_path:
            warnings.append("checkpoint missing")
        else:
            warnings.append("checkpoint not configured")
        return ModelInfo(
            name=model.name,
            task=model.task,
            modality=model.modality,
            checkpoint=Path(model.checkpoint_path).name if model.checkpoint_path else None,
            loaded=loaded,
            parameter_count=parameter_count,
            device=self.device.type,
            development_only=model.development_only,
            maturity=model.maturity,
            enabled=model.enabled,
            warnings=warnings,
        )

    def all_info(self) -> list[ModelInfo]:
        return [self.info(model) for model in self._models.values()]

    def loaded_count(self) -> int:
        return len(self._loaded)
