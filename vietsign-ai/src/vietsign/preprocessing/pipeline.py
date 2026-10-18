"""Keypoint preprocessing pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from vietsign.data.adapters import create_adapter
from vietsign.data.config import load_dataset_config
from vietsign.data.schema import SignSample
from vietsign.data.video import inspect_video
from vietsign.preprocessing.cache import (
    CacheMismatchError,
    cache_paths,
    load_keypoint_cache,
    preprocessing_fingerprint,
    save_keypoint_cache,
)
from vietsign.preprocessing.config import PreprocessingConfig
from vietsign.vision.doctor import extractor_doctor
from vietsign.vision.extractors import ExtractorUnavailableError
from vietsign.vision.mediapipe import MediaPipeConfig, MediaPipeKeypointExtractor
from vietsign.vision.mediapipe_tasks import MediaPipeTasksConfig, MediaPipeTasksKeypointExtractor
from vietsign.vision.processing import (
    compute_acceleration,
    compute_velocity,
    detection_coverage,
    interpolate_short_gaps,
    normalize_tensors,
)
from vietsign.vision.quality import sequence_quality, smooth_ema
from vietsign.vision.rtmpose import RTMPoseConfig, RTMPoseKeypointExtractor
from vietsign.vision.sampling import sample_video_frames


class ExtractionQualityError(RuntimeError):
    """Raised when extraction produced no usable landmarks."""


@dataclass
class SamplePreprocessResult:
    sample_id: str
    status: str
    cache_npz: str | None = None
    cache_metadata: str | None = None
    coverage: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    frame_count: int = 0
    interpolated_values: int = 0
    elapsed_seconds: float = 0.0


def create_extractor(config: PreprocessingConfig):
    if config.extractor.type == "mediapipe":
        return MediaPipeKeypointExtractor(
            MediaPipeConfig(
                hands=config.extractor.hands,
                pose=config.extractor.pose,
                face=config.extractor.face,
            )
        )
    if config.extractor.type == "mediapipe_tasks":
        readiness = extractor_doctor(config)
        if not readiness["ready_for_preprocessing"]:
            raise ExtractorUnavailableError(
                "MediaPipe Tasks extractor is not ready for safe in-process inference. "
                f"Doctor report: {readiness['mediapipe']}"
            )
        return MediaPipeTasksKeypointExtractor(
            MediaPipeTasksConfig(
                hands=config.extractor.hands,
                pose=config.extractor.pose,
                face=config.extractor.face,
                hand_model=config.extractor.hand_model,
                pose_model=config.extractor.pose_model,
                face_model=config.extractor.face_model,
                mirrored=config.input.mirrored,
            )
        )
    if config.extractor.type == "rtmpose":
        return RTMPoseKeypointExtractor(
            RTMPoseConfig(
                runtime=config.rtmpose.runtime,
                backend=config.rtmpose.backend,
                device=config.rtmpose.device,
                mode=config.rtmpose.mode,
                keypoint_threshold=config.rtmpose.keypoint_threshold,
                det_model=config.rtmpose.det_model,
                pose_model=config.rtmpose.pose_model,
            )
        )
    raise ValueError(f"Unsupported extractor: {config.extractor.type}")


def preprocess_sample(
    sample: SignSample,
    config: PreprocessingConfig,
    *,
    extractor: MediaPipeKeypointExtractor | None = None,
) -> SamplePreprocessResult:
    """Preprocess one sample and optionally save cache."""

    started = time.perf_counter()
    fingerprint = preprocessing_fingerprint(config)
    npz_path, metadata_path = cache_paths(config.cache.root, sample)
    if config.cache.enabled and config.cache.reuse and npz_path.exists() and metadata_path.exists():
        try:
            arrays, metadata = load_keypoint_cache(
                npz_path, metadata_path, expected_fingerprint=fingerprint
            )
            coverage = metadata.get("coverage", {})
            return SamplePreprocessResult(
                sample_id=sample.sample_id,
                status="cached",
                cache_npz=str(npz_path),
                cache_metadata=str(metadata_path),
                coverage=coverage,
                frame_count=int(arrays["coordinates"].shape[0]),
                elapsed_seconds=time.perf_counter() - started,
            )
        except CacheMismatchError:
            pass

    video_metadata = inspect_video(sample.video_path)
    if not video_metadata.readable:
        return SamplePreprocessResult(
            sample_id=sample.sample_id,
            status="failed",
            error=video_metadata.error or "unreadable video",
            elapsed_seconds=time.perf_counter() - started,
        )

    active_extractor = extractor or create_extractor(config)
    try:
        frames = sample_video_frames(
            sample.video_path,
            target_fps=config.video.target_fps,
            num_frames=config.video.num_frames,
            sampling=config.video.sampling,
        )
        sequence = active_extractor.extract_video(
            frames,
            source_video=sample.video_path,
            fps=video_metadata.fps,
        )
        tensors = sequence.to_tensors()
        if active_extractor.name in {"mediapipe_tasks", "rtmpose"} and int(tensors.valid_mask.sum()) == 0:
            raise ExtractionQualityError("No valid landmarks detected in the entire video.")
        image_size = None
        if video_metadata.width is not None and video_metadata.height is not None:
            image_size = (video_metadata.width, video_metadata.height)
        tensors = normalize_tensors(tensors, method=config.normalization.type, image_size=image_size)
        if config.interpolation.enabled:
            tensors = interpolate_short_gaps(tensors, max_gap=config.interpolation.max_gap)
        if config.smoothing.enabled:
            tensors = smooth_ema(tensors, alpha=config.smoothing.alpha)
        velocity = compute_velocity(tensors) if config.features.velocity else None
        acceleration = (
            compute_acceleration(velocity, tensors.valid_mask)
            if config.features.acceleration and velocity is not None
            else None
        )
        coverage = detection_coverage(tensors)
        quality = sequence_quality(tensors)
        interpolated_values = int((tensors.source_mask == 2).sum())

        if config.cache.enabled:
            save_keypoint_cache(
                npz_path,
                metadata_path,
                tensors,
                sample=sample,
                extra_arrays={
                    name: value
                    for name, value in {"velocity": velocity, "acceleration": acceleration}.items()
                    if value is not None
                },
                metadata={
                    "fingerprint": fingerprint,
                    "extractor": active_extractor.name,
                    "extractor_version": active_extractor.version,
                    "extractor_status": active_extractor.status,
                    "normalization": config.normalization.type,
                    "coverage": coverage,
                    "quality": quality,
                    "smoothing": config.smoothing.model_dump(mode="json"),
                    "velocity": velocity is not None,
                    "acceleration": acceleration is not None,
                    "tensor_shape": {
                        "coordinates": list(tensors.coordinates.shape),
                        "valid_mask": list(tensors.valid_mask.shape),
                    },
                },
            )

        return SamplePreprocessResult(
            sample_id=sample.sample_id,
            status="processed",
            cache_npz=str(npz_path) if config.cache.enabled else None,
            cache_metadata=str(metadata_path) if config.cache.enabled else None,
            coverage=coverage,
            frame_count=len(frames),
            interpolated_values=interpolated_values,
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as error:  # noqa: BLE001
        return SamplePreprocessResult(
            sample_id=sample.sample_id,
            status="failed",
            error=str(error),
            elapsed_seconds=time.perf_counter() - started,
        )
    finally:
        if extractor is None:
            active_extractor.close()


def preprocess_dataset(config: PreprocessingConfig) -> dict[str, Any]:
    dataset_config = load_dataset_config(config.dataset_config)
    samples = create_adapter(dataset_config).load_samples()
    extractor = create_extractor(config)
    results: list[SamplePreprocessResult] = []
    try:
        for sample in samples:
            results.append(preprocess_sample(sample, config, extractor=extractor))
    finally:
        extractor.close()

    low_coverage = [
        result.sample_id
        for result in results
        if result.status in {"processed", "cached"}
        and (
            result.coverage.get("left_hand_coverage", 0.0)
            < config.warnings.hand_coverage_threshold
            or result.coverage.get("right_hand_coverage", 0.0)
            < config.warnings.hand_coverage_threshold
        )
    ]
    return {
        "processed": sum(result.status == "processed" for result in results),
        "cached": sum(result.status == "cached" for result in results),
        "failed": sum(result.status == "failed" for result in results),
        "low_hand_coverage": low_coverage,
        "results": [result.__dict__ for result in results],
    }
