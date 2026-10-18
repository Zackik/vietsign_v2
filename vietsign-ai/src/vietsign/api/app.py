"""FastAPI application for VietSignAI product inference."""

import csv
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from vietsign import __version__ as VIETSIGN_VERSION
from vietsign.inference.config import ApiConfig, load_api_config
from vietsign.inference.errors import InferenceError, PipelineUnavailableError
from vietsign.inference.hf_videomae import ViSignVideoMAEPredictor
from vietsign.inference.model_manager import ModelManager
from vietsign.inference.orchestrator import VietSignPipeline
from vietsign.inference.pipelines import pipeline_infos
from vietsign.inference.schemas import (
    ErrorResponse,
    FeatureRequest,
    HealthResponse,
    RecognizeTranslateRequest,
    TranslationRequest,
)
from vietsign.inference.video import safe_upload_file
from vietsign.utils.logging import get_logger

LOGGER = get_logger(__name__)


def create_app(config: ApiConfig | None = None):
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, JSONResponse
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install the 'api' optional dependency to run the FastAPI application.") from error

    api_config = config or load_api_config("configs/api/default.yaml")
    manager = ModelManager(api_config)
    pipeline = VietSignPipeline(manager)
    pretrained_predictor = ViSignVideoMAEPredictor(
        "models/pretrained/vi-sign-language-videomae-base",
        device="cpu",
    )
    app = FastAPI(title="VietSignAI API", version=VIETSIGN_VERSION)
    app.state.config = api_config
    app.state.manager = manager
    app.state.pipeline = pipeline
    app.state.pretrained_predictor = pretrained_predictor
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_config.cors.allow_origins,
        allow_credentials=api_config.cors.allow_credentials,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(InferenceError)
    async def inference_error_handler(request: Request, error: InferenceError):
        request_id = getattr(request.state, "request_id", uuid4().hex)
        return JSONResponse(
            status_code=error.status_code,
            content=ErrorResponse(
                request_id=request_id,
                error=error.code,
                message=error.message,
                details=error.details,
            ).model_dump(mode="json"),
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id", uuid4().hex)
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers["x-request-id"] = request.state.request_id
        LOGGER.info(
            "request_id=%s method=%s path=%s status=%s elapsed_ms=%.2f",
            request.state.request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        infos = pipeline_infos(manager)
        ready = [info.name for info in infos if info.ready]
        model_infos = manager.all_info()
        readiness = {
            "real_extractor_ready": False,
            "real_islr_checkpoint_ready": any(
                item.task in {"isolated_recognition", "rgb_recognition"}
                and item.maturity != "development"
                for item in model_infos
            ),
            "real_cslr_checkpoint_ready": any(
                item.task == "continuous_recognition" and item.maturity != "development" for item in model_infos
            ),
            "real_translation_checkpoint_ready": any(
                item.task == "gloss_translation" and item.maturity != "development" for item in model_infos
            ),
        }
        return HealthResponse(
            status="healthy" if ready else "degraded",
            version=VIETSIGN_VERSION,
            device=manager.device.type,
            available_pipelines=ready,
            extractor_status="deferred_real_acceptance",
            loaded_models_count=manager.loaded_count(),
            real_vsl_ready=all(readiness.values()),
            readiness=readiness,
            warnings=["Phase 05 real-video acceptance deferred.", "Synthetic checkpoints are development-only."],
        )

    @app.get("/models")
    def models():
        return {"models": [item.model_dump(mode="json") for item in manager.all_info()]}

    @app.get("/pipelines")
    def pipelines():
        return {"pipelines": [item.model_dump(mode="json") for item in pipeline_infos(manager)]}

    @app.get("/guide/videos")
    def guide_videos(limit: int = 500, label: str | None = None, unique_labels: bool = True):
        samples = _load_guide_samples()
        if label:
            samples = [sample for sample in samples if sample["label"] == label]
        if unique_labels:
            samples = _one_sample_per_label(samples)
        safe_limit = max(limit, 0)
        return {
            "count": min(safe_limit, len(samples)),
            "total": len(samples),
            "videos": samples[:safe_limit],
        }

    @app.get("/guide/videos/{sample_id}/file")
    def guide_video_file(sample_id: str):
        sample = _guide_sample_by_id(sample_id)
        if sample is None:
            raise HTTPException(status_code=404, detail={"message": "Guide video not found.", "sample_id": sample_id})
        path = Path(str(sample["video_path"]))
        if not path.exists():
            raise HTTPException(status_code=404, detail={"message": "Guide video file is missing.", "path": str(path)})
        webm_path = _ensure_browser_guide_video(path, str(sample["sample_id"]))
        return FileResponse(webm_path, media_type="video/webm", filename=webm_path.name)

    @app.post("/translate")
    def translate(request: TranslationRequest, http_request: Request):
        gloss = request.glosses if request.glosses is not None else request.gloss
        if gloss is None:
            raise InferenceError("Provide gloss or glosses.", details={"field": "gloss"})
        return pipeline.translate_glosses(gloss, request_id=http_request.state.request_id, model_name=request.model)

    @app.post("/predict/continuous")
    def predict_continuous(request: FeatureRequest, http_request: Request):
        return pipeline.recognize_continuous(
            request.features_path,
            request_id=http_request.state.request_id,
            model_name=request.model,
        )

    @app.post("/recognize-and-translate")
    def recognize_translate(request: RecognizeTranslateRequest, http_request: Request):
        return pipeline.recognize_and_translate(
            request.features_path,
            request_id=http_request.state.request_id,
            cslr_model_name=request.cslr_model,
            translation_model_name=request.translation_model,
        )

    @app.post("/predict/isolated")
    async def predict_isolated_video(request: Request):
        form = await request.form()
        video = form.get("video")
        if video is None or not hasattr(video, "file"):
            raise InferenceError("Missing uploaded video.", details={"field": "video"})
        model_name = form.get("model")
        top_k = int(form.get("top_k") or 5)
        with safe_upload_file(video.file, video.filename or "upload.mp4", api_config.upload) as video_path:
            return pipeline.recognize_isolated_rgb(
                video_path,
                request_id=request.state.request_id,
                model_name=str(model_name) if model_name else None,
                top_k=top_k,
            )

    @app.post("/predict/pretrained-video")
    async def predict_pretrained_video(request: Request):
        form = await request.form()
        video = form.get("video")
        if video is None or not hasattr(video, "file"):
            raise InferenceError("Missing uploaded video.", details={"field": "video"})
        top_k = int(form.get("top_k") or 5)
        with safe_upload_file(video.file, video.filename or "camera.webm", api_config.upload) as video_path:
            result = pretrained_predictor.predict(video_path, top_k=top_k)
        return {
            "request_id": request.state.request_id,
            "task": "isolated_recognition",
            "label": result["label"],
            "confidence": result["confidence"],
            "top_k": result["top_k"],
            "processing": {"total_ms": result["processing_ms"]},
            "model": result["model"],
            "development_only": False,
            "warnings": [
                "Realtime uses short camera clips, so output updates with a small delay.",
                "This is isolated-sign recognition over the ViSignLanguage 100-label vocabulary.",
            ],
        }

    @app.post("/predict/continuous/upload")
    async def predict_continuous_upload(request: Request):
        form = await request.form()
        video = form.get("video")
        if video is None or not hasattr(video, "file"):
            raise InferenceError("Missing uploaded video.", details={"field": "video"})
        with safe_upload_file(video.file, video.filename or "upload.mp4", api_config.upload):
            raise PipelineUnavailableError("Raw-video continuous extraction awaits Phase 05 real acceptance.")

    return app


app = create_app()

def _load_guide_samples() -> list[dict[str, str | int | float | None]]:
    metadata_path = Path("data/metadata/visignlanguage_video_split.csv")
    if not metadata_path.exists():
        return []

    samples: list[dict[str, str | int | float | None]] = []
    seen_paths: set[str] = set()
    with metadata_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            video_path = row.get("video_path", "")
            if not video_path or video_path in seen_paths:
                continue
            seen_paths.add(video_path)
            samples.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "label": row.get("label", ""),
                    "label_id": _optional_int(row.get("label_id")),
                    "split": row.get("split", ""),
                    "video_path": video_path,
                    "duration": _optional_float(row.get("duration")),
                    "fps": _optional_float(row.get("fps")),
                    "frame_count": _optional_int(row.get("frame_count")),
                }
            )
    return sorted(samples, key=lambda item: (str(item["label"]), str(item["sample_id"])))

def _guide_sample_by_id(sample_id: str) -> dict[str, str | int | float | None] | None:
    for sample in _load_guide_samples():
        if sample["sample_id"] == sample_id:
            return sample
    return None

def _one_sample_per_label(
    samples: list[dict[str, str | int | float | None]],
) -> list[dict[str, str | int | float | None]]:
    by_label: dict[str, dict[str, str | int | float | None]] = {}
    for sample in samples:
        label = str(sample["label"])
        current = by_label.get(label)
        if current is None or _sample_sort_key(sample) < _sample_sort_key(current):
            by_label[label] = sample
    return [by_label[label] for label in sorted(by_label)]

def _sample_sort_key(sample: dict[str, str | int | float | None]) -> tuple[float, str]:
    duration = sample.get("duration")
    duration_value = float(duration) if isinstance(duration, int | float) else 9999.0
    return duration_value, str(sample["sample_id"])

def _ensure_browser_guide_video(source: Path, sample_id: str) -> Path:
    from fastapi import HTTPException

    cache_dir = Path("outputs/cache/guide_videos")
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{sample_id}.webm"
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-an",
        "-vf",
        "scale=224:224:force_original_aspect_ratio=decrease,pad=224:224:(ow-iw)/2:(oh-ih)/2",
        "-c:v",
        "libvpx",
        "-deadline",
        "realtime",
        "-cpu-used",
        "4",
        "-b:v",
        "700k",
        str(target),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise HTTPException(
            status_code=500,
            detail={"message": "Could not prepare browser-compatible guide video.", "error": str(error)},
        ) from error
    return target

def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)

def _optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))
