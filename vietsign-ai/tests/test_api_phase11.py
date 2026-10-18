from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from vietsign.api.app import create_app
from vietsign.inference.config import ApiConfig, InferenceModelConfig, RealtimeConfig, UploadLimits
from vietsign.inference.realtime import RealtimeSession

CSLR_CHECKPOINT = Path("outputs/runs/continuous_ctc_transformer-20261005-220931/best.pt")
TRANSLATION_CHECKPOINT = Path("outputs/runs/gloss_translation_transformer-20261005-222257/best.pt")
FEATURES = Path("data/cache/keypoints/continuous_synthetic/cs_val_002.npz")


def _client() -> TestClient:
    config = ApiConfig(
        upload=UploadLimits(max_upload_mb=1, max_duration_seconds=5),
        realtime=RealtimeConfig(window_size=4, stride=2, minimum_frames=2),
        models=[
            InferenceModelConfig(
                name="synthetic_cslr",
                task="continuous_recognition",
                modality="keypoints",
                config_path=Path("configs/training/continuous_ctc.yaml"),
                checkpoint_path=CSLR_CHECKPOINT,
                development_only=True,
            ),
            InferenceModelConfig(
                name="synthetic_translator",
                task="gloss_translation",
                modality="text",
                config_path=Path("configs/training/gloss_translation.yaml"),
                checkpoint_path=TRANSLATION_CHECKPOINT,
                development_only=True,
            ),
        ],
    )
    return TestClient(create_app(config))


def test_health_models_and_pipelines() -> None:
    client = _client()

    health = client.get("/health").json()
    models = client.get("/models").json()
    pipelines = client.get("/pipelines").json()

    assert health["real_vsl_ready"] is False
    assert "continuous_translate" in health["available_pipelines"]
    assert any(model["development_only"] for model in models["models"])
    assert any(item["name"] == "continuous_translate" and item["ready"] for item in pipelines["pipelines"])


def test_translation_endpoint() -> None:
    client = _client()
    response = client.post("/translate", json={"gloss": "TOI NGAY-MAI BENH-VIEN DI"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["translation"] == "Ngày mai tôi sẽ đi bệnh viện."
    assert payload["development_only"] is True


def test_continuous_and_recognize_translate_endpoints() -> None:
    client = _client()
    continuous = client.post("/predict/continuous", json={"features_path": str(FEATURES)})
    combined = client.post("/recognize-and-translate", json={"features_path": str(FEATURES)})

    assert continuous.status_code == 200
    assert continuous.json()["glosses"]
    assert combined.status_code == 200
    assert combined.json()["glosses"]
    assert "translation" in combined.json()


def test_invalid_upload_error_schema() -> None:
    client = _client()
    response = client.post(
        "/predict/isolated",
        files={"video": ("bad.txt", b"not a video", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["error"] == "unsupported_format"


def test_realtime_window_smoothing_and_duplicate_suppression() -> None:
    session = RealtimeSession(window_size=4, stride=2, minimum_frames=2, confidence_threshold=0.5, duplicate_cooldown_windows=1)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    assert session.add_frame(frame, timestamp=0.0) == []
    events = session.add_frame(frame, timestamp=0.1)
    assert events[0].type == "prediction_started"
    committed = session.update_prediction("TOI", 0.8)
    assert [event.type for event in committed] == ["prediction_updated", "prediction_committed"]
    duplicate = session.update_prediction("TOI", 0.9)
    assert [event.type for event in duplicate] == ["prediction_updated"]
    assert session.clear().type == "prediction_cleared"
