# VietSignAI API

Phase 11 adds a product API shell around the existing recognition and
translation foundations. It does not add new research models.

## Startup

Install API dependencies:

```bash
python -m pip install -e ".[api]"
```

Run locally:

```bash
python -m vietsign.cli serve --config configs/api/default.yaml
```

The default host is `127.0.0.1`. Do not expose the development server publicly.

## Endpoints

```text
GET  /health
GET  /models
GET  /pipelines
POST /predict/isolated
POST /predict/continuous
POST /predict/continuous/upload
POST /translate
POST /recognize-and-translate
```

`/predict/continuous` and `/recognize-and-translate` currently support
preprocessed feature caches for software validation. Raw-video continuous
prediction intentionally reports unavailable until Phase 05 real-video
acceptance passes.

## Example Requests

Translate glosses:

```bash
curl -X POST http://127.0.0.1:8000/translate \
  -H "content-type: application/json" \
  -d '{"gloss":"TOI NGAY-MAI BENH-VIEN DI"}'
```

Recognize continuous glosses from a synthetic feature cache:

```bash
curl -X POST http://127.0.0.1:8000/predict/continuous \
  -H "content-type: application/json" \
  -d '{"features_path":"data/cache/keypoints/continuous_synthetic/cs_val_002.npz"}'
```

Run the synthetic CSLR -> translation composition:

```bash
curl -X POST http://127.0.0.1:8000/recognize-and-translate \
  -H "content-type: application/json" \
  -d '{"features_path":"data/cache/keypoints/continuous_synthetic/cs_val_002.npz"}'
```

## Readiness

`/health` distinguishes service health from real VSL readiness:

- `status`: whether the API process can serve requests.
- `real_vsl_ready`: false until real extractors and real checkpoints are
  accepted.
- `extractor_status`: reports Phase 05 real-video acceptance state.

`/pipelines` reports each pipeline and why it is ready or unavailable.
Development checkpoints are marked with `development_only: true`.

## Upload Safety

Uploaded videos use safe temporary filenames under the configured temp
directory and are deleted after the request. The API validates:

- extension allowlist
- maximum upload size
- decode success
- frame count
- duration limit

The API does not persist uploaded videos by default.

## Error Schema

Normal failures return structured responses:

```json
{
  "request_id": "...",
  "error": "unsupported_format",
  "message": "Unsupported video extension: .txt",
  "details": {}
}
```

Stack traces are not exposed in normal API responses.
