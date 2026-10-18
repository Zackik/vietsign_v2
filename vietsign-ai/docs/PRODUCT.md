# Product Shell

Phase 11 turns VietSignAI into a usable local application architecture:

```text
API + inference orchestration + minimal web console
```

It does not add new recognition or translation research models.

## Architecture

```text
FastAPI routes
-> VietSignPipeline orchestrator
-> ModelManager
-> existing Phase 04/09/10 prediction helpers
```

Routes do not call low-level tensors or model internals directly.

## Web App

The minimal web app lives in `apps/web` and includes:

- Home / pipeline overview
- Video Recognition
- Realtime / Webcam
- Model Status

Build:

```bash
cd apps/web
npm install
npm run build
```

The webcam page initializes browser camera capture but does not fabricate
realtime predictions.

## Development Labels

Synthetic CSLR and translation checkpoints are marked:

```text
development_only: true
```

The API and UI expose this status so synthetic software-validation models are
not presented as production VSL systems.

## Privacy

Uploaded videos may contain identifiable people. The default API policy is:

- temporary processing only
- unique server-side filenames
- automatic cleanup after request
- no persistent video storage
- no logging of uploaded video content

## Current Limitations

- Phase 05 real-video acceptance remains deferred.
- No real VSL checkpoint is bundled.
- Synthetic API demos validate software plumbing only.
- No authentication, database, cloud deployment, or persistent user storage is
  implemented.
