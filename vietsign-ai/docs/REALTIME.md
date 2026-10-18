# Realtime Inference Architecture

Phase 11 prepares backend-independent realtime session logic. It does not claim
production webcam recognition quality.

## Session State

`RealtimeSession` maintains:

- rolling frame buffer
- timestamps
- configurable window size
- stride
- minimum frame count
- last committed prediction
- recent prediction history
- duplicate suppression cooldown

Default config lives in `configs/api/default.yaml`:

```yaml
realtime:
  window_size: 32
  stride: 8
  minimum_frames: 16
  smoothing: majority_vote
  confidence_threshold: 0.5
  duplicate_cooldown_windows: 3
```

## Events

The backend event model uses:

```text
prediction_started
prediction_updated
prediction_committed
prediction_cleared
```

This is transport-independent and can later feed WebSocket or native UI
clients.

## Smoothing

The implemented smoothing strategy is majority vote over recent predictions,
with a confidence threshold and duplicate cooldown. This avoids emitting a new
sign every frame.

## WebSocket Status

The current phase implements the session logic and webcam UI capture. A full
binary-frame WebSocket transport is intentionally left for a later hardening
pass so the backend does not fake realtime output.
