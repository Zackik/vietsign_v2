# VietSignAI Validation Report

## Status Legend

- IMPLEMENTED: code exists.
- SOFTWARE-VALIDATED: tested with synthetic fixtures or static checks.
- REAL-DATA-VALIDATED: verified with real local VSL data.
- NOT YET VALIDATED: blocked by missing data or acceptance evidence.

## Subsystems

| Subsystem | Status | Evidence |
| --- | --- | --- |
| Extractor | NOT YET VALIDATED | REAL VIDEO REQUIRED; doctor ready=True |
| ISLR | REAL-DATA-VALIDATED (LIMITED) | VOYA native `[T,1605]` direct-feature baselines trained on 4-class Sprint 03 and 10-class Sprint 04 subsets; NOT signer-independent; canonical VietSign55 mapping unverified |
| RGB | SOFTWARE-VALIDATED | Phase 07 tests pass; no real subset RGB benchmark |
| Multimodal | SOFTWARE-VALIDATED | Phase 08 tests pass; no real aligned subset |
| CSLR | SOFTWARE-VALIDATED | Phase 09 synthetic CTC only |
| Translation | SOFTWARE-VALIDATED | Phase 10 synthetic paired text only |
| API | SOFTWARE-VALIDATED | Phase 11 API tests pass |
| Realtime | SOFTWARE-VALIDATED | Rolling-window/session tests pass |

## Real Video Acceptance

- Status: REAL VIDEO REQUIRED
- Videos found: 0
- RTMPose doctor ready: True

## Real Dataset Availability

- Any real dataset available: True, limited to a controlled VOYA_VSL native-feature subset.
- Discovery: `{"vsl400": {"root": "data/raw/vsl400", "exists": false, "video_count": 0, "metadata_files": [], "available": false}, "vsl1000": {"root": "data/raw/vsl1000", "exists": false, "video_count": 0, "metadata_files": [], "available": false}, "custom": {"root": "data/raw/custom", "exists": false, "video_count": 0, "metadata_files": [], "available": false}}`

## Blockers

- No real acceptance video found under data/acceptance.
- No real local VSL dataset videos found under data/raw.
- VOYA native feature data is available locally, but signer metadata is unavailable.
- VOYA canonical feature ordering remains unverified, so VietSign55 conversion is blocked.
- Sprint 04 static baselines are strong, so native VOYA class separability may include static/file-level distribution signals.
- No real continuous gloss-annotated VSL dataset found.
- No paired gloss-Vietnamese real translation corpus found.
