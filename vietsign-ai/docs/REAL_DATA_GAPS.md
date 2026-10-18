# Real Data Gaps

## ISLR

Real labeled isolated VSL raw clips are not available in the current repository
unless a local ignored dataset is added under `data/raw`.

VOYA_VSL native feature data is now available as a controlled partial download
under `data/raw/voya_vsl_probe`. This supports a direct `[T,1605]` source-layout
baseline, not canonical VietSign55 training.

Current raw-video discovery:

```json
{
  "vsl400": {
    "root": "data/raw/vsl400",
    "exists": false,
    "video_count": 0,
    "metadata_files": [],
    "available": false
  },
  "vsl1000": {
    "root": "data/raw/vsl1000",
    "exists": false,
    "video_count": 0,
    "metadata_files": [],
    "available": false
  },
  "custom": {
    "root": "data/raw/custom",
    "exists": false,
    "video_count": 0,
    "metadata_files": [],
    "available": false
  }
}
```

## CSLR

No real continuous VSL videos with gloss sequence annotations were discovered.
Do not train Phase 09 on isolated clips and call it continuous recognition.

## SLT

No real paired gloss-sequence/Vietnamese-sentence corpus was discovered.
Phase 10 remains infrastructure-only until such data exists.

## VOYA Remaining Gaps

- No signer metadata, so evaluation is sample-stratified and
  `signer_independent=false`.
- No authoritative 1605 feature ordering, so canonical VietSign55 conversion
  remains blocked.
- Only 4 class files were downloaded for Sprint 03; this is not a full VOYA
  benchmark.
- Sprint 04 expands to 10 class files, but still uses a capped partial dataset
  and remains sample-stratified rather than signer-independent.
- Static and temporal-shuffled baselines remain strong in Sprint 04, so future
  work should audit class/file-level artifacts before treating accuracy as
  signer-generalized recognition evidence.
