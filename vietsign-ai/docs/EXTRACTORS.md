# VietSignAI Extractors

Phase 05 prepares VietSignAI for real VSL preprocessing. The canonical output
remains backend-independent:

```text
coordinates = [T,55,3]
valid_mask = [T,55]
confidence = [T,55]
source_mask = [T,55]
```

## MediaPipe Tasks

The installed MediaPipe package exposes the Tasks API. Real inference requires
task model assets that are not committed to this repository:

```text
models/mediapipe/hand_landmarker.task
models/mediapipe/pose_landmarker.task
```

Configure them in:

```text
configs/preprocessing/mediapipe_tasks.yaml
```

Install them with:

```bash
python -m vietsign.cli extractor assets install mediapipe
```

The installer downloads from official Google MediaPipe model storage:

- `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
- `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`

If assets are missing, preprocessing fails with an actionable error. The system
must not silently produce all-missing keypoints and call that success.

## RTMLib / RTMPose

RTMLib/RTMPose is optional and does not require installing the full MMPose stack
for this backend:

```bash
python -m pip install -e ".[rtmpose]"
python -m vietsign.cli extractor assets install rtmpose --mode balanced
```

Configure model files in:

```text
configs/preprocessing/rtmpose.yaml
```

Phase 05C uses `rtmlib.Wholebody(..., to_openpose=True)`. RTMLib returns up to
134 OpenPose-style landmarks per detected person:

```text
body      = 0:18
face      = 24:92
left_hand = 92:113
right_hand = 113:134
```

VietSignAI maps those into the canonical model contract:

```text
coordinates = [T,55,3]
valid_mask  = [T,55]
J = 55 = 21 left hand + 21 right hand + 13 upper-body pose
```

RTMPose is a 2D backend here, so canonical `z` is `0.0` and frame metadata
records `z_source=unavailable_2d_backend`.

The backend is isolated behind the same `KeypointExtractor` contract. Downstream
training still consumes cached canonical tensors and does not know which backend
created them.

## Diagnostics

```bash
python -m vietsign.cli extractor doctor --config configs/preprocessing/mediapipe_tasks.yaml
python -m vietsign.cli extractor test --video path/to/video.mp4 --config configs/preprocessing/mediapipe_tasks.yaml --visualize
python -m vietsign.cli extractor compare --video path/to/video.mp4 --configs configs/preprocessing/mediapipe_tasks.yaml configs/preprocessing/rtmpose.yaml
python -m vietsign.cli extractor acceptance-test --video path/to/video.mp4 --config configs/preprocessing/rtmpose.yaml
```

`doctor` only reports ready when an actual inference backend and required model
assets are available.

Doctor verifies more than file existence: it imports the Tasks API and attempts
to initialize both HandLandmarker and PoseLandmarker in an isolated subprocess.
If initialization crashes or fails, readiness remains false.

The Phase 05C acceptance test only passes when a real video produces:

```text
actual landmark detector -> [T,55,3] -> real non-zero valid_mask -> Phase 04 Transformer
```

More precisely, it requires `valid_mask.sum()>0`, `pose_coverage>0`, and at
least one of `left_hand_coverage` or `right_hand_coverage` greater than zero.
All-false `valid_mask` outputs are a hard failure and should be fixed in the
extractor before moving to later phases.

## Quality Metrics

Phase 05 records:

- left-hand coverage
- right-hand coverage
- pose coverage
- overall valid-joint ratio
- empty-frame ratio
- empty-sequence flag
- interpolation ratio
- large coordinate jump count

Low quality samples are reported, not automatically deleted.

## Real VSL Workflow

1. Place raw data under `data/raw/vsl400/`.
2. Create or verify normalized metadata such as `data/metadata/vsl400_subset.csv`.
3. Start with a small subset: 3-5 classes, multiple signers if available.
4. Run extractor diagnostics.
5. Preprocess keypoints into `data/cache/keypoints/`.
6. Review quality reports and visualizations under `outputs/quality_review/`.
7. Train the Phase 04 Transformer only after rejecting empty/invalid caches.

If VSL-400 is not locally available, dataset-specific parsing remains blocked;
the normalized CSV path remains the supported onboarding path.

## Apple Silicon

MediaPipe Tasks should run on local CPU/Apple Silicon when model assets are
available. RTMPose dependencies may be heavier and are not required for the base
installation.
