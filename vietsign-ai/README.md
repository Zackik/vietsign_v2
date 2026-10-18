# VietSignAI

VietSignAI is a long-term Vietnamese Sign Language recognition and translation project.
This repository currently implements the project foundation, dataset contracts,
canonical keypoint tensors, a keypoint-only Phase 04 Transformer baseline, the
Phase 05 extractor readiness/quality gate, Phase 06 graph-model baselines, and
Phase 07 RGB VideoMAE baseline, Phase 08 multimodal RGB+keypoint fusion,
Phase 09 continuous gloss recognition with CTC, and Phase 10 gloss-to-Vietnamese
translation. Phase 11 adds a local product API, realtime session foundation, and
minimal web console. Phase 12 adds real-data validation/audit tooling and a
machine-readable validation manifest.

The target pipeline is:

```text
Video
-> Hand / Body / Face Extraction
-> Pose Features + RGB Features
-> Temporal Modeling
-> Multimodal Fusion
-> VSL Recognition
-> Gloss
-> Vietnamese Translation
```

Real-data Vietnamese translation quality and HaMeR are not implemented yet.
Phase 05C is only accepted after a real video passes actual RTMLib/RTMPose
extraction with non-zero pose and hand landmarks.

## Installation

Use Python 3.11 or newer.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional dependency groups are prepared for later phases:

```bash
python -m pip install -e ".[api]"
python -m pip install -e ".[transformers]"
python -m pip install -e ".[pose]"
python -m pip install -e ".[rtmpose]"
python -m pip install -e ".[tracking]"
```

Do not install heavy optional groups until the relevant phase needs them.

## Basic Usage

Load the default configuration:

```bash
python -m vietsign.cli config configs/default.yaml
```

Check device detection:

```bash
python -m vietsign.cli device
```

Validate, inspect, or summarize a dataset config:

```bash
python -m vietsign.cli dataset inspect --config configs/data/vsl400.yaml
python -m vietsign.cli dataset validate --config configs/data/vsl400.yaml
python -m vietsign.cli dataset stats --config configs/data/vsl400.yaml
```

Extract and cache visual keypoints:

```bash
python -m vietsign.cli preprocess keypoints --config configs/preprocessing/mediapipe.yaml
python -m vietsign.cli preprocess inspect-video path/to/video.mp4 --config configs/preprocessing/mediapipe.yaml
python -m vietsign.cli visualize keypoints --features data/cache/keypoints/<dataset>/<sample_id>.npz
```

The Phase 3 keypoint tensor contract is documented in
[docs/KEYPOINTS.md](docs/KEYPOINTS.md). By default:

```text
coordinates = [T, 55, 3]
valid_mask = [T, 55]
```

where `55 = 21 left-hand joints + 21 right-hand joints + 13 upper-body pose joints`.

Train, evaluate, and predict with the keypoint-only baseline:

```bash
python -m vietsign.cli train --config configs/training/baseline.yaml
python -m vietsign.cli evaluate --config configs/training/baseline.yaml --checkpoint outputs/runs/<run>/best.pt
python -m vietsign.cli predict --config configs/training/baseline.yaml --features data/cache/keypoints/<dataset>/<sample_id>.npz --checkpoint outputs/runs/<run>/best.pt
```

See [docs/MODELS.md](docs/MODELS.md) and [docs/TRAINING.md](docs/TRAINING.md).

Graph model baselines use the same cached keypoints and training pipeline:

```bash
python -m vietsign.cli train --config configs/training/stgcn.yaml
python -m vietsign.cli train --config configs/training/ctrgcn.yaml
python -m vietsign.cli model info --config configs/training/stgcn.yaml
python -m vietsign.cli benchmark models --synthetic-smoke --configs configs/training/baseline.yaml configs/training/stgcn.yaml configs/training/ctrgcn.yaml configs/training/videomae.yaml
```

Synthetic benchmark outputs are explicitly labeled `NON-VSL / NOT A VSL
BENCHMARK`; they validate software plumbing only.

RGB VideoMAE branch:

```bash
python -m vietsign.cli train --config configs/training/videomae.yaml
python -m vietsign.cli preprocess rgb --config configs/preprocessing/videomae.yaml
python -m vietsign.cli predict-video --video path/to/video.mp4 --config configs/training/videomae.yaml --checkpoint outputs/runs/<run>/best.pt
```

Multimodal RGB + keypoint fusion:

```bash
python -m vietsign.cli train --config configs/training/multimodal_late_fusion.yaml
python -m vietsign.cli train --config configs/training/multimodal_cross_attention.yaml
python -m vietsign.cli predict-multimodal --video path/to/video.mp4 --keypoints data/cache/keypoints/<dataset>/<sample_id>.npz --config configs/training/multimodal_late_fusion.yaml --checkpoint outputs/runs/<run>/best.pt
```

Continuous gloss recognition with CTC:

```bash
python -m vietsign.cli train --config configs/training/continuous_ctc.yaml
python -m vietsign.cli evaluate-continuous --config configs/training/continuous_ctc.yaml --checkpoint outputs/runs/<run>/best.pt
python -m vietsign.cli predict-continuous --features data/cache/keypoints/<dataset>/<sample_id>.npz --checkpoint outputs/runs/<run>/best.pt
```

The Phase 09 continuous pipeline predicts gloss token sequences, not natural
Vietnamese sentences. It uses CTC with blank index `0`, greedy decoding, and
gloss error rate (GER). See
[docs/CONTINUOUS_RECOGNITION.md](docs/CONTINUOUS_RECOGNITION.md).

Gloss-to-Vietnamese translation:

```bash
python -m vietsign.cli train --config configs/training/gloss_translation.yaml
python -m vietsign.cli evaluate --config configs/training/gloss_translation.yaml --checkpoint outputs/runs/<run>/best.pt
python -m vietsign.cli translate --gloss "TOI NGAY-MAI BENH-VIEN DI" --config configs/training/gloss_translation.yaml --checkpoint outputs/runs/<run>/best.pt
python -m vietsign.cli recognize-and-translate --features data/cache/keypoints/<dataset>/<sample_id>.npz --cslr-checkpoint outputs/runs/<cslr-run>/best.pt --translation-checkpoint outputs/runs/<translation-run>/best.pt
```

Phase 10 translation consumes gloss sequences only. It is separate from
MediaPipe, RTMPose, VideoMAE, and CTC internals. Synthetic translation metrics
are labeled software validation and are not real VSL performance. See
[docs/TRANSLATION.md](docs/TRANSLATION.md).

Product API and web console:

```bash
python -m pip install -e ".[api]"
python -m vietsign.cli serve --config configs/api/default.yaml

cd apps/web
npm install
npm run build
```

See [docs/API.md](docs/API.md), [docs/REALTIME.md](docs/REALTIME.md), and
[docs/PRODUCT.md](docs/PRODUCT.md). Development API responses expose
`development_only: true` for synthetic checkpoints.

Real-data validation and readiness reporting:

```bash
python -m vietsign.cli validation phase12 --output outputs/validation/phase12_manifest.json
```

This generates `docs/VALIDATION_REPORT.md`, `docs/REAL_DATA_GAPS.md`, and
`outputs/validation/checkpoint_manifest.json`. If no real acceptance video or
dataset exists locally, the report says so directly.

Check extractor readiness for real preprocessing:

```bash
python -m vietsign.cli extractor assets install mediapipe
python -m vietsign.cli extractor doctor --config configs/preprocessing/mediapipe_tasks.yaml
python -m vietsign.cli extractor test --video path/to/video.mp4 --config configs/preprocessing/mediapipe_tasks.yaml --visualize
```

RTMLib/RTMPose path for Phase 05C:

```bash
python -m pip install -e ".[rtmpose]"
python -m vietsign.cli extractor assets install rtmpose --mode balanced
python -m vietsign.cli extractor doctor --config configs/preprocessing/rtmpose.yaml
python -m vietsign.cli extractor acceptance-test --video path/to/real_video.mp4 --config configs/preprocessing/rtmpose.yaml
```

The acceptance gate requires `coordinates=[T,55,3]`, `valid_mask=[T,55]`,
`valid_mask.sum()>0`, `pose_coverage>0`, and at least one non-zero hand
coverage before running the Phase 04 Transformer compatibility smoke test.

See [docs/EXTRACTORS.md](docs/EXTRACTORS.md) for MediaPipe Tasks model assets,
RTMLib/RTMPose setup, and real-data quality review.

Future command shape:

```bash
vietsign preprocess
vietsign train
vietsign evaluate
vietsign predict
```

Those workflow commands currently report that they are not implemented.

## Project Structure

```text
src/vietsign/
  api/            Future API adapters.
  data/           Future dataset abstractions.
  evaluation/     Future metrics and evaluation workflows.
  features/       Future pose/RGB feature engineering.
  inference/      Future prediction workflows.
  models/         Future model definitions.
  training/       Future training orchestration.
  utils/          Shared config, logging, device, and reproducibility utilities.
  vision/         Future visual extraction components.
configs/          YAML configuration files.
docs/             Architecture and roadmap notes.
scripts/          Future automation scripts.
tests/            Foundation tests.
data/             Local datasets, ignored by default.
outputs/          Local experiment outputs, ignored by default.
```

Dataset layout:

```text
data/
  raw/
    vsl400/
    vsl1000/
  metadata/
  processed/
  cache/
```

See [docs/DATASETS.md](docs/DATASETS.md) for the normalized schema, custom CSV
format, validation behavior, and signer-independent evaluation notes.

## Development Roadmap

1. Foundation: project packaging, config, logging, device handling, seeds, CLI, tests.
2. Data contracts: dataset manifests, split definitions, annotation schema, validation.
3. Visual extraction: canonical hand/body/face landmark contract and real extractor gates.
4. Temporal modeling: keypoint-only Transformer baseline.
5. Real extractor validation: RTMLib/RTMPose acceptance on local real videos.
6. Graph recognition: ST-GCN and CTR-GCN baselines over VietSign55.
7. RGB recognition: VideoMAE branch and RGB embedding cache.
8. Multimodal fusion: combine pose features and RGB embeddings.
9. Continuous gloss recognition: CTC over variable-length sign sequences.
10. Gloss-to-Vietnamese translation: translate gloss sequences into natural Vietnamese.
11. Product API and realtime shell: local FastAPI service and minimal web console.
12. Real data hardening: validation manifest, dataset audits, checkpoint maturity, and readiness reporting.
13. Real-data experiments: accepted real extraction, real checkpoints, and human evaluation.

## Testing

```bash
python -m pytest
```
