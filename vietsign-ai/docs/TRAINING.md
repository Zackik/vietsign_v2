# VietSignAI Training

Phase 04 trains an isolated sign classifier from cached keypoint features.
Preprocessing and training are separate: training never recomputes keypoints.

## Train

```bash
python -m vietsign.cli train --config configs/training/baseline.yaml
```

Resume:

```bash
python -m vietsign.cli train --config configs/training/baseline.yaml --resume outputs/runs/<run>/last.pt
```

## Evaluate

```bash
python -m vietsign.cli evaluate \
  --config configs/training/baseline.yaml \
  --checkpoint outputs/runs/<run>/best.pt
```

Evaluation reports Top-1, Top-5, macro F1, weighted F1, per-class accuracy, and
confusion matrix. Top-5 adapts when fewer than five classes exist.

## Predict

```bash
python -m vietsign.cli predict \
  --config configs/training/baseline.yaml \
  --features data/cache/keypoints/<dataset>/<sample_id>.npz \
  --checkpoint outputs/runs/<run>/best.pt
```

Prediction operates directly on cached tensors and returns softmax confidence.
A fully empty sequence is rejected instead of producing a misleading normal
prediction.

Phase 05 extractor changes do not require model changes: any backend that writes
the canonical cache contract can be consumed by this training pipeline.

## Outputs

Each training run writes:

```text
outputs/runs/<run_name>-<timestamp>/
  config.json
  config.yaml
  history.csv
  metrics.json
  best.pt
  last.pt
  evaluation.json
  confusion_matrix.png
```

Checkpoints store state dictionaries, optimizer/scheduler state, epoch, config,
class mapping, best metric, and model metadata.

Synthetic smoke-test metrics are for software verification only and are not VSL
benchmark results.
