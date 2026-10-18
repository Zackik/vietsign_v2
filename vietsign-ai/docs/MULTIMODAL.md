# VietSignAI Multimodal Fusion

Phase 08 adds RGB + keypoint fusion for isolated sign classification. It does
not implement continuous recognition, CTC, translation, FastAPI, or a web UI.

## Batch Contract

Multimodal batches contain:

```text
pixel_values:  [B,T_rgb,C,H,W]
coordinates:   [B,T_pose,J,C_pose]
valid_mask:    [B,T_pose,J]
temporal_mask: [B,T_pose]
labels:        [B]
sample_ids:    list[str]
signer_ids:    list[str | None]
modality_mask: {"rgb": [B], "pose": [B]}
```

`T_rgb` and `T_pose` do not need to match.

## Alignment

RGB and pose are aligned by `sample_id`, never by list position. The dataset
validates:

- matching `sample_id`
- matching label when metadata is available
- compatible source video when keypoint metadata records it

Missing pose or failed RGB decode is rejected by default. RGB is not allowed to
silently hide an empty extractor result.

## Encoders

RGB encoder:

```text
VideoMAE pixel_values -> rgb_embedding [B,D_rgb]
VideoMAE tokens       -> rgb_tokens [B,N_rgb,D_rgb]
```

Pose encoder:

```text
Temporal Transformer coordinates -> pose_embedding [B,D_pose]
Temporal Transformer tokens      -> pose_tokens [B,T_pose,D_pose]
```

Both standalone classifiers retain their original classification behavior.

## Fusion Modes

Late concat:

```text
rgb_embedding -> Linear(shared_dim)
pose_embedding -> Linear(shared_dim)
[rgb ; pose] -> LayerNorm -> MLP -> classifier
```

Gated fusion:

```text
g = sigmoid(W[rgb ; pose])
fused = g * rgb + (1-g) * pose
```

Cross-attention:

```text
pose tokens query RGB tokens
attended pose tokens + pose tokens
temporal-mask-aware pooling
classifier
```

## Freezing

Initial recommended mode:

```yaml
model:
  freeze_rgb: true
  freeze_pose: false
```

Differential learning rates are available through:

```yaml
optimizer:
  rgb_lr: 0.00001
  pose_lr: 0.0001
  fusion_lr: 0.0003
```

## Commands

Train late fusion:

```bash
python -m vietsign.cli train --config configs/training/multimodal_late_fusion.yaml
```

Train cross-attention:

```bash
python -m vietsign.cli train --config configs/training/multimodal_cross_attention.yaml
```

Predict with explicit keypoints:

```bash
python -m vietsign.cli predict-multimodal \
  --video example.mp4 \
  --keypoints data/cache/keypoints/<dataset>/<sample_id>.npz \
  --config configs/training/multimodal_late_fusion.yaml \
  --checkpoint outputs/runs/<run>/best.pt
```

## Synthetic Smoke Warning

Synthetic multimodal benchmarks are labeled:

```text
SYNTHETIC SOFTWARE VALIDATION - NON-VSL / NOT A VSL BENCHMARK
```

They verify software plumbing only and must not be used to claim model quality
or rank architectures.
