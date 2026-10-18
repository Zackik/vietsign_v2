# VietSignAI RGB Models

Phase 07 adds a separate RGB-video recognition branch for isolated sign
classification. It does not modify the canonical keypoint contract.

## RGB Contract

Project-level RGB batches use:

```text
pixel_values: Tensor[B,T,C,H,W]
labels: Tensor[B]
temporal_mask: Tensor[B,T]
sample_ids: list[str]
signer_ids: list[str | None]
```

`T` is the configured clip length, `C=3`, and frames are RGB after resize, crop,
and normalization.

## VideoMAE

The primary RGB baseline is `videomae`.

Default production-style config:

```yaml
model:
  type: videomae
  pretrained_model: MCG-NJU/videomae-base
  pretrained: true
  freeze_backbone: true
```

If pretrained weights are unavailable, the loader fails with an actionable
message. Tests and synthetic smoke runs explicitly use `pretrained: false` with
a tiny local VideoMAE-compatible backbone. That mode is for software validation
only and must not be described as pretrained.

## Frozen vs Fine-Tuned

Frozen backbone is the default because early development may happen on Apple M1
hardware with limited memory. In frozen mode, only the classification head is
trainable.

Fine-tuning can be enabled by setting:

```yaml
model:
  freeze_backbone: false
```

Gradient accumulation is supported:

```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 4
```

## Preprocessing

The RGB pipeline supports:

- uniform deterministic sampling for evaluation
- random contiguous temporal sampling for training
- resize
- center crop
- optional random crop
- ImageNet-style normalization by default

Horizontal flip is disabled by default because sign-language left/right
semantics can change meaning.

## RGB Embedding Cache

Offline embedding extraction writes safe NPZ + JSON files:

```text
data/cache/rgb/videomae/<dataset>/<sample_id>.npz
data/cache/rgb/videomae/<dataset>/<sample_id>.json
```

The cache metadata includes model identity, preprocessing identity, frame
sampling config, schema version, and fingerprint. Changing model, frame count,
sampling, resize, crop, normalization, or freeze/backbone config changes the
fingerprint.

## Commands

Train RGB VideoMAE:

```bash
python -m vietsign.cli train --config configs/training/videomae.yaml
```

Precompute RGB embeddings:

```bash
python -m vietsign.cli preprocess rgb --config configs/preprocessing/videomae.yaml
```

Predict one video:

```bash
python -m vietsign.cli predict-video \
  --video example.mp4 \
  --config configs/training/videomae.yaml \
  --checkpoint outputs/runs/<run>/best.pt
```

## Synthetic Smoke

Synthetic/generated-video runs are labeled:

```text
SYNTHETIC SOFTWARE VALIDATION - NON-VSL / NOT A VSL BENCHMARK
```

They verify video loading, frame sampling, RGB preprocessing, VideoMAE forward,
checkpointing, evaluation, prediction, and benchmark report generation. They do
not measure VSL recognition quality.
