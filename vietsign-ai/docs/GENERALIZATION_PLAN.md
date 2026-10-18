# VietSignAI Generalization Plan

This plan tracks the move from a dataset-specific demo toward recognition that
works across signers, cameras, views, lighting, and backgrounds.

## Target

```text
multi-view / multi-signer video
→ RGB + hand/body keypoints
→ domain-aware training and evaluation
→ realtime sliding-window recognition
→ Vietnamese text output
```

## Metadata Contract

Every real isolated-sign dataset should normalize to:

```text
sample_id, video_path, label, label_id, signer_id, split,
dataset, camera, view_id, domain_id, duration, fps,
frame_count, width, height
```

Required for multi-view datasets:

```text
sample_id, video_path, label, label_id, signer_id, split, view_id
```

`camera` should identify the physical camera when available. `view_id` should
identify viewpoint such as `front`, `left`, `right`, or the dataset's native
view name. `domain_id` should identify broader capture conditions such as
studio, webcam, phone, low-light, or dataset source.

## Evaluation Gates

Do not call a model robust until it reports:

- overall top-1/top-5 accuracy
- per-class accuracy
- unseen-signer accuracy
- per-camera accuracy
- per-view accuracy
- worst-domain accuracy
- webcam/realtime holdout accuracy

## Near-Term Steps

1. Convert Multi-VSL/VSL400 manifests into the normalized metadata contract.
2. Train/fine-tune the RGB VideoMAE baseline on the larger multi-view data.
3. Evaluate by signer, camera, view, and domain.
4. Add sliding-window smoothing to the realtime web path.
5. Add a keypoint branch and fusion only after the RGB baseline is measured.
