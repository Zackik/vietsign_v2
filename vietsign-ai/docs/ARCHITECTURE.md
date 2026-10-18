# VietSignAI Architecture

This repository has progressed past the initial foundation. It now includes the
dataset layer, canonical keypoint preprocessing contract, a keypoint-only Phase
04 Transformer baseline, Phase 05 extractor diagnostics/acceptance gates, and
Phase 06 graph recognition baselines.
It also includes a Phase 07 RGB VideoMAE branch that remains separate from the
keypoint models, plus Phase 08 multimodal fusion baselines.
Continuous gloss recognition is implemented as a Phase 09 software foundation.
Gloss-to-Vietnamese translation is implemented as a Phase 10 software
foundation with synthetic paired text only. Phase 11 adds product API
orchestration and a minimal web console. Phase 12 adds validation tooling for
real-video acceptance, dataset audits, checkpoint manifests, and readiness
reports.

## Long-Term Pipeline

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

## Current Implementation

- `configs/default.yaml` stores data, preprocessing, feature, model, training,
  and device settings.
- `vietsign.config` loads and validates YAML configuration.
- `vietsign.utils.device` chooses CUDA, Apple Silicon MPS, or CPU safely.
- `vietsign.utils.reproducibility` sets Python, NumPy, and PyTorch seeds.
- `vietsign.utils.logging` provides shared structured logging.
- `vietsign.cli` exposes the command surface for future workflows.
- `vietsign.vision` exposes backend-independent keypoint extractors.
- `vietsign.vision.rtmpose` maps RTMLib/RTMPose Wholebody output into
  `coordinates=[T,55,3]` and `valid_mask=[T,55]`.
- `vietsign.models.temporal_transformer` provides the Phase 04 keypoint-only
  recognition baseline.
- `vietsign.models.graph` defines the VietSign55 anatomical skeleton graph.
- `vietsign.models.graph_models` provides ST-GCN and CTR-GCN baselines over the
  same `[B,T,J,C]` keypoint batches.
- `vietsign.models.registry` builds `temporal_transformer`, `stgcn`, and
  `ctrgcn` without model-specific training stacks.
- `vietsign.models.videomae` provides the RGB VideoMAE adapter and embedding
  classifier.
- `vietsign.training.rgb` defines RGB video loading, preprocessing, batching,
  and embedding cache helpers.
- `vietsign.models.multimodal` combines VideoMAE RGB features and Temporal
  Transformer pose features with concat, gated, or cross-attention fusion.
- `vietsign.training.multimodal` aligns RGB and pose samples by `sample_id`.
- `vietsign.continuous` provides gloss vocabularies, CTC decoding, and gloss
  sequence metrics for continuous recognition.
- `vietsign.models.continuous` wraps the Phase 04 Temporal Transformer encoder
  with a CTC head that emits logits shaped `[B,T,V]`.
- `vietsign.training.continuous` loads continuous gloss metadata, builds padded
  variable-length batches, and prepares CTC targets.
- `vietsign.translation` provides gloss/Vietnamese normalization,
  vocabularies, paired dataset loading, deterministic translation metrics, and a
  pretrained adapter boundary.
- `vietsign.models.translation` provides the custom Transformer seq2seq
  baseline for gloss-to-Vietnamese translation.
- `vietsign.inference` provides model management, pipeline registry, upload
  validation, realtime session primitives, and orchestration schemas.
- `vietsign.api` exposes FastAPI routes that call the inference orchestrator
  rather than low-level model internals directly.
- `vietsign.validation` discovers local real videos/datasets, audits metadata
  and signer leakage, records checkpoint maturity, and generates the Phase 12
  validation manifest.

## Future Modules

- `data`: dataset manifests, splits, annotation validation, and loaders.
- `vision`: frame sampling plus hand, body, and face extraction.
- `features`: pose features, RGB features, normalization, and serialization.
- `models`: temporal encoders, fusion modules, recognition heads, translators.
- `training`: experiment setup, checkpointing, metrics, and distributed training.
- `inference`: batch and single-video prediction workflows.
- `evaluation`: recognition and translation metrics.
- `api`: future service interface.

Phase 05C is only accepted once a local real video passes actual extractor
inference with non-zero pose and hand coverage. Real paired VSL/Vietnamese
training data, real checkpoints, and production deployment are not implemented
yet.
