# VOYA Direct-Feature Baseline

This document records the first real-data recognition baseline in VietSignAI
using VOYA_VSL native features.

## Why Native Features

VOYA_VSL stores real extracted sign-language features as `sequences` with shape
`[N,60,1605]`. A single sample therefore has:

```text
features = [T,1605]
temporal_mask = [T]
label
sample_id
source_file
source_index
provenance
```

These are source-layout features. They are not canonical VietSign55 keypoints.

## Canonical Mapping Status

Canonical VOYA to VietSign55 conversion remains blocked. The 1605 dimension is
compatible with several mathematical landmark hypotheses, but the authoritative
feature ordering is still unavailable.

Do not use ST-GCN or CTR-GCN on this representation. The anatomical graph
topology is unknown.

## Data Organization

The inspected files show that `class_XXXX.npz` is one semantic class file, not a
multi-label shard. For the controlled Sprint 03 subset:

- `class_0001.npz`: label `1`, `1001` samples.
- `class_0002.npz`: label `2`, `1001` samples.
- `class_0003.npz`: label `3`, `1001` samples.
- `class_0004.npz`: label `4`, `1001` samples.

`labels.json` contains `4363` contiguous keys from `class_0001` through
`class_4363`; values are Vietnamese gloss/word labels. Some values are
duplicates and must not be blindly treated as unique display text.

## Split Limitation

No signer metadata has been discovered. The current experiment therefore uses:

```text
split_type = sample_stratified
signer_independent = false
```

The result is a real VOYA_VSL source-feature experiment, but it is not a
signer-independent benchmark.

Exact feature-array hashes are checked so exact duplicate sequences do not cross
train/validation/test splits. This is not signer-leakage protection.

## Normalization

The baseline supports:

- `none`
- `per_feature_zscore`

Sprint 03 uses train-only per-feature z-score statistics. Validation and test
samples are never used to fit normalization.

## Model

The model is an input adapter plus existing Temporal Transformer machinery:

```text
[B,T,1605]
→ LayerNorm
→ Linear(1605,D)
→ GELU
→ Dropout
→ Temporal Transformer encoder
→ masked temporal mean pooling
→ classifier
```

The configured Sprint 03 model has `156494` trainable parameters.

## Experiment Protocol

Configuration:

- `configs/models/voya_direct_transformer.yaml`
- `configs/training/voya_direct_baseline.yaml`

Outputs:

- `outputs/validation/voya_direct_stats.json`
- `outputs/real_experiments/voya_direct/voya_direct_baseline/config.yaml`
- `outputs/real_experiments/voya_direct/voya_direct_baseline/dataset_stats.json`
- `outputs/real_experiments/voya_direct/voya_direct_baseline/split_manifest.json`
- `outputs/real_experiments/voya_direct/voya_direct_baseline/normalization.json`
- `outputs/real_experiments/voya_direct/voya_direct_baseline/training_history.csv`
- `outputs/real_experiments/voya_direct/voya_direct_baseline/best.pt`
- `outputs/real_experiments/voya_direct/voya_direct_baseline/evaluation.json`
- `outputs/real_experiments/voya_direct/voya_direct_baseline/confusion_matrix.png`
- `outputs/real_experiments/voya_direct/voya_direct_baseline/per_class_metrics.csv`
- `outputs/real_experiments/voya_direct/voya_direct_baseline/top_confusions.csv`

Checkpoint maturity:

```text
experimental_real_data
```

## Interpretation Limits

The Sprint 03 result is real-data evidence that the software can train on VOYA
native `[T,1605]` features. It does not prove:

- canonical VietSign55 compatibility
- signer-independent generalization
- full VOYA performance
- RGB, multimodal, CSLR, or translation performance

## Sprint 04 Follow-Up

Sprint 04 adds a separate scaled benchmark under:

```text
outputs/real_experiments/voya_direct/sprint04/
```

The Sprint 03 run remains a 4-class proof-of-pipeline. Sprint 04 uses 10
classes and includes temporal-shuffle, mean-pooling, single-frame,
no-normalization, and label-shuffle diagnostics. The scaled run suggests that
the selected VOYA native features are highly class-separable and that static
feature information is strong.
