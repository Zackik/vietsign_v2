# VietSignAI Models

Phase 04 implemented the first isolated-sign recognition baseline from cached
Phase 03 keypoint tensors. Phase 06 adds graph-based baselines that consume the
same canonical keypoint contract. Phase 07 adds a separate RGB VideoMAE branch,
Phase 08 adds RGB+keypoint fusion, Phase 09 adds continuous gloss recognition
with CTC, and Phase 10 adds gloss-to-Vietnamese translation.

## Input Contract

The canonical cached tensors are:

```text
coordinates: [T, 55, 3]
valid_mask: [T, 55]
confidence: [T, 55]
source_mask: [T, 55]
velocity: [T, 55, 3] optional
```

Training batches use:

```text
coordinates: Tensor[B, T, J, C]
valid_mask: Tensor[B, T, J]
temporal_mask: Tensor[B, T]
labels: Tensor[B]
```

`J` depends on enabled feature groups. With all default groups enabled:

```text
J = 55 = 21 left hand + 21 right hand + 13 upper-body pose
```

With validity enabled, each joint feature is:

```text
x, y, z, valid
```

so `C=4`. Optional confidence, source mask, and velocity append more channels.

## Temporal Transformer Baseline

Pipeline:

```text
[B,T,J,C]
-> per-joint linear projection
-> masked joint aggregation per frame
-> temporal positional encoding
-> Transformer Encoder
-> masked temporal mean pooling
-> classification head
-> logits [B,num_classes]
```

The baseline uses sinusoidal temporal positional encoding and masked mean
pooling. Padding frames are masked in self-attention and pooling.

## Graph Model Baselines

Supported model registry names:

```text
temporal_transformer
stgcn
ctrgcn
videomae
```

All three use the same training command and cached keypoint dataset. Change only
the YAML config:

```bash
python -m vietsign.cli train --config configs/training/stgcn.yaml
python -m vietsign.cli train --config configs/training/ctrgcn.yaml
```

ST-GCN pipeline:

```text
[B,T,J,C]
-> mask invalid joints
-> transpose to [B,C,T,V]
-> spatial graph convolution over VietSign55 adjacency
-> temporal convolution blocks
-> temporal-mask-aware global pooling
-> classification head
```

CTR-GCN uses the same canonical graph input but adds a channel-wise adaptive
topology refinement branch on top of the anatomical adjacency. It is a stronger
graph baseline, not a renamed ST-GCN.

## Missing Data Handling

Invalid coordinates are multiplied by `valid_mask`, and validity can be
concatenated as an explicit input channel. Fully empty sequences are rejected by
default through:

```yaml
data_quality:
  minimum_valid_joint_ratio: 0.05
  reject_empty: true
```

This is important because the current MediaPipe runtime may produce all-missing
tensors when task model assets are unavailable.

## VietSign55 Graph

The canonical graph has 55 nodes in the exact tensor order:

```text
0:21   left hand
21:42  right hand
42:55  upper-body pose
```

Hand edges follow anatomical finger chains:

```text
wrist -> thumb_cmc -> thumb_mcp -> thumb_ip -> thumb_tip
wrist -> index_mcp -> index_pip -> index_dip -> index_tip
wrist -> middle_mcp -> middle_pip -> middle_dip -> middle_tip
wrist -> ring_mcp -> ring_pip -> ring_dip -> ring_tip
wrist -> pinky_mcp -> pinky_pip -> pinky_dip -> pinky_tip
```

Body edges connect each hand wrist to the matching pose wrist, then:

```text
pose wrist -> elbow -> shoulder
left shoulder <-> right shoulder
shoulder -> hip
nose/eyes/ears connected through the upper-body pose subset
```

Feature subsets such as hands-only, pose-only, left-hand-only, and right-hand-only
use induced subgraphs, so excluded regions do not remain as meaningless nodes.

Generate a graph visualization with:

```bash
python -m vietsign.cli model info --config configs/training/stgcn.yaml
```

The visualization is saved to `outputs/graphs/vietsign55.png`.

## Benchmarking

Synthetic smoke comparison is supported to verify software behavior:

```bash
python -m vietsign.cli benchmark models \
  --synthetic-smoke \
  --configs configs/training/baseline.yaml configs/training/stgcn.yaml configs/training/ctrgcn.yaml configs/training/videomae.yaml
```

The output files are:

```text
outputs/benchmarks/phase06-smoke/comparison.csv
outputs/benchmarks/phase06-smoke/comparison.json
```

Synthetic metrics are labeled:

```text
SYNTHETIC SOFTWARE VALIDATION - NON-VSL / NOT A VSL BENCHMARK
```

Do not rank models by synthetic Top-1/F1. Those runs only verify training,
checkpointing, evaluation, prediction, and comparison tooling.

## RGB VideoMAE

RGB models use a separate input contract:

```text
pixel_values = [B,T,C,H,W]
```

The VideoMAE adapter supports frozen-backbone mode by default and reports total
and trainable parameters separately. See [RGB_MODELS.md](RGB_MODELS.md).

## Multimodal Fusion

Multimodal models combine:

```text
RGB VideoMAE embedding/tokens
Pose Temporal Transformer embedding/tokens
```

Supported fusion modes:

```text
concat
gated
cross_attention
```

The multimodal batch contract allows `T_rgb != T_pose` and aligns modalities by
`sample_id`. See [MULTIMODAL.md](MULTIMODAL.md).

## Continuous Gloss Recognition

The Phase 09 CTC recognizer reuses the Temporal Transformer frame encoder:

```text
[B,T,J,C]
-> Temporal Transformer frame tokens [B,T,D]
-> CTC projection
-> logits [B,T,V]
```

`V` is the gloss vocabulary size including the CTC blank token at index `0`.
Padding is handled separately with `temporal_mask`; the blank token is not a
padding label. Greedy decoding takes the argmax path, collapses repeated token
ids, removes blank ids, and returns a gloss sequence.

Continuous batches use:

```text
coordinates: Tensor[B,T,J,C]
valid_mask: Tensor[B,T,J]
temporal_mask: Tensor[B,T]
targets: Tensor[sum(target_lengths)]
input_lengths: Tensor[B]
target_lengths: Tensor[B]
```

See [CONTINUOUS_RECOGNITION.md](CONTINUOUS_RECOGNITION.md) for the gloss data
contract, decoder behavior, and GER metric definition.

## Gloss Translation

The Phase 10 custom text model is:

```text
source gloss ids [B,S]
-> Transformer encoder
target Vietnamese ids [B,T]
-> Transformer decoder with causal mask
-> logits [B,T,V_target]
```

It uses separate gloss/source and Vietnamese/target vocabularies. Translation
special tokens are `<PAD>`, `<BOS>`, `<EOS>`, and `<UNK>`; CTC `<blank>` is not
used by the translation target vocabulary.

See [TRANSLATION.md](TRANSLATION.md) for dataset, decoding, metrics, and
oracle-vs-predicted gloss evaluation notes.
