# VietSignAI Graph Models

Phase 06 adds ST-GCN and CTR-GCN baselines for isolated sign classification.
They consume the same cached keypoint tensors as the Temporal Transformer:

```text
coordinates = [T,55,3]
valid_mask  = [T,55]
training input = [B,T,J,C]
```

## VietSign55 Topology

The graph node order exactly matches the canonical tensor contract:

```text
0      left_hand:wrist
20     left_hand:pinky_tip
21     right_hand:wrist
41     right_hand:pinky_tip
42     pose:nose
52     pose:left_wrist
53     pose:right_wrist
54     pose:right_hip
```

Edges are anatomical:

- Each hand uses wrist-to-finger chains for thumb, index, middle, ring, and pinky.
- Left hand wrist connects to `pose:left_wrist`.
- Right hand wrist connects to `pose:right_wrist`.
- Pose wrists connect through elbow to shoulder.
- Shoulders connect to each other and to hips.
- Nose, eyes, and ears connect through the upper-body pose subset.

Self connections are included in the normalized adjacency used by baseline
ST-GCN. Optional cross-hand relationships are reserved for future experiments
and are not enabled by default.

## ST-GCN

ST-GCN converts `[B,T,J,C]` to `[B,C,T,V]`, applies spatial graph convolution
with the VietSign55 adjacency, then temporal convolution blocks, residual
connections, dropout, and temporal-mask-aware global pooling.

Invalid joints are zeroed using `valid_mask`; padded frames are excluded using
`temporal_mask`.

## CTR-GCN

CTR-GCN keeps the same input and graph contract but adds a channel-wise adaptive
topology refinement branch. The anatomical graph remains the base structure,
while learned relations can refine it per sample/channel.

## Feature Subsets

The graph is induced from selected canonical joints. Supported subsets include:

```text
hands only
pose only
hands + pose
left hand only
right hand only
```

No separate preprocessing cache is created for graph models.

## Synthetic Benchmark Warning

Synthetic smoke runs are labeled:

```text
SYNTHETIC SOFTWARE VALIDATION - NON-VSL / NOT A VSL BENCHMARK
```

They verify software plumbing only: training, checkpointing, evaluation,
prediction, report writing, masks, and model forward passes.
