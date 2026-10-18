# VietSignAI Keypoint Contract

Phase 3 converts videos into backend-independent keypoint tensors. This is a
data contract for Phase 4+ models, not a recognition model.

## Canonical Tensor Shape

Default model-ready arrays use:

```text
coordinates = [T, J, 3]
valid_mask  = [T, J]
confidence  = [T, J]
source_mask = [T, J]
```

Where:

- `T` is the number of sampled frames.
- `J = 55` by default.
- `3` is `x, y, z`.
- `valid_mask=false` means the coordinate is missing and stored as zero.
- `source_mask`: `0=missing`, `1=original detection`, `2=interpolated`.

Default `J = 55` is:

- left hand: 21 joints
- right hand: 21 joints
- pose: 13 upper-body joints
- face: disabled by default

## Hand Joint Order

Each hand has exactly 21 joints:

```text
0 wrist
1 thumb_cmc
2 thumb_mcp
3 thumb_ip
4 thumb_tip
5 index_mcp
6 index_pip
7 index_dip
8 index_tip
9 middle_mcp
10 middle_pip
11 middle_dip
12 middle_tip
13 ring_mcp
14 ring_pip
15 ring_dip
16 ring_tip
17 pinky_mcp
18 pinky_pip
19 pinky_dip
20 pinky_tip
```

Tensor regions:

```text
coordinates[:, 0:21, :]   = left hand
coordinates[:, 21:42, :]  = right hand
coordinates[:, 42:55, :]  = pose
```

## Pose Joints

The Phase 3 pose subset is:

```text
nose
left_eye
right_eye
left_ear
right_ear
left_shoulder
right_shoulder
left_elbow
right_elbow
left_wrist
right_wrist
left_hip
right_hip
```

This focuses on upper-body sign-language cues while allowing future full-body
pose adapters.

## Coordinate Systems

MediaPipe landmarks are image-relative when available. The canonical schema does
not expose MediaPipe objects.

Implemented normalization strategies:

- `none`: keep extractor coordinates.
- `image`: divide pixel coordinates by image width/height when pixel coordinates
  are provided.
- `body_relative`: use shoulder midpoint as origin and shoulder distance as
  scale. If either shoulder is missing or scale is near zero, that frame is left
  unchanged.

## Missing Landmarks

Frames are never dropped just because hands or pose are missing.

Missing keypoints are represented as:

```text
coordinates = 0
valid_mask = false
confidence = NaN
source_mask = 0
```

This lets future models distinguish a real coordinate near zero from an absent
detection.

## Interpolation

Short-gap interpolation is optional:

```yaml
interpolation:
  enabled: true
  max_gap: 3
```

Only gaps between two valid detections are interpolated. Large gaps and fully
missing sequences remain missing. Interpolated values use `source_mask=2`.

## Motion Features

Optional velocity is:

```text
velocity[t] = coordinates[t] - coordinates[t - 1]
```

Velocity is zero when either frame is invalid for that joint. Acceleration is
computed from adjacent valid velocity spans when enabled.

## Cache Format

Keypoints are cached as:

```text
data/cache/keypoints/<dataset>/<sample_id>.npz
data/cache/keypoints/<dataset>/<sample_id>.json
```

The NPZ stores:

```text
coordinates
valid_mask
confidence
source_mask
timestamps
frame_indices
joint_names
velocity              optional
acceleration          optional
```

The JSON stores sample metadata, extractor information, normalization type,
detection coverage, tensor shape, schema version, and preprocessing fingerprint.
Pickle is not used.

## Backend Notes

The adapters are isolated under `vietsign.vision`. They do not leak MediaPipe
protobuf objects or RTMLib objects. Phase 05 requires explicit model assets for
real inference and fails clearly when assets are missing.

RTMLib/RTMPose uses `to_openpose=True` and maps:

```text
RTMLib left_hand = 92:113 -> coordinates[:, 0:21, :]
RTMLib right_hand = 113:134 -> coordinates[:, 21:42, :]
RTMLib body OpenPose joints -> coordinates[:, 42:55, :]
```

The canonical pose body subset is exactly:

```text
nose, left_eye, right_eye, left_ear, right_ear,
left_shoulder, right_shoulder, left_elbow, right_elbow,
left_wrist, right_wrist, left_hip, right_hip
```

Future MMPose or HaMeR adapters must return the same `FrameKeypoints` and
`KeypointSequence` structures.
