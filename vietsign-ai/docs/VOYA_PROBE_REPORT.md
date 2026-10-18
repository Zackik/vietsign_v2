# VOYA Structure Probe Report

This report probes one selected VOYA_VSL class file. It does not verify canonical VietSign55 landmark ordering.

## Summary

- Dataset: `VOYA_VSL`
- Class: `class_0001`
- Mapping status: `STRONGLY_SUPPORTED_BUT_UNVERIFIED`
- Canonical conversion: `blocked_without_authoritative_layout`
- Safe adapter decision: `B. canonical conversion unverified, but direct 1605-feature training possible`

## Observed NPZ

- Path: `data/raw/voya_vsl_probe/class_0001.npz`
- SHA256: `9090bac1398e699997bfccca2754e3ec040283b54ec29cd5e6f3e15a22d22f5f`
- Keys: `['sequences', 'labels']`
- Number of sequences: `1001`
- Sampled sequence shape: `[60, 1605]`
- Feature dim: `1605`
- NaN count: `0`
- Inf count: `0`
- Zero ratio: `0.002616822429906542`

## Candidate Layouts

- `pose25_face468_left21_right21_xyz`: matches_feature_dim=True, expected_feature_dim=1605
- `face468_pose25_left21_right21_xyz`: matches_feature_dim=True, expected_feature_dim=1605
- `pose33_face460_left21_right21_xyz`: matches_feature_dim=True, expected_feature_dim=1605

## Direct Feature Training

- Status: `feasible`
- Input contract: `[T,1605] frame-level source-layout features`
- Reason: A temporal model can project 1605-dim VOYA frames directly. This is a source-layout baseline, not VietSign55 graph/canonical training.

## Canonical Mapping Decision

The `[T,1605]` tensor can be used for a direct VOYA source-layout baseline. Conversion into VietSign55 `[T,55,3]` remains blocked until the VOYA feature ordering is confirmed from authoritative documentation or source code.
