# VOYA_VSL Data Investigation

VOYA_VSL is hosted on Hugging Face at `Kateht/VOYA_VSL`.

## Remote Metadata

The inspected remote metadata reports:

- Access: public Hugging Face dataset.
- License: MIT.
- Storage size: approximately 56.3 GB.
- Files: `Merged/class_0001.npz` through `Merged/class_0161.npz`,
  plus `README.md`, `dataset.py`, and `labels.json`.
- `dataset.py` declares:

```text
sequences: Array2D(shape=(60,1605), dtype=float32)
labels: int32
```

The dataset builder iterates each class `.npz` and yields individual
`sequences` and `labels` rows.

## Local Fetch

Dry-run:

```bash
python -m vietsign.cli dataset fetch voya-vsl --dry-run
```

Metadata-only:

```bash
python -m vietsign.cli dataset fetch voya-vsl --metadata-only
```

One-class structure probe:

```bash
python -m vietsign.cli dataset fetch voya-vsl \
  --target data/raw/voya_vsl_probe \
  --classes class_0001 \
  --max-class-size-mb 1024

python -m vietsign.cli dataset probe-voya \
  --root data/raw/voya_vsl_probe \
  --class class_0001 \
  --visualize
```

Full download command:

```bash
huggingface-cli download Kateht/VOYA_VSL --repo-type dataset --local-dir data/raw/voya_vsl --local-dir-use-symlinks False
```

Downloaded data must not be committed.

## Mapping Status

`1605 / 3 = 535` coordinate triplets, but public metadata does not define which
feature indices correspond to pose, face, left hand, right hand, confidence, or
visibility.

Therefore VietSignAI does **not** map VOYA into canonical VietSign55 yet.

Current safe status:

```text
source-layout adapter/audit only
canonical_vietsign55_mapping: not_implemented
```

The converter deliberately raises an error until the source feature layout is
verified.

## Sprint 02 Probe Result

A controlled probe downloaded only metadata plus `Merged/class_0001.npz`.

Observed local file:

- `class_0001.npz` size: `352808182` bytes.
- SHA256: `9090bac1398e699997bfccca2754e3ec040283b54ec29cd5e6f3e15a22d22f5f`.
- Keys: `sequences`, `labels`.
- `sequences.shape`: `[1001, 60, 1605]`.
- Sample sequence shape: `[60, 1605]`.
- Labels shape: `[1001]`.
- NaN count: `0`.
- Inf count: `0`.

The probe found plausible mathematical candidate layouts such as
`pose25_face468_left21_right21_xyz`, because `(25 + 468 + 21 + 21) * 3 = 1605`.
This is useful evidence, but it is not authoritative mapping proof.

Safe current decision:

```text
VOYA direct-feature training: ready for [T,1605] source-layout baseline
VOYA -> VietSign55 [T,55,3]: still blocked
mapping_status: STRONGLY_SUPPORTED_BUT_UNVERIFIED
```

Generated local artifacts:

- `outputs/validation/voya_probe.json`
- `docs/VOYA_PROBE_REPORT.md`
- `outputs/validation/voya_probe_structure.png`
