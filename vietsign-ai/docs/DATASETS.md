# VietSignAI Datasets

Phase 2 adds a normalized dataset layer. It does not implement pose extraction,
MediaPipe, RTMPose, model training, or translation.

## Supported Dataset Sources

- `custom_csv`: implemented. Loads a normalized CSV manifest.
- `vsl400`: configured. Uses a normalized metadata CSV and verifies that the
  configured VSL-400 root exists. Native VSL-400 parsing is intentionally not
  guessed until the real structure is verified.
- `vsl1000`: reserved. No structure is assumed yet.

## Normalized Sample Schema

Downstream preprocessing and models consume `SignSample` records rather than
dataset-specific directories.

Required fields:

```text
sample_id
video_path
label
label_id
split
dataset
```

Optional fields:

```text
signer_id
camera
duration
fps
frame_count
width
height
metadata
```

Example:

```json
{
  "sample_id": "vsl400_000001",
  "video_path": "data/raw/vsl400/videos/000001.mp4",
  "label": "xin_chao",
  "label_id": 0,
  "signer_id": "signer_01",
  "split": "train",
  "dataset": "vsl400",
  "camera": null,
  "metadata": {}
}
```

## Directory Structure

```text
data/
  raw/
    vsl400/
    vsl1000/
  metadata/
  processed/
  cache/
```

Raw datasets, processed outputs, cached data, and common video files are ignored
by Git. Keep only small metadata examples or `.gitkeep` files under version
control.

## VSL-400 Configuration

The default VSL-400 config is:

```yaml
dataset:
  name: vsl400
  adapter: vsl400
  root: data/raw/vsl400
  metadata_csv: data/metadata/vsl400.csv
```

Place the real VSL-400 files under `data/raw/vsl400/`, or edit
`configs/data/vsl400.yaml` to point to another local root. Do not use absolute
machine-specific paths in committed configs.

Because the native VSL-400 layout has not been verified in this repository yet,
provide a normalized metadata CSV at `data/metadata/vsl400.csv` before loading it.

For Phase 05 real-data onboarding, start with a small normalized subset such as
`data/metadata/vsl400_subset.csv`. Do not alter the original raw dataset. If the
real VSL-400 directory is not locally available, dataset-specific parsing remains
blocked and should not be guessed.

## Custom CSV Format

Use the `custom_csv` adapter for local or future custom VSL datasets.

Required columns:

```text
sample_id,video_path,label,label_id,split
```

Recommended optional columns:

```text
signer_id,camera,dataset
```

Any additional columns are preserved in the sample `metadata` dictionary.
Relative `video_path` values are resolved against the configured dataset root.

Example config:

```yaml
dataset:
  name: custom_vsl
  adapter: custom_csv
  root: data/raw/custom_vsl
  metadata_csv: data/metadata/custom_vsl.csv
```

## Validation

Dataset validation reports:

- missing videos
- duplicate sample IDs
- duplicate video paths
- invalid labels or label IDs
- unknown split names
- missing class IDs in a contiguous label range
- corrupted or unreadable videos
- signer leakage across train, validation, and test splits

Signer leakage is reported but never silently fixed.

## Signer-Independent Evaluation

Signer-independent evaluation matters because sign-language recognition systems
can otherwise learn signer-specific appearance, motion habits, camera positions,
or recording conditions instead of the sign itself. A model can look strong on a
random video split while failing on new signers.

When every sample has a `signer_id`, VietSignAI can generate deterministic splits
that keep each signer in exactly one split. If signer IDs are missing, the system
will clearly state that signer-independent splitting cannot be guaranteed.

## Adding Another Adapter

1. Create a class that extends `BaseDatasetAdapter`.
2. Keep dataset-specific parsing inside the adapter.
3. Return only `list[SignSample]`.
4. Add the adapter option to `DatasetSourceConfig`.
5. Add tests with tiny fixtures, not real dataset downloads.
