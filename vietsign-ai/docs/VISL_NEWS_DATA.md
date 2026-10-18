# ViSL-News Data Investigation

ViSL-News is a sentence-level Vietnamese Sign Language dataset hosted on
Hugging Face at `kha2612/ViSL-News`.

## Remote Metadata

The dataset card reports:

- 26,395 sentence-level samples.
- 67 hours of signing video.
- 5 signers.
- Vietnamese text paired with sentence-level signing video clips.
- Splits:
  - train: 24,399 rows
  - validation: 997 rows
  - test: 999 rows
- License field: `other`.

The Hugging Face repository metadata currently lists CSV split files:

```text
train/train_info.csv
val/val_info.csv
test/test_info.csv
```

The dataset viewer exposes fields including:

```text
file_id
start
end
text
signer_id
```

## CSLR Suitability

ViSL-News does not expose gloss sequence annotations in the inspected public
metadata. Therefore it is **not** a standard CTC CSLR dataset.

Do not feed Vietnamese sentence text into CTC as gloss labels.

## SLT Suitability

ViSL-News may be useful for gloss-free sentence-level video-to-Vietnamese
translation research once actual video accessibility is resolved.

It is not directly compatible with the Phase 10 gloss-to-Vietnamese translator,
because Phase 10 consumes symbolic gloss sequences rather than raw videos or
sentence-level video clips.

## Current Local Status

No local ViSL-News files were found under:

```text
data/raw/visl_news/
```

Current status:

```text
metadata_verified_remote_not_downloaded
```
