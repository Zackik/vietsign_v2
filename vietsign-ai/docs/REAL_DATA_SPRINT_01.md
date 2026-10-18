# Real Data Sprint 01 Report

## VOYA_VSL

- Access status: public Hugging Face metadata verified.
- Remote size: approximately 56.3 GB.
- Remote class files: 161 class `.npz` files listed.
- Local status: not downloaded.
- Local size: 0 bytes.
- Real class count: not locally verified.
- Sample count: not locally verified.
- Signer metadata: not found in public metadata.
- Split metadata: not found in public metadata.
- Mapping status: source layout unverified; VietSign55 mapping blocked.
- Real experiment status: not run.

## VSL400

- Access status: controlled access with Data Usage Agreement.
- Authorized files available locally: no.
- Adapter status: `WAITING_FOR_AUTHORIZED_DATA`.
- Next action: obtain authorized access, place files under `data/raw/vsl400`,
  then inspect actual layout before parser finalization.

## Parallel-Corpus-Vie-VSL

- Access status: public GitHub metadata verified.
- Actual pair count: not locally verified.
- License status: unknown; requires clarification before redistribution or
  automated training use.
- Phase 10 adapter status: local text adapter implemented for `Vie10k.txt` and
  `VSL10k.txt`.
- Real translation experiment status: not run.

## ViSL-News

- Remote row counts reported:
  - train: 24,399
  - validation: 997
  - test: 999
- Signer metadata: `signer_id` field appears in public viewer metadata.
- Actual video availability: unresolved; repo metadata lists CSV files.
- Gloss annotations: no.
- Usability:
  - CSLR: not suitable for CTC gloss recognition without gloss labels.
  - SLT: possible future gloss-free sentence-level video-to-text dataset, not
    directly Phase 10 gloss-to-Vietnamese.

## Real Data Readiness

- `real_keypoint_dataset_ready`: false locally.
- `real_islr_experiment_ready`: false locally.
- `real_video_extractor_accepted`: false.
- `real_translation_text_data_ready`: false locally.

## Blockers

- VOYA_VSL full data not downloaded.
- VOYA feature layout not mapped to VietSign55.
- VSL400 requires authorized access.
- Parallel corpus files not present locally and license is unclear.
- ViSL-News has no gloss labels and local media access is unresolved.
