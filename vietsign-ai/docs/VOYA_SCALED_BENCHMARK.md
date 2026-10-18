# VOYA Scaled Benchmark and Leakage Audit

Sprint 04 scales the VOYA native-feature experiment beyond the Sprint 03
4-class proof-of-pipeline.

## Scope

- Dataset: `VOYA_VSL`
- Representation: native `[T,1605]`
- Canonical VietSign55: `false`
- Signer-independent: `false`
- Split: deterministic sample-stratified
- Checkpoint maturity: `experimental_real_data`

No anatomical graph model is used because the VOYA feature ordering is still not
authoritatively known.

## Label Catalog

`labels.json` contains `4363` entries.

- Unique display strings: `4353`
- Unique normalized labels: `4352`
- Duplicate display labels: `10`
- Same-root groups after stripping parenthetical markers: many regional and
  lexical variants are present.

Generated artifacts:

- `outputs/validation/voya_label_analysis.json`
- `outputs/validation/voya_label_groups.csv`

## Downloaded Scale

Sprint 04 downloaded and inspected `10` class NPZ files:

```text
class_0001 ... class_0010
```

Downloaded NPZ disk size: `3507396741` bytes.

The 10-class experiment used a deterministic cap of `60` samples per class.

## Leakage and Artifact Audits

- Exact duplicate sequence instances in the selected subset: `407`
- Cross-split exact duplicate count after enforcement: `0`
- Near-duplicate audit: diagnostic only, based on cosine similarity over
  temporal mean and standard-deviation vectors.
- File-artifact audit compares class-level feature mean, variance, zero ratio,
  sequence variance, and coordinate ranges.

Because each semantic class lives in a separate NPZ file, class-level
distribution differences remain a concern. They are not proof of leakage, but
they help explain why static baselines can perform strongly.

## Results

All results are real VOYA native-feature results, not signer-independent VSL
recognition.

| Variant | Top-1 | Top-5 | Macro F1 | Weighted F1 | Chance Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Temporal Transformer | 0.8056 | 1.0000 | 0.7327 | 0.7419 | 0.1000 |
| Mean-pooling MLP | 0.7222 | 1.0000 | 0.6189 | 0.6470 | 0.1000 |
| Single-frame MLP | 0.8333 | 1.0000 | 0.7455 | 0.7727 | 0.1000 |
| Temporal-shuffled Transformer | 0.7778 | 1.0000 | 0.7124 | 0.7048 | 0.1000 |
| Transformer, no normalization | 0.3333 | 0.8333 | 0.1586 | 0.1877 | 0.1000 |
| Label-shuffle negative control | 0.1111 | 0.4167 | 0.0258 | 0.0287 | 0.1000 |

The label-shuffle negative control behaved near chance, so this run did not
trigger the leakage stop condition.

## Interpretation

The normal Transformer improves over mean pooling, but the single-frame and
temporal-shuffled diagnostics remain strong. This suggests that the selected
VOYA classes are highly separable under native features and that temporal order
is not the only useful signal in this subset.

The no-normalization ablation is much weaker, indicating train-only z-score
normalization is important for this pipeline.

## Artifacts

Main directory:

```text
outputs/real_experiments/voya_direct/sprint04/
```

Important files:

- `summary.json`
- `variant_metrics.csv`
- `split_manifest.json`
- `duplicate_audit.json`
- `near_duplicate_audit.json`
- `file_artifact_audit.json`
- per-variant `evaluation.json`
- per-variant `training_history.csv`
- per-variant `confusion_matrix.png`

## Limits

This is still not a validated signer-generalized benchmark because signer
identity is unavailable. Canonical VietSign55 conversion remains unverified.
