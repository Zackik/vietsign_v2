# Continuous Gloss Recognition

Phase 09 adds a continuous sign language recognition foundation. It predicts
ordered gloss token sequences from cached VietSign55 keypoints. It does not
translate glosses into natural Vietnamese.

## Scope

Implemented:

- Continuous sample schema with ordered gloss labels.
- Gloss vocabulary with CTC blank token.
- Temporal Transformer encoder plus CTC projection head.
- PyTorch CTC loss.
- Greedy CTC decoding.
- Gloss error rate metrics.
- Synthetic smoke tests for training, checkpoint reload, evaluation, and
  prediction.

Not implemented:

- Real continuous VSL dataset training.
- Beam search language-model decoding.
- Gloss-to-Vietnamese translation.
- Semantic interpretation of predicted glosses.

## Data Contract

Continuous metadata is CSV-based for the current foundation. Required columns:

```text
sample_id
video_path
glosses
split
```

Optional columns include `signer_id`, `sentence_id`, `translation`, and
`dataset`. The `glosses` field is an ordered sequence split by whitespace or
pipe characters.

Example:

```csv
sample_id,video_path,glosses,signer_id,split
s001,data/raw/example.mp4,TOI MUON NUOC,signer_a,train
```

The cached visual feature contract remains:

```text
coordinates = [T,55,3]
valid_mask = [T,55]
```

Training expands the joint feature channel according to config. With validity
enabled:

```text
coordinates = [B,T,55,4]
valid_mask = [B,T,55]
temporal_mask = [B,T]
```

## Gloss Vocabulary

The vocabulary is stored in checkpoints as:

```json
{
  "id_to_token": ["<blank>", "GLOSS_A", "GLOSS_B"],
  "blank_token": "<blank>"
}
```

`blank_index = 0` by default. The blank token is a CTC symbol and is distinct
from temporal padding. Padding is represented by `temporal_mask` and CTC
`input_lengths`.

## Model Contract

The continuous recognizer wraps the Phase 04 Temporal Transformer frame encoder:

```text
[B,T,J,C]
-> frame tokens [B,T,D]
-> CTC projection
-> logits [B,T,V]
```

where `V` is the gloss vocabulary size including the blank token. The CTC loss
internally uses log-probabilities shaped `[T,B,V]`, as expected by PyTorch.

The model rejects impossible CTC batches where any `target_length` is greater
than its corresponding `input_length`.

## Decoding

Greedy decoding:

1. Select the highest-probability token id at each timestep.
2. Collapse consecutive duplicate ids.
3. Remove blank ids.
4. Map ids back to gloss tokens.

This preserves repeated glosses only when the CTC path contains a blank between
the two occurrences.

Beam search is intentionally stubbed for a later phase.

## Metrics

Gloss error rate is edit distance over gloss tokens:

```text
GER = (substitutions + deletions + insertions) / reference_gloss_count
```

The evaluator reports substitutions, deletions, insertions, reference length,
exact sequence accuracy, and common substitution pairs.

Synthetic smoke metrics are software-validation signals only. They are not VSL
benchmarks.

## Commands

Train:

```bash
python -m vietsign.cli train --config configs/training/continuous_ctc.yaml
```

Evaluate:

```bash
python -m vietsign.cli evaluate-continuous --config configs/training/continuous_ctc.yaml --checkpoint outputs/runs/<run>/best.pt
```

Predict from a keypoint cache:

```bash
python -m vietsign.cli predict-continuous --features data/cache/keypoints/<dataset>/<sample_id>.npz --checkpoint outputs/runs/<run>/best.pt
```
