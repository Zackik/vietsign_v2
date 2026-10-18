# Gloss-to-Vietnamese Translation

Phase 10 adds a separate sign language translation layer:

```text
Gloss sequence -> Vietnamese sentence
```

It does not train a direct video-to-Vietnamese model. Visual recognition remains
Phase 09 CSLR, and translation consumes the resulting gloss sequence.

## Task Boundaries

```text
ISLR: video -> one sign class
CSLR: video -> gloss sequence
SLT:  gloss sequence -> Vietnamese sentence
```

Glosses are intermediate symbolic labels. They are not Vietnamese words, and
their order may differ from natural Vietnamese word order.

## Dataset Contract

Translation metadata is paired text data with required CSV columns:

```text
sample_id
gloss_sequence
vietnamese_text
split
```

Optional columns include `signer_id`, `source_video`, and `dataset`.

Example:

```csv
sample_id,gloss_sequence,vietnamese_text,split
one,TOI NGAY-MAI BENH-VIEN DI,Ngày mai tôi sẽ đi bệnh viện.,train
```

The normalized sample keeps both representations:

```text
glosses: ["TOI", "NGAY-MAI", "BENH-VIEN", "DI"]
source_text: "TOI NGAY-MAI BENH-VIEN DI"
target_text: "Ngày mai tôi sẽ đi bệnh viện."
```

## Normalization

Vietnamese text normalization is conservative:

- Unicode NFC normalization.
- Whitespace cleanup.
- Basic punctuation spacing cleanup.
- Optional lowercasing by config.
- Vietnamese diacritics are preserved.

Gloss normalization is separate:

- Whitespace cleanup.
- Optional uppercase policy.
- No gloss token is translated into Vietnamese text during normalization.

## Vocabularies

The custom Transformer uses separate source and target vocabularies:

```text
<PAD>
<BOS>
<EOS>
<UNK>
```

CTC `<blank>` is not part of translation vocabularies. Padding and CTC blank are
separate concepts.

## Model

The baseline model is:

```text
source gloss ids
-> source embedding + positional encoding
-> Transformer encoder
-> Transformer decoder with causal mask
-> target projection
-> logits [B,T_target,V_target]
```

Training uses teacher forcing and `CrossEntropyLoss(ignore_index=PAD)`.

Pretrained text-to-text support is exposed as an adapter boundary. CI does not
download external weights, and missing optional dependencies fail clearly.

## Decoding

Greedy autoregressive decoding is implemented:

```text
<BOS> -> token -> token -> ... -> <EOS>
```

Generation is capped by `decoding.max_new_tokens`, so it cannot loop forever.
Beam search is deliberately not faked in this phase.

## Evaluation

Supported deterministic metrics:

- whitespace-token BLEU
- character-level chrF
- exact match

Vietnamese tokenization is simple whitespace tokenization for this baseline.
This is useful for software validation, not a claim of linguistic superiority.

Two evaluation modes must stay distinct:

- Oracle gloss: ground-truth glosses -> Vietnamese.
- Predicted gloss: CSLR predicted glosses -> Vietnamese.

## Commands

Train:

```bash
python -m vietsign.cli train --config configs/training/gloss_translation.yaml
```

Evaluate oracle gloss translation:

```bash
python -m vietsign.cli evaluate --config configs/training/gloss_translation.yaml --checkpoint outputs/runs/<run>/best.pt
```

Translate one gloss sequence:

```bash
python -m vietsign.cli translate --gloss "TOI NGAY-MAI BENH-VIEN DI" --config configs/training/gloss_translation.yaml --checkpoint outputs/runs/<run>/best.pt
```

Translate a Phase 09 predictions CSV:

```bash
python -m vietsign.cli translate --input predictions.csv --output translations.csv --config configs/training/gloss_translation.yaml --checkpoint outputs/runs/<run>/best.pt
```

Compose Phase 09 and Phase 10 on cached continuous features:

```bash
python -m vietsign.cli recognize-and-translate \
  --features data/cache/keypoints/<dataset>/<sample_id>.npz \
  --cslr-checkpoint outputs/runs/<cslr-run>/best.pt \
  --translation-checkpoint outputs/runs/<translation-run>/best.pt
```

## Limitations

The included translation fixture is:

```text
SYNTHETIC TRANSLATION SOFTWARE VALIDATION - NOT REAL VSL PERFORMANCE
```

No real paired VSL gloss/Vietnamese corpus is included yet. Synthetic BLEU,
chrF, or exact-match values must not be reported as real translation quality.
