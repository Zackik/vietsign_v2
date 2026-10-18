# Parallel-Corpus-Vie-VSL Investigation

The repository `BichDiep/Parallel-Corpus-Vie-VSL` provides Vietnamese/VSL text
resources.

## Remote Files Listed Publicly

The inspected GitHub listing includes:

```text
Corpus-Vie-VSL-10K.xlsx
Parallel Corpus Vie-VSL.rar
VSL-Lexicon.rar
VSL10k.txt
VSL10k_phantich.txt
Vie-Traveling.xlsx
Vie10k.txt
Vie10k_phantich.txt
README.md
```

The README says the data is intended for research and educational purposes.
No explicit license file was verified during this sprint.

## Local Adapter

VietSignAI now includes a local adapter for text files:

```text
data/raw/parallel_corpus_vie_vsl/Vie10k.txt
data/raw/parallel_corpus_vie_vsl/VSL10k.txt
```

It produces `TranslationSample` rows while preserving provenance:

```text
dataset = Parallel-Corpus-Vie-VSL
original_line_index
original_vietnamese
original_vsl
license_status = unknown
```

## Current Status

No local corpus files were found. No real Phase 10 experiment was run.

Before automated training, clarify license/redistribution terms.
