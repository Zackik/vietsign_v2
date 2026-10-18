# VietSignAI Roadmap

## Phase 1: Foundation

Status: implemented.

- Python package layout.
- YAML configuration.
- Safe device detection for CUDA, MPS, and CPU.
- Shared logging.
- Reproducible seed setup.
- CLI command skeleton.
- Foundation tests and documentation.

## Phase 2: Data Contracts

Status: implemented.

- Define manifest schema for videos, labels, signers, splits, and metadata.
- Add dataset validation scripts.
- Establish storage conventions under `data/`.
- Add small synthetic fixtures for tests.

## Phase 3: Visual Extraction

Status: canonical tensor contract implemented.

- Store extracted features in a versioned format.
- Maintain `coordinates=[T,55,3]`, `valid_mask=[T,55]`.
- Keep `J=55` fixed as 21 left hand + 21 right hand + 13 upper-body pose joints.

## Phase 4: Recognition Modeling

Status: keypoint-only Transformer baseline implemented.

## Phase 5: Real Extractor Validation

Status: in progress.

- Validate RTMLib/RTMPose on real local videos.
- Require non-zero pose coverage and at least one non-zero hand coverage.
- Reject all-false `valid_mask` caches before moving on.
- Keep visualizations and cache metadata for quality review.

## Phase 6: Graph Recognition Models

Status: implemented with synthetic software-validation smoke tests.

- Define VietSign55 anatomical graph.
- Add ST-GCN and CTR-GCN baselines.
- Use the shared cached-keypoint training, evaluation, checkpoint, and metrics pipeline.
- Support feature-subset induced subgraphs.
- Save synthetic comparison reports labeled NON-VSL.

## Phase 7: RGB Video Recognition

Status: implemented with generated-video software-validation smoke tests.

- Add RGBVideoDataset over existing SignSample metadata.
- Add RGB contract `[B,T,C,H,W]`.
- Add VideoMAE adapter with frozen-backbone default.
- Add RGB embedding cache as NPZ + JSON with fingerprinting.
- Add video prediction command and RGB preprocessing command.
- Extend synthetic benchmark representation to include VideoMAE.

## Phase 8: Multimodal Fusion

Status: implemented with synthetic software-validation smoke tests.

- Combine pose features and RGB features.
- Evaluate fusion strategies on validated datasets.
- Include late concat, gated fusion, and cross-attention fusion baselines.

## Phase 9: Continuous Gloss Recognition

Status: implemented with synthetic software-validation smoke tests.

- Define continuous sample schema with ordered gloss sequences.
- Build gloss vocabulary with CTC blank token at index `0`.
- Add CTC recognizer on top of the Phase 04 Temporal Transformer encoder.
- Add greedy CTC decoding and gloss error rate evaluation.
- Save and reload gloss vocabulary with checkpoints.

Synthetic Phase 09 runs validate software behavior only. They are not VSL
benchmarks and do not claim real continuous-sign performance.

## Phase 10: Vietnamese Translation

Status: implemented with synthetic software-validation smoke tests.

- Translate gloss sequences into natural Vietnamese with a custom Transformer
  baseline.
- Keep translation separate from visual recognition and CSLR checkpoints.
- Support oracle-gloss evaluation and Phase 09 predicted-gloss composition.
- Report BLEU, chrF, and exact match on synthetic fixtures only.

Synthetic Phase 10 metrics validate software behavior only. They are not real
VSL translation-quality claims.

## Phase 11: Product API and Realtime Shell

Status: implemented with synthetic software-validation API tests.

- Add model-independent inference orchestration.
- Add lazy model/checkpoint manager.
- Add FastAPI health, model, pipeline, translation, CSLR, and composition routes.
- Add upload validation and temporary cleanup policy.
- Add realtime rolling-window/session primitives.
- Add a minimal TypeScript web console.

## Phase 12: Real-Data Hardening and Validation

Status: implemented as validation tooling; real data still required.

- Add real-video discovery and extractor readiness reporting.
- Add dataset discovery and signer-leakage audit utilities.
- Add checkpoint maturity and checkpoint manifest reporting.
- Add granular API readiness fields.
- Generate `outputs/validation/phase12_manifest.json`.
- Generate `docs/VALIDATION_REPORT.md` and `docs/REAL_DATA_GAPS.md`.

Current status is software-validated only because no real acceptance video or
real local VSL dataset is present in the repository.

## Phase 13: Real Data and Evaluation

Status: started with VOYA direct-feature baseline.

- Train and evaluate VOYA native `[T,1605]` source-layout baseline.
- Keep VOYA outputs under `outputs/real_experiments/`.
- Mark VOYA checkpoints as `experimental_real_data`.
- Acquire or define a real paired VSL gloss/Vietnamese dataset.
- Run real Phase 05 extraction acceptance before raw-video pipelines.
- Add human evaluation and robust predicted-gloss translation analysis.

Current VOYA limits:

- `signer_independent=false` because signer metadata is unavailable.
- `canonical_vietsign55=false` because the VOYA 1605 feature ordering is not
  authoritative.

## Not Yet Implemented

- HaMeR.
- Real paired VSL/Vietnamese benchmark.
