# Predeclared ogbn-arxiv matched-initialization propagation control

Frozen 2026-09-25 16:23 UTC, before inspecting any result or score from the
separate 300-epoch all-nine repeat. This is a bounded, post hoc mechanistic
extension on **one** official graph and split. It is not a new benchmark search.

## Fixed design

- Dataset and graph: the existing `ogbn_arxiv_pilot.load_official_ogbn_arxiv`
  loader, the unmodified OGB temporal train/valid/test indices, all node
  features, and `to_undirected` citation edges. The existing
  `ogbn_arxiv_300_results/dataset_manifest.json` is the required dataset
  fingerprint. No new graph preprocessing or split.
- Seeds: exactly 0, 1, 2. Each seed creates **one pair** of arms:
  `tied` (the repository's `models.TABMModel`) and
  `untied_propagation`. Four members in each arm.
- Architecture: SAGE, two residual propagation blocks, width 128, LayerNorm,
  GELU, dropout 0.2, input/output BatchEnsemble projector blocks, and the
  existing output normalization. In the untied arm only, the single initialized
  `residual_modules` propagation stack is replaced by four **independent deep
  copies** of that exact stack, one per member. Within each arm the
  input/output projector blocks and output normalization retain their original
  sharing. The tied arm retains one shared propagation stack.
- Pairing: construct one tied model after seeding Python, NumPy, PyTorch and
  CUDA; deep-copy its initialized components for the untied model without
  sampling any new weights. The four untied propagation stacks start with
  identical values to the tied stack but have distinct parameter storage.
  Verify all four initial member logit matrices are equal between arms on a
  synthetic graph. Capture the RNG state immediately after pair construction;
  restore that identical state before each arm's training. Record initial
  model/stack/factor hashes, parameter and storage checks, and RNG hashes.
  Stop before training if any equality or isolation check fails.
- Optimization in both arms: exactly **300** full-batch epochs, no early
  stopping; AdamW, learning rate 0.001, weight decay 0; sequential member
  forwards with the mean of the four training-label member cross entropies
  and one optimizer step per epoch. Validate pooled (mean-logit) predictions
  after every epoch. Run tied then untied for each seed, with the RNG reset
  before the second arm. No hyperparameter, architecture, seed, or epoch-budget
  changes based on observed scores.
- Checkpoint selection within each arm: maximum pooled validation accuracy;
  ties by minimum pooled validation cross entropy (the pilot's 1e-12
  comparison tolerance), then earliest epoch. Save one complete arm state at
  that epoch. Restore and check the selected validation metrics before using
  test indices or labels. Score the official test set once per restored
  checkpoint. Never use test scores for selection.
- Primary descriptive contrast: per-seed selected test accuracy of
  `tied` minus `untied_propagation`; report all three differences, their
  mean and sample standard deviation. Also report selected validation
  accuracy/CE, selected epoch, parameter counts, elapsed training/evaluation
  time and peak PyTorch allocated memory. Three optimization seeds on one
  temporal split are not independent graph replications; no significance or
  generalization claim is planned.
- Output: a new `experiments_iclr/ogbn_arxiv_untied_results/` root with a
  run config, dataset manifest, initialization audit, per-seed/per-arm
  `epochs.csv`, selected checkpoint, selected metrics and logits, summary,
  and checksums. Require a new empty result root. Partial artifacts are
  explicitly incomplete and are never silently accepted on resume. A
  separate verifier must fail closed on missing, extra, malformed, changed,
  wrong-seed, wrong-arm, non-300-epoch or selection-inconsistent artifacts.

## Source and configuration freeze

The existing 300-epoch run configuration was inspected for design settings
only, before any 300-epoch scores. Its source hashes match the current files:

| Source | SHA256 |
| --- | --- |
| `experiments_iclr/ogbn_arxiv_pilot.py` | `7abf975fcf900f52a01a51652605b2d2519aaee02918948732ee5401952dd89b` |
| `models.py` | `07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303` |
| `experiments_iclr/ogbn_arxiv_300_results/run_config.json` | `ee26cd93647d77c964a554960ab4893fdc95e5bd7d73fd9479e0a8c32a1bde97` |
| Existing 300-epoch launch script (read only) | `7d8f556c20f6ec2b2c63eb64d1e3952dad77028bd8a1b4d2568e9352c8cad036` |

The new runner and verifier SHA256 values will be appended in a separate
source-lock manifest after CPU preflight. Those values must be checked before
any later GPU run. Implementation changes after examining the existing repeat
scores require a new disclosed protocol version, not a quiet overwrite.

## Execution gate

This document authorizes preparation and CPU synthetic verification only.
Do not start GPU training now. Schedule a later run only when higher-priority
queued work has completed, an approved GPU is idle, and the source lock and
dataset fingerprint checks pass. Do not modify or interrupt active launchers
or the ongoing 300-epoch repeat.
