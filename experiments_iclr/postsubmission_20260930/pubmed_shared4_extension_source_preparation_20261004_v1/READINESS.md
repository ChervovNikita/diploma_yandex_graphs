# Pubmed shared4 extension: source readiness

**Status: source prepared; new bridge parity, resource admission and an external scientific release remain pending.** No new fit, numerical model import, checkpoint deserialization, TEST access, server write or change to a sealed packet was performed.

## Literature and configuration decision

The saved HeaRT Table 4 reports Pubmed SAGE **9.40 ± 0.70% MRR** and NCNC **8.58 ± 0.59% MRR**. The six completed, serialized-replay-audited native VALID cells are:

| Model | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| SAGE | 9.03% | 7.34% | 7.76% |
| NCNC | 8.81% | 7.89% | 8.21% |

Approximately 8% VALID MRR is plausible in this saved published-result scale. These are different-stage results with three local seeds and authenticated local pool custody; this is no published-score reproduction claim. A performance-seeking development configuration check is **not necessary on present evidence**. Retain the prescribed author configuration. Resolve any failed source parity as an implementation repair in a new packet, without tuning to raise MRR.

The saved Table 7 literally says 18,717 Pubmed nodes. The authenticated local raw Planetoid/PyG task has 19,717 nodes and 500 features. Preserve that discrepancy and the local authority. The saved `generate_pubmed.sh` invokes `--dataset cora`; it must not regenerate this task's splits or negative pool.

## Minimal bridge

`bridge.py` is a deferred library with no launcher. It reuses **v2** of the already qualified `CompletionDecoder` and `graph_ops`, which also underpin the established shared4 predictive factory. It replaces that factory's Collab encoder with the exact native Pubmed `GCN` constructor. No earlier source file is edited.

Pubmed uses `puregcn`, an input Linear projection, residual disabled, and an **active one-layer JK scalar**. The author implementation directly multiplies its one-layer output by `jkparams`; it applies no softmax. The `ln=True` constructor flag does not create an encoder LayerNorm/ReLU on this puregcn path. The bridge retains these actual native operations, parameters, gradients and state. Reusing the Collab learned GCN/LN/ReLU encoder would change the task model.

The predictor facade provides:

- `multidomainforward`: all four unbounded member logits, so unchanged native TRAIN applies the existing mean positive/negative log-sigmoid loss.
- `forward`: arithmetic mean raw logit with native `[Q,1]` shape, so unchanged native VALID preserves its row groups, negative flattening, ties and rounded metrics.
- Native unweighted `SparseTensor` CSR/COO adjacency converted into the qualified query graph without a dataset loader, neighborhood cap, candidate split or additional draw.

The native body can retain its sampler, permutation, record masking, single encoder pass per batch, positive-then-negative calls and Adam. The full policy is 36 updates per 1,024-row epoch, dropping the shuffled 812-row TRAIN tail; 512-row VALID groups retain all 2,216 rows and all 500 candidates each. The maximum is 9,999 epochs, evaluation every five epochs, no initialized VALID, first strict rounded-MRR maximum, and stop at `kill_cnt > 10`.

Source algebra gives 613,903 total shared4 parameters and 544,778 active parameters, including the honestly retained unused fixed-pt `ptlin` state. Native Pubmed NCNC has 591,364 total and 525,315 active parameters. These counts still require runtime parameter-name/shape authentication. Private versus pooled has identical capacity; contrasts against single baselines also change factorization, member serving and capacity.

## Six fresh fits, three paired seeds

Prespecify private and pooled-after-clamp arms for seeds **0, 1, 2**. Fresh constructors use the same native seed point, shared weights and decoder layout. The established sorted r/s sign assignment uses a local CPU Generator and a fixed Pubmed task domain; it does not alter the global TRAIN RNG.

The only arm difference is each member's own clamped completion weight versus the arithmetic member mean **after** the nonlinear clamp. Recursive scores remain under `no_grad`; each receiver's outer feature transform and final decoder remain private. Both arms serve the same mean of raw logits.

Require equal initial full state/RNG and RNG-neutral digests of the actual native negative draw, full permutation including its tail, batch records and epoch boundary RNG through all jointly executed epochs. Do not introduce per-epoch reseeding or substitute an explicit stream. Different selected and stop epochs are permitted by the native policy; compare stream equality only on their executed overlap. Matching baseline seed labels does not imply identical baseline/shared4 streams because four decoder paths consume additional dropout draws.

All six earlier baseline cells remain immutable scalar comparison context. No baseline, qualification, engineering-resource, replay or partial-fit state may initialize these fits. Report every seed delta (pooled minus private), its mean, median, sample SD, signs and minimum, plus per-arm scores/epochs and all attempt costs. Three blocks on one released graph/pool are exploratory.

## Remaining prerequisites

1. **New bridge/source review.** Review this bridge and the future six-fit driver, arm binding, native loss/serving/state/selector integration, source pins and accounting. Existing baseline releases authorize no extension.
2. **New geometry parity.** Small fresh in-memory checks for the 500-to-256 puregcn/active-JK path, Pubmed clamp settings, unit-factor native decoder limit, native adjacency conversion, masked/duplicate records, empty/asymmetric neighborhoods, logits/loss/gradients/Adam and full state restoration. Reuse completed generic decoder/routing proofs. No engineering state becomes a scientific donor.
3. **Pairing/observer integration.** Authenticate fresh initial state/RNG for all three seeds and a reviewed observer of unchanged native negative/permutation bodies. Check selector first ties/stop and selected-state mode binding with fabricated bookkeeping.
4. **New Pubmed resource envelope.** A separately released engineering-only seed-0 pair of five complete native TRAIN epochs per mode, then their first full native VALID serves, gives 360 total updates and two complete VALID serves. Include fifth-epoch state replay, inclusive wall/RSS/physical occupancy/CUDA peaks and candidate-work counts. Preserve exact batches, float32, no degree caps and `splitsize=-1`. No quality threshold or score-based configuration choice. Existing native/Collab timings do not establish this geometry's feasibility.
5. **New scientific driver and release.** Freeze new output paths, stages, source manifest, operational caps and approved v3 supervisor binding after admission. Preserve native fit policy and separately released selected-state replay. All six fits and replay PASS plus supervisor COMPLETE are required before declaring completion.

Prospective parity arithmetic reuses the existing elementwise `atol=rtol=128*float32eps` rule (1.52587890625e-5) for numerical native/bridge comparisons. State restoration, input/order/support, paired RNG/streams and replay per-query/rounded metrics require exact equality. No tolerance amendment after an observed mismatch.

The proposed scientific resource caps copy the completed native six-fit precedent (six hours, 70 GiB allocated, 75 GiB reserved, 32 GiB RSS); they are not a feasibility claim and require a new admission/release. A breach is incomplete. Retain all outputs and incurred costs; no automatic retry, resume, reduced batch, degree cap, candidate split, shortened schedule or precision switch.

## Verification and custody

Only stdlib AST parsing and metadata/hash checks were performed. Numerical/runtime parity is **unverified**. `SOURCE_BINDINGS.json` hashes 32 existing sources and scalar receipts, including the audited baseline summary and completed qualification evidence. No model/data payload was opened. `MANIFEST.json` seals this new packet; prior packets remain untouched.
