# Independent source audit: post-all21 reader

## Verdict

**V1 is ineligible for scientific readout.** Two source validation gaps were repaired in a separate, disabled V2. No incorrect result or outcome-dependent choice was observed; outcomes were not opened. V1 and all frozen training sources are preserved.

## Required fixes, now implemented

1. **P2 — Join predictions, histories and costs to the exact completed attempts.** V1 `analysis.py:213–230,255–259` checks the supplied reference/result pair but never compares independent checkpoint hashes/own selected epochs to the original COMPLETE bodies whose costs it sums. It also omits an exact COMPLETE/origin anchor for vanilla and pool results. A same-seed, same-seal file from another execution could pass. V2 binds the original18 origin receipt reused by centered3 and the final `CENTERED_RECORDS.json`; indexes exact kind/base/member records; checks expected execution paths and origin hashes; matches every independent reference checkpoint SHA, own selected epoch and seed to its charged complete body; and anchors every own history. Cost summaries use those completed records.

2. **P2 — Enforce the frozen input anchors.** V1 `analysis.py:54–59,83` checks a release-supplied protocol hash without requiring the known frozen protocol SHA and does not enforce the original18 directory named in the exact frozen scope. V2 requires both anchors and the expected closure filenames. Scientific rules are unchanged.

Final result JSON is compared to the completed record by a canonical JSON provenance hash. The sealed shared producers persist `counters` in RESULT.json separately from the record they return to COMPLETE/CENTERED_RECORDS; `counters` is the sole field omitted for this comparison. This adds no metric recalculation or float parity test.

## Scientific design retained

- All four frozen vanilla/centered pool versus independent4 pool/member0 contrasts and every paired seed are reported. Member0 comes from the same fresh independent pool reference.
- Every member and served pool retains TRAIN/VALID AUROC, NLL, accuracy, Brier and selected epochs; mean/worst own quality and complete comparable eval-TRAIN curves remain. All original independent fit/serving costs, centered final records, outer execution costs and differing serving residency remain visible.
- Error intersection, any/unique correct coverage, rescue/harm and confidence/NLL subsets cover every role row. Error/rank pairs remain within each bank/seed; all positive-negative pairs, including ties, are visited.
- All three paired differences, sample SD and descriptive df2 t95 intervals remain. These describe optimizer-seed variation on an already-used development split, without confirmatory p-values or graph-population uncertainty.
- All21 plus exact actual parent/worker/group/birth/boot/CUDA closure still precede numeric/reference access. No selected examples, threshold tuning, best-of-two primary or causal geometry claim is introduced. Frozen tied-path/capacity controls and independent task/split confirmation remain necessary.

## Verification and release requirement

V2 passed AST and fixed source/protocol byte-hash verification, including sealed own payloads. Scientific diagnostic functions and the contrast/gate/component block are AST-identical to V1. Both release and custody templates remain disabled. No result/score arrays, model checkpoints, remote state, or numerical/model execution were accessed.

Before a later root release, fill V2's two additional bindings: `original18_origins` and `centered_records`. V2 rejects a mismatch; it grants no early or partial readout.

## Exact packets

| Packet | Manifest SHA256 | Analysis SHA256 |
|---|---|---|
| V1 | `adaa808ef200f7496d0576ca30868ad437621375ec371424b8449376ed179b73` | `e8d0ca488e8cfb6f862303c9d3736a8453f3f857dbbeb2c901e71fa92e13d1ae` |
| V2 | `8fb9e551cc3dc0dcb8e9311467d5e2a7a8aa460cc1da73b4bf3ca677d3b94405` | `4af0e1e5590ed0b5bfe5c0f2d49f9a660509a6da714786261c36b286cc54fd7a` |

V2 seal SHA256: `37e58f4fb12e1d99c62bea4329a343f988919c10fe1168533c39b28accb625e9`.

See `AUDIT.json` for source anchors, findings and the full static verification record.
