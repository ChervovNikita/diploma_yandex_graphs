# Independent static source review: selected VALID error analyzer V1

5 October 2026. **REQUIRES_SOURCE_CORRECTION**: one P2 constant-vector correlation defect; no other concrete defect found in the requested bounded scope. Verified manifest `8e27abd43676c3af9398a5808a03ed4049b3aa4c4e81e4b4c4added634579d96`, seal `da20696e401637c63b3bd075ff69b18f45e9546c3b638763f5206a6e7eb9eec3`, both Python source hashes, all seven payload hashes/lengths and bound V2 source-authentication hashes. Both sources parse without import. This review did not execute code, access data/score payloads or use SSH.

## F1 — Constant continuous columns can produce a false defined Pearson correlation

**P2**, `metrics.py:145–164`. Centering with `aa−aa.mean()` and `bb−bb.mean()` can leave a common nonzero FP64 rounding residue even when a represented probability/residual vector is exactly constant. The squared deviations can then be positive, so `den>0` and the code emits a defined correlation (possibly1) rather than its promised `undefined_zero_variance` null. Constant binary error vectors are safe; continuous probability and residual panels are affected. No actual occurrence or numerical test is claimed here.

Narrow repair in a fresh source version: explicitly detect an exactly constant represented vector, or use anchored centering such as `ac=aa−aa[0]; ac-=ac.mean()` and the analogous b operation. Preserve empty handling and all definitions/denominators; no arbitrary near-zero variance threshold is called for. Preserve V1.

## Verified contracts

| Scope | Static correspondence |
|---|---|
| Custody and boundary | Runner52–76 authenticates its own source seal/payload and pinned V2 source before stdlib custody import. Existing unchanged custody binds complete15/9 lineage and three own VALID packs before numerical decode. Metrics319–362 decodes canonical public graph/VALID mask and all own selected-logit banks, then slices each to6123 VALID rows before probability arithmetic. Single is independent member0 alias. No original model/training/checkpoint, TRAIN/control/TEST label or feature decode path. |
| Native scoring | Metrics52–73 uses raw FP32 logit argmax for singles/members; four-member native class uses FP32 softmax mean. Stable FP64 log-softmax/member probabilities/Torch arithmetic mean supply Brier and native log-mixture NLL. Precision discrepancies are retained. |
| Population and public degree | Metrics76–89,319–322 uses fixed bins of canonical unique undirected non-self degree, covering every VALID node. Class/degree subsets have explicit support; empty ratios are null. No prediction graph or feature-derived degree is introduced. |
| Confusions and signs | Metrics127–142 reports all25 confusion cells, class recall and precision with correct support. Metrics167–199 defines left-only correct as recovery of a right error and right-only correct as loss of a right correct case; gap is recovery minus loss over the full subset. Conditional denominators are right-error/right-correct counts, double fault, error union or the stated class support. |
| Member quality and pooling | Metrics223–249,275–291 decomposes bank accuracy gap into mean raw-member accuracy gap plus pooling-gain gap, with correct4N denominators and identity residual. Member-correct histograms and conditional bank accuracy preserve every count. |
| Brier ambiguity | Metrics253–260 uses the same FP64 p and q for mean-member Brier, squared deviation and pooled Brier. Metrics292–309 preserves paired pooled difference = member difference minus ambiguity difference, including residuals and empty subsets. |
| Other diagnostics | Fixed15-bin ECE includes closed probability1 endpoint; erroneous selected-class confidence, native FP32 top margin and FP64 true margin are explicit. Pair double-fault/Jaccard/conditional error metrics are correctly normalized. Pearson is descriptive population arithmetic; only constant-column handling needs F1 repair. |
| Output and interpretation | Metrics333–412 requires all nine3-by-3 panels before terminal complete, uses the same table rows for JSON/CSV and retains separate three-block mean/min/max. Failures are preserved. README/result explicitly prohibit fresh confirmation, node/split iid inference, causal collapse/information-absence inference, attainable oracle from any-member-correct, fitting/refit/promotion or TEST authority. No automatic intervention or oracle LP is implemented. |

## Handoff

Root alone executes after the fresh narrow correlation repair and its bounded source-delta review. This finding does not call for a new study, unrelated test gate or repeated original acquisition. The V3 projection audit is a separate result.
