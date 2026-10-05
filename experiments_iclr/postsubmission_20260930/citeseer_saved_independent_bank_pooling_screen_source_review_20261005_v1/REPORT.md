# Independent static review of saved-bank pooling screen v1

Source-only assessment on 5 October 2026. No analyzer, transport, fixture, score payload, model, dataset or remote command was executed. The reviewed analyzer hash is `8c114844ab2fa511dad9198fe67a6415e68d6997e46a1658a0362399f819af33`. This review does not grant execution authority.

**Disposition: resolve the three admission/replay findings before execution.** Preserve v1 and prepare a new version. Two transport findings also need resolution or an explicit narrower qualification.

| Finding | Source location | Action |
| --- | --- | --- |
| P1: incomplete terminal authentication | `analyze.py:60–84` | The cohort and prior-result documents are pinned, but only twelve constituent `FREEZE.json` files are authenticated. README promises all 36 terminals before score access. Restore the complete 36-terminal canonical-path/hash sweep before the first score hash or load; retain the exact prospectively declared roster. |
| P2: exact raw replay not established | `analyze.py:119–134`; prior `analyze_valid.py:143–146` | The new concatenated 501-column bank reduction differs from the native separate positive/negative reductions. Mathematical equivalence does not certify bitwise floating replay. A scalar MRR tolerance of `1e-7` also cannot certify identical per-query ranks. Use the native reductions and compare all 681 ordinary-independent query ranks against a custody-bound reference, plus exact native MRR. Declare that additional reference in the new protocol. |
| P1: admission is not version-bound | `execute.py:65–74` | A file with `source_review_resolved=true` is sufficient, regardless of which sources or review it admits. Before contact, require this task identity and approved analyzer, protocol, transport-client and independent-review path/hash bindings to match the actual files. The current fixed analyzer/protocol constants alone do not bind approval to them. |
| P2: uncertain transport can lose its local receipt | `execute.py:80–88` | The outer SSH timeout/exception occurs before `TRANSPORT.json` is written. Preserve an exclusive local dispatch-intent marker before contact and a terminal timeout/exception receipt. Refuse automatic redispatch after an uncertain outcome. The remote child has a separate bounded timeout and new-directory refusal, which is useful but does not replace local custody. |
| P2: canonical staging-parent check is late | `execute.py:20–36` | The analyzer rejects aliased input/output parents, but the transport can stage files through an aliased `phase` parent before the analyzer starts. Check the existing phase parent resolves to its declared canonical path before creating the new owned root, then confirm the created root identity. |

## Checks that pass by source inspection

The candidate ranks are descending, complete 501-candidate midranks. Cumulative tie starts and reverse cumulative tie ends produce the correct average position, then scatter returns each original candidate's rank. Negative mean member midrank gives fixed equal-weight Borda, including ties. There is no candidate-query propagation or label-dependent Borda weighting.

The native evaluation formula is `1 + 0.5*(strictly greater negatives + greater-or-equal negatives)`, followed by float32 reciprocal ranks and their mean. Three fixed seed blocks and twelve prospectively identified independent constituents are used. Payload loading uses CPU and `weights_only=True`, with shape, dtype, finite-value, input-identity, checkpoint and selected-epoch checks. Original source/checkpoint identities are inherited frozen declarations; no original source or checkpoint is rewritten or loaded. Canonical input paths reject symlink aliases.

The interval oracle is valid for **candidate-specific scalar convex logit pools**: each positive can attain its member maximum and each negative its member minimum. It uses labels and separate weights for each candidate. It is not a learned predictor, a bound for one shared weight vector, a generalization result, or a ceiling for arbitrary nonlinear/hidden-state aggregation.

The three-block t interval and sign-flip calculation are descriptive, with their assumptions and minimum two-sided p of 0.25 disclosed. VALID already selected checkpoints; there is no independent confirmation or shared-bank result.

## Interpretation limits

Strict-majority-correct/raw-mean-wrong counts establish magnitude domination of pair margins, not nuisance scale or miscalibration causally. Rename them descriptively or keep that qualification explicit; Borda also discards potentially useful confidence. Pair counts omit ties and are not independent samples. Favorable Borda MRR supports this fixed independent-bank development diagnostic only. Clarify that checkpoints were validation-selected while all 227 released positive links were retained; the positive-query population was not selected by outcomes.

Only the fresh review subtree was written. No score/cache/history/checkpoint payload was opened or hashed, and no remote contact, fitting, TEST access, canonical-state edit or method promotion occurred.
