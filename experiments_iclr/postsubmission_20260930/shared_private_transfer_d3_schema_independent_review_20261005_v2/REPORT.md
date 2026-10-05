# D3 v2 — targeted repair re-review

## Result

**All three prior findings are resolved within the requested repair scope.** No remaining correction is identified in the checked wording. This is a source/Markdown repair review, not an expanded audit, analyzer review, data/custody verification or execution approval.

Reviewed assessment manifest `b25b3805394fa8bc5867ba421feebcc6975d4b358601fb91356e6ec223b79c2b`, report `3ea06ac73e1f3ea8b5695fb69c7c2963bed4aaaddb24328afaaafaa7e857c40b`. Its four manifest entries match their hashes/sizes. V2 retains byte-identical producer input bindings and execution-disabled metadata from v1. The preserved v1 independent review supplied the producer-source evidence; no expanded source audit was performed.

| Checked item | Repair assessment |
|---|---|
| P2 capable-single loss schema | Resolved. V2 specifies one union loss per pass, with separate positive/negative normalization, and explicitly says four historical stream-specific values cannot be recovered. Four logged stream losses remain specific to F4/row0's four-call branches. |
| P2 cycle/final CUDA peak cutoff | Resolved. V2 labels the maximum cycle value as a recorded pre-VALID snapshot, states subsequent/final VALID can raise the final FREEZE peak, and reports the final retained peak separately. Cumulative-after-reset scope and the prohibition on summing peaks remain explicit. |
| P3 raw self-loop handling | Resolved. V2 distinguishes loader filtering of raw self-loop rows from duplicate nonself rejection and graph/support validation. |
| Added TRAIN-only forward-counter scope | Correct. The producer increments these counters from TRAIN episode receipts; VALID forward calls do not increment them. V2 now explicitly excludes interpreting them as total fit-forward counts. |
| Added ordinary loss scaling | Correct. Four-route ordinary inner losses are summed and the outer objective is multiplied by four; shared-four/single controls retain their source-defined normalized/mean objectives. V2 warns against direct raw cross-arm magnitude comparisons. |

No target imports, numerical/function/fixture execution, history/FREEZE/model/feature/score payload reads, network/server contact, existing-packet edits or extra agents occurred. Missing full39 data remains the assessment's stated later prerequisite and was not treated as a defect. V1 assessment and its review remain preserved.
