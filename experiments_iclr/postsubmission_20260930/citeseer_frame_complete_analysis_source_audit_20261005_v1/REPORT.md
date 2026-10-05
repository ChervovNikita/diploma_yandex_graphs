# Static audit of the complete Citeseer development analysis

The reviewed analysis implements the frozen seven-family, three-block development comparison. No score, prediction, checkpoint or actual fit-history payload was opened for this audit. No server, scientific-code import, numerical experiment, GPU operation or source repair was performed.

The analysis source, protocol, plan and retained training manifest have matching SHA-256 bindings. The retained manifest's seven payload files also match their recorded sizes and hashes. The findings below concern the completeness of the analysis audit, not an observed training error or predictive outcome.

## Two repairs are warranted before execution

1. **Reconstruct the first native stopping event.** `analyze_valid.py:114–115` only checks that the last epoch is 55 epochs after the selected epoch, with an unconditional exception at epoch 9999. That is necessary for an ordinary early stop but does not establish that the runner stopped at the *first* eleven consecutive non-improving validation checks. For example, a history can have a best value at epoch 5, eleven misses through epoch 60, a later improvement at epoch 65 and eleven further misses through epoch 120. The current test accepts selected epoch 65 and last epoch 120, even though the retained runner would have stopped at epoch 60. The max-9999 branch can likewise hide an earlier stopping event. Reconstruct `best_valid=0`, reset on strict improvement and increment on every other entry; require the first miss count above ten to be the final history entry and final training epoch. If no such event exists, require the full 9999-epoch termination. This is a static counterexample, not an actual history or experiment.

   Relevant source: analysis lines 107–115; retained `run.py:286–312`; pinned HeaRT donor runner lines 525–538. The existing first-maximum check at analysis lines 110–111 is correct for the chronological history list. The analysis should also check that history selectors are native four-decimal values, as the retained `metric` returns rounded scores at `run.py:177–181`.

2. **Preserve preflight failures.** Analysis lines 62–88 perform host/path/GPU binding, plan and cohort authentication, protocol checking, all terminal hash checking, and `START.json` creation before entering `try` at line 89. Any exception in that region is raised without the `FAILURE.json` receipt written by lines 175–177. Existing input bytes remain intact, but the promise to record every analysis failure is incomplete. A failed attempt also cannot be distinguished from a quiet incomplete-cohort return solely through the analysis directory. Preserve a failure receipt for authorized preflight failures while retaining the normal incomplete-cohort return and existing files. Wrong-host/path failures must not cause writes on an unauthorized host. If a wrapper owns that receipt, document and bind its exact source.

## Full-cohort gate and family reconstruction

The outcome gate has the required order. Lines 65–78 authenticate the plan, require a complete cohort freeze with the fixed plan/source identities and all 36 distinct planned fit IDs. Lines 83–84 authenticate every physical `FREEZE.json` hash before the first prediction/history/checkpoint hash or semantic read at lines 94–116. Numerical imports occur only after that gate. An incomplete-cohort return reads owned progress metadata only.

There is a minor strictness gap in lines 77–78: creating a dictionary can collapse duplicated registry entries. The checks enforce the required 36 unique IDs, but do not enforce that `completed_physical_fits` itself has exactly 36 rows with no duplicates. The queue producer naturally emits one row per fit; explicitly checking raw registry length and uniqueness would make the stated “exact 36-fit” authentication complete. This is not an outcome-access leak and does not omit any of the required 36 authenticated terminal files.

The ordinary ensemble reconstruction is correct. Lines 134 and 137 reuse each block's prospectively fixed native single as ordinary member zero, then use members 1–3 at seeds `base+5*m`. Lines 135 and 137 reconstruct the four separately fitted framed members at the same seed offsets and axes fixed by the plan. Every other family uses its own one physical fit, including the four-member banks whose saved validation arrays already contain the member mean. Lines 139–140 average only ordinary/framed independent constituent arrays. No bank is accidentally averaged four times, and no better aggregate epoch, member, coefficient or seed is selected. The plan's 12 physical fits per block resolve to seven served families per block.

The selected arrays have complete shapes `(227,)` and `(227,500)`, float32 finite values and consistent input identities. Midpoint tie ranks match the retained native evaluator. The original runner validates queries in sequential order with complete tails, and all fits share identical input identities; this is sufficient for the fixed component-wise pooling used here. It is still reliance on the reviewed writer and identity bindings, not an independent inference replay from each checkpoint.

## Hash coverage and limits

The analysis authenticates plan bytes, its own source through the protocol, all physical freezes, each fit's config, history, selected checkpoint and selected validation arrays. It cross-checks every planned row field, input identity, selected epoch and rounded score, then records hashes of the served prediction and per-query outputs. This is substantial custody coverage.

The analysis compares the recorded source-manifest identity to a pinned constant but does not rehash the source manifest and its payloads itself. The retained training runner already did that before every fit at `run.py:64–71`; the local static audit independently verified those retained source bytes. Thus the analysis's source authenticity claim depends on that reviewed writer. It should not be described as independent checkpoint replay or an independent audit of every native operator.

The fit writer records a job-file hash, but the analysis does not verify that job file or its hash. It verifies the job embedded in the authenticated config against every planned row instead. This is sufficient for the named arm/seed/axis/factor/member/block fields and leaves ancillary job provenance dependent on the reviewed queue/runner. No observed mismatch is asserted.

## Statistical interpretation

The three-block summaries and paired contrasts match the fixed plan. The descriptive `t(2)` intervals are explicitly conditional on one graph/split and a normal seed-variation approximation. They are not graph, split or query-population confidence intervals. The sign-flip calculation enumerates all eight sign patterns, states its sign-symmetry assumption, and acknowledges the minimum attainable two-sided p-value of 0.25. Holm correction covers the four predeclared primary contrasts; with this minimum and four comparisons, all Holm-adjusted primary p-values must be 1. These tests cannot substantiate a statistically significant improvement.

This remains a validation development screen because validation chose every checkpoint. Per-query output is useful for later diagnostics but cannot supply 227 independent graph-level replicates. The saved limitations explicitly retain the one-split, three-block, concurrent-work timing and lack-of-novelty/publication-readiness boundaries. A positive contrast can motivate separately frozen heldout confirmation and broader paired work; it cannot establish the requested superiority or acceptance by itself.

## Failure custody

The queue source uses fresh fit folders, preserves child stdout/stderr and artifacts, stops on a nonzero child exit, records `QUEUE_FAILURE.json` and does not retry or substitute seeds. It emits the cohort freeze only after every retained fit succeeds. Inside the analysis `try`, errors record `FAILURE.json` and retain partial outputs; exclusive JSON creation avoids ordinary overwrites. Preflight failures remain the explicit gap above. `KeyboardInterrupt` is also outside the analysis's `Exception` handler, so interrupted attempts retain bytes without a complete failure receipt. No failure evidence or actual cohort terminal was inspected in this audit.

## Scope

Only retained source, protocol, plan, admission metadata and manifests were inspected. No actual scores, histories, prediction arrays, checkpoints, data/model payloads or remote state were read. The root should preserve this audited source version and make any repair in a separately identified successor before outcome access.
