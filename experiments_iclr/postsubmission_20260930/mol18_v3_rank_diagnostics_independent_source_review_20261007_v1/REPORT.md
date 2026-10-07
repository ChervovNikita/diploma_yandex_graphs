# Mol18 v3: independent source correctness review

7 October 2026. Reviewed the added `rank_repair.py` functions and contrast/error paths independently. **No material source correctness defect identified in the reviewed additions.** This is a source assessment, with no scientific result or publication verdict.

## Findings

- **PR and proper scores (`rank_repair.py:97–172`).** AP groups exact raw-logit ties and uses the step precision–recall integral. The separately labeled trapezoidal area uses the stated recall-zero/precision-one endpoint. Brier and class-specific/overall BCE use the same labels and each member’s raw logits or the original served mean logit. The float64 sigmoid is stable for finite float32 scores. Ten fixed probability bins include probability one in the final bin; empty bins are null, and ECE/MCE use positive-event frequencies.
- **Paired diagnostics (`rank_repair.py:336–371`, `protocol.py:136–145`).** All six fixed contrasts keep the three optimizer seeds and the existing df2 interval. Differences use left minus right; lower-is-better metrics retain that direction, and member worst competence selects the appropriate minimum or maximum. Missing seed cells return unavailable summaries. Empty O common-inversion cohorts give null rates and unavailable paired attribution, rather than zero repair evidence.
- **Exact O identity (`rank_repair.py:30–84,175–203,299–321`).** The hash serializes signed little-endian int64 original-ID pairs in positive-major order. Candidate IDs/labels, baseline positions/IDs, state domains and identity schema/hash are checked. A malformed candidate comparison changes its cell to `analysis_failed`, excluding it from paired summaries. All recorded analysis errors prevent complete-family success.
- **Coverage and pooling (`rank_repair.py:204–275`).** On the strict O common-inversion mask, candidate strict wins are acquired coverage. The joint coverage × pool-state table separates full wins, half-credit ties and pooling losses. All nine rank transitions retain gains, harms and both cohort/global normalization. Same-slot loss/tie/win transitions examine every positive–negative pair, avoiding the weaker inference from unchanged aggregate member AUC. Independent-bank slot matching is labeled arbitrary; unchanged pair orders permit scale explanations without identifying an actual scale intervention or causal graph mechanism.

## Preserved contracts and evidence limits

Direct comparison confirms byte-identical v2/v3 `protocol.py` and identical collector executable AST after removing the docstring. V2/v3 seals, v3 input bindings and bound dependency seals match. No model call was added: the diagnostics reuse the existing raw archive and O freeze. The existing checkpoint, closure, serving and release contracts were not rewritten in this review.

Root’s bound qualification receipt reports all 20 artificial helper checks passed, zero serving calls and no scientific input/checkpoint access. I inspected that receipt and the fixture source; I did not rerun it. The fixture does not execute the complete collector/analysis orchestration or establish actual role/checkpoint custody, restored-model serving or scientific runtime correctness. Those remain root-controlled qualification work before an enabled readout release.

No source changes, role/checkpoint/live-output reads, remote access, numerical imports, job execution or publication were performed. `STATIC_CHECK.json` records the independent stdlib integrity/AST checks.
