# Static follow-up audit of Citeseer analysis v2

This addendum is bound to final `analyze_valid.py` SHA-256 `4006920c39200f9eb93ecffaaf55718be3917fe579295e9bbd4ba7045d9f8cee`. The v1 audit and source are preserved. Only source/protocol/manifest bytes were inspected; no actual histories, scores, predictions, checkpoints, server state or scientific execution were accessed.

## Findings resolved

- **Exact native selection and first stopping event:** `selection_audit.py:5–31` now verifies the every-five-epoch history, finite positive four-decimal MRR selectors and native schedule. It reconstructs strict-improvement resets and every miss. An entry after the first eleven-miss stop raises immediately. The final epoch must equal that first event; if none occurred, it must be epoch 9999. The selected epoch and rounded maximum must match the freeze, and updates must equal three per epoch. This closes the v1 continued-past-native-stop counterexample and the max-epoch bypass. The helper is called before array loading at `analyze_valid.py:118` in the final source.

- **Preflight and interrupted failure custody:** `analyze_valid.py:65–70` starts with `authorized=False` and sets it only after cwd, hostname, script path and singleton GPU UUID are verified. The gate and START creation are inside the failure handler. Lines 178–185 catch both ordinary exceptions and `KeyboardInterrupt`, retain stage/timing/zero-update metadata, write a fresh failure receipt only on the verified route and otherwise print it for outer client custody. The earlier wrong-host write risk is closed.

- **Exact raw cohort registry:** lines 83–85 require exactly 36 raw registry rows and the 36 unique planned IDs. Duplicate or extra rows are rejected before outcome access. Every planned fit's terminal hash is checked at lines 93–95 before any history, checkpoint or prediction access.

- **Helper custody:** lines 89–92 authenticate `ANALYSIS_MANIFEST.json` through the protocol and hash its analysis/helper payloads. The final protocol, source and manifest bindings agree. No helper execution or regression-result certification was performed by this auditor.

## Warranted conclusions and remaining limits

No further concrete correctness defect was found in the final static source. Ordinary member-zero reuse, the three prospective seed blocks, all seven family constructions, individual independent-constituent selection and mean-raw-logit serving remain consistent with the frozen plan. The incomplete-cohort branch still returns using progress metadata only; it imports no scientific library and opens no outcome payload.

This conclusion is limited to the reviewed source. It does not certify actual cohort completion, actual selection histories, numerical outcomes or independent checkpoint inference. Source-manifest authenticity still relies on the reviewed training writer; ancillary original job hashes are not independently rechecked by the analysis, although all prospective row fields are checked inside authenticated configs. These are unchanged, disclosed custody limits rather than observed mismatches.

The screen remains validation-selected development on one fixed graph/split with three blocks. Exact two-sided sign-flip p-values cannot be below 0.25; Holm correction of four primary comparisons therefore produces adjusted p-values of 1. Descriptive seed intervals do not establish graph, split or query-population generality. A positive development outcome would require separately frozen confirmation; it would not establish novelty or manuscript acceptance.

No source was repaired by this auditor. All v1 findings, the interim v2 pre-authorization write warning and the final resolved status are preserved in this addendum's machine-readable findings.
