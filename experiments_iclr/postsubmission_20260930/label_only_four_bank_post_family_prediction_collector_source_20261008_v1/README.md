# Disabled v2 post-family prediction collector

Source preparation only. The collector requires an exact enabled root release
declaring whole-family opening, the v2 `FAMILY_CLOSURE.json` hash and all twelve
existing selected-state hashes in seed/arm order. It checks three successful
closed 1100-epoch blocks and twelve complete correction records before loading
numeric roles or selected states. The published collector commit must descend
from the completed scientific execution commit. The shipped release is disabled.

`run(release_path=..., release_sha256=..., later_execution_authorized=False)` and
the CLI refuse by default. Root supplies the seven variable release fields:
`enabled`, `source_review_approved`, `root_whole_family_opened`,
`collector_manifest_sha256`, `execution_source_commit`, `family_closure_sha256`,
and `selected_states`. Each selected row is exactly `{seed, arm, sha256}` for
6101/6203/6307 and C4/S_joint4head/U4_sharedB/S_one_path. Runtime, sole GPU,
original complete numeric roles and source custody reuse the sealed v2 owner.

Following the bound `SOURCE_API_MAP.md`, each arm's own coherent
`arm_final_selected` state is reconstructed once. One explicit `SERVE` capture
and native forward supplies both its native logits and `serve_banks(H,base,ids)`
on all 5274 development IDs. Total: twelve reconstructed native/corrector states,
twelve native forwards and thirty-nine correction attention branches. C4/U4
have four prediction members; both singles have one. Constructor/restore calls
and observed Adam objects are charged: twelve native Adam objects and twenty-one
corrector Adam objects on complete collection. No optimizer update, backward,
TRAIN call, reselection, calibration, mask draw or new gate is performed.
Internal partial constructor work on a failed API call is reported as unobserved.

The once-only output contains three CSV tables:

- `SAME_STATE_CORRECTION.csv`: 36 seed/arm/cohort rows; whole, structurally
  covered and no-visible-TRAIN-neighbor supports; native/corrected accuracy,
  stable NLL, Brier and member metrics; repairs, harms and wrong-label churn.
- `CO_PRIMARY_DECOMPOSITION.csv`: 18 seed/reference/cohort rows for C4 versus
  each co-primary; signed pipeline = native-epoch + correction differences,
  exact correctcount identity and retained floating-point residuals; across-arm
  flows compare corrected endpoints with potentially different native epochs.
- `ROUTE_COMPLEMENTARITY.csv`: 18 C4/U4 seed/cohort rows; disagreement, any-member
  correct coverage, pooling benefit, pooled-only rescues, pool harms and strict
  common wrong rivals. Hidden joint-single heads are never treated as members.

Coverage uses only incoming nonself edge records and permitted TRAIN source IDs.
No-neighbor correction logits must equal their own native logits exactly;
probability equality allows the fixed 2e-7 rounding tolerance. Full selected
correctcounts must reproduce saved endpoints; failure preserves all artifacts
and does not trigger another selection. Accuracy differences use percentage
points; NLL/Brier differences retain their original signs. These diagnostics
describe consumed development endpoints and do not revise the frozen gate.

`SERVER_ONLY_RAW/` retains native/member logits and native/served probabilities
in the allocation output; its parent is private and raw files have mode0600.
Only compact CSVs and receipts are intended for local collection. Runtime writes
`RUN`, work-only `PROGRESS`, `COMPLETE` or `FAILURE`, and `COST_TERMINAL` receipts
with exact state/raw/table hashes, wall/CPU/RSS/CUDA/storage costs. All partial
files survive failures. One process is bounded by a 3600s alarm,32GiB GPU/RSS caps
and36GiB fresh GPU admission; Mol18 coexecution is allowed. No retry or resume.

Native-own-best reconstruction for future C&S is explicitly separate and
deferred: the existing screen API does not accept that state kind. This collector
neither forges a checkpoint kind nor acquires another native model. Attention
histograms/compatibility maps are deferred. Existing sources remain unchanged.
Preparation used AST/compile/JSON and metadata hashes only: no numerical imports,
fixtures, datasets, checkpoints, outcomes, execution or server access.
