# Bounded v3 roles and runtime rereview

Immutable source: `learnable_internal_be_contrastive_multitask_suite_20261007_v3`, MANIFEST SHA256 `cbbb1396ddd1d5479c7b98ef4d709b6e36c19cb55daef8c26bcf76ee883ea4b9`. This review preserves both preceding reviews and checks the requested R7–R10 corrections without a requested favorable verdict. It is source-readiness review, not execution, scientific acceptance, or whole-goal completion.

R7, R9 and R10 are closed. R8 now bounds cleanup inside the admitted absolute deadline, but one narrow failure-classification defect remains: an observed cap overrun can still be reported as complete. `REVIEW.json` therefore has `approved: false` for this exact source. Correct that success condition in a separately sealed successor before fit-source approval; no broad additional gates or scientific run are requested.

## Exact evidence

All27 manifest files match sealed hashes and sizes, all12 Python files parse, and all three native dependencies retain their exact hashes. The core loader, runner, models, selection helper, objectives, exporter custody plan, provider/runtime pin, dependencies and staging source are byte-identical to reviewed v2. Modified runtime/supervisor/export boundary/test/budget/protocol text was inspected directly. The source/stage/30-record CPU evidence is bound to this v3 manifest.

No framework/model imports, data/label/checkpoint/logit payload reads, GPU work, remote access, source edits or CPU numerical reruns occurred. The previous limits on actual remote export authority/raw custody verification still apply. BINDINGS.json and READ_SCOPES.json record these boundaries.

## R7: genuine measured resource fields — closed

`runtime.resource_measurements:147–157` requires `type(peak) is int`, positive bytes within80GiB, actual int/float seconds, positive finite time bounded by46800, and genuine integer identity member/view counts. Boolean fields therefore cannot masquerade as bytes/time/counts; NaN/Inf and out-of-domain values are rejected. `admit:111–113` calls that helper and checks exact genuine work member/view counts against the bound workload identity.

The existing CPU source calls the actual helper with a valid metadata case and invalid Boolean, nonfinite, zero/negative, fractional/string cases. Those calls are meaningful source evidence for the repaired guard; they do not constitute real measured GPU memory or time. The80GiB source ceiling is an admission domain, not verified available memory or feasibility for a full two-view four-member graph.

## R8: deadline wait bounds repaired; terminal success gap remains

The job/config fixes10seconds of cleanup inside the absolute hard cap: WikiCS/molecule32390 active+10cleanup=32400, Collab46790+10=46800. Worker admission requires genuine integer active/grace fields, their sum, and exact config active budget. Parent receipt binds them with source/job/identity/deadline custody.

`supervise.py:39–53` stops active computation at the earlier active deadline. TERM wait is bounded by at most5seconds and time remaining to hard deadline; KILL/reap wait uses only the remaining time. It has no former unbounded final `wait()`, and unreaped timeout is preserved as `hard_timeout_unreaped`. Parent elapsed timing includes launch/preparation. The CPU cap check verifies config arithmetic; it does not execute actual timeout/kill behavior. Static source confirms these bounded waits.

The remaining defect is at54–67. For an exit-code-zero worker, status becomes `complete` at56. The finally block then computes `cap_exceeded` at63 but does not change status. The final exit test checks only exit code/status. If scheduler delay causes the parent to observe a completed worker after the hard deadline, it can write `status: complete, cap_exceeded: true` and exit zero. This contradicts PROTOCOL.md:80's explicit statement that an observed scheduler/cleanup overrun is never a complete fit.

Compute elapsed/cap/reap facts once and normalize any observed overrun or unreaped child to a failed terminal status before serialization and process success. Preserve the actual inclusive seconds and cap/reap facts. This is a concrete success-path correction; deterministic OS timing or bitwise floating-point identity is not required.

## R9: discriminating Adam restoration fixture — closed

The test now makes the saved ordinary histories differ by7*(m+1), then perturbs every live Adam tensor by113 after snapshots. Both ordinary and packed untied branches compare parameter groups and every saved optimizer tensor, including moments and steps. Ordinary model snapshots also differ from the synchronized joint bank. Missing or wrong optimizer restoration would now fail these assertions, and both branches still assert body/global mode and live CPU RNG preservation.

The runner/helper source remains unchanged and already restores own checkpoints only for ordinary independent4 and pooled model/all Adam states for the coupled packed untied arm. These30 CPU records include two meaningful selector custody cases. They remain tiny synthetic CPU tests, not CUDA stochastic trajectories, data competency, full epochs, or complete resource qualification.

## R10: exact molecular boundary — closed

Exporter documentation/stdout/receipt now use `TEST_target_values_parsed: false` and explicitly record `all_graph_public_count_metadata_parsed` for molecules. PROTOCOL.md:82 declares that all41127 public node/edge-count rows support offset location, are not fitted feature statistics/targets, and only selected TRAIN/VALID target/atom/bond/edge values are parsed. No stronger zero-numeric-metadata-traversal claim remains.

The underlying molecular code is unchanged: it streams global raw gzip bytes, parses selected TRAIN/VALID rows and complete count metadata, and emits only those two roles. Original atom/bond categorical fields and local topology are retained with reciprocal edge/bond interleaving. This precise source boundary still requires separate data-only authorization and actual export custody review. No export ran in this review or in the saved source preparation evidence.

## Preserved admission/export invariants and scientific scope

Byte-identical v2 core retains safe NPZ ZIP/header checks before numeric arrays load; hard official role counts/full schema domains; exact public WikiCS features/topology; TRAIN/VALID role disjointness and collab canonical support exclusion; every-member full-label two-view supervision and complete tails/full horizons; canonical repeated/reversed link-target positives; finite member/pool/representation/parameter/Adam checks; ordinary independent own selectors/mixed serving modes and synchronized coupled selectors; persistent dropout streams/live transition RNG; closed-score complete validation and pooling definitions.

Exact source/config/export review, full-work resource identity, current allocation/runtime/provider path/hash, genuine work member/view fields and live parent PID/start ticks/job/source/deadline binding remain admission requirements. All three fit and three export templates retain false authorization/source approval defaults. No active queues, automatic retries or next-family release were added. The closure text now consistently opens comparisons only after all24 cells in a separately adopted full task family complete or preserve failures; the72-cell matrix is maximal, with later family adoption separate. The method reviewer assesses that correction independently.

Concrete exporter source exists for the authenticated safe WikiCS input, original authorized Collab archive allowlist, and original raw/scaffold molecule data. Actual official exports/hydration/output receipts and independently approved exact output custody are still absent. Source hash checks are not acquisition or data-role audit results.

Actual complete representative resource qualification remains external work: each admitted arm/config/data/runtime/member/view workload must demonstrate full TRAIN backward/Adam, complete VALID and checkpoint serialization with closed metrics, and WikiCS local/global modes, plus measured memory/time. CPU/staging receipts do not certify retaining full graph activations or completing native horizons. Family cost/failure/missing-cell custody and root adoption/release remain required. No backbone/horizon/population shrink or seed/arm pruning is allowed to rescue failure. Nominal caps are ceilings, not forecasts or efficiency measurements.

This source admits only `anogena-2-0`, exact GPU UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac`, exact repository/phase/cwd, existing interpreter and installed provider path/hash pin. It is not a77 port. A future77 execution needs a separately reviewed host/device/runtime/source/data/resource binding; no77 access or readiness claim is supplied.

No TEST scoring wrapper/release or unused-population result exists. Evidence-response/common-error diagnostics, confirmation, and capacity/compute matching remain future work. A matched original GNNM/boundary-projector port is prospectively required before attributing improvement to internal versus boundary factor placement; the present primary contrast tests the fixed contrastive addition. No predictive improvement, efficiency, novelty or broad-validity conclusion follows from source readiness.

## Recommendation

Preserve v3 and repair only the observed-cap-overrun success classification before source approval. Then use a fresh exact review to begin separately authorized official export audits and full resource qualification on the current allocation. Scientific fits remain disabled until all concrete source/data/runtime/config/workload/supervisor/family conditions are satisfied. Future77 and TEST work remain separately unimplemented/unadmitted.
