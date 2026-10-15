# Final scoring integration source v2

This package connects the frozen `final_selected_scoring_v4` APIs to the actual
portable dispatcher and accepted scalar reporter. It is source preparation,
not campaign GO, native qualification or scientific results. No original
GNNM paper experiment is repeated. Original GNNM remains a baseline on the
new proposed datasets.

## Inventory and boundaries

- Training inventory stays **6,460 physical fits + 270 VALID I4 exports**.
- Final stage has **6,460 own selected-fit TEST serving tasks + 270 I4 TEST
  aggregations**, with **zero new training fits**.
- Reports retain **6,190 expected predictor rows**, including **270** same-bank
  mean diagnostics. Missing, failed and unstarted outcomes are explicit.
- A fit serves on the same immutable physical worker that trained it. The
  existing ownership is stricter than same-host grouping: each I4 and its four
  native bodies share one worker. Checkpoints/logits stay on their host.
- Cross-host exchange contains immutable JSON outcome shards and scalar
  reports. It does not require arbitrary model-state relocation.

This is the coherent repaired scorer4/inference-overlay source rebind described in SOURCE_REBIND.md. Native numerical/resource readiness remains pending. Frozen/published predecessors are unchanged. `SOURCE_BINDINGS.json` records
the actual parent bytes. `SOURCE_DELTA.md` enumerates the minimal successors.

## Callable preparation and execution

With this expansion directory on `sys.path`:

```python
from final_scoring_integration_v2.projection import project_ownership, project_release
from final_scoring_integration_v2.offline_seal import build_host_shard, merge_host_shards
from final_scoring_integration_v2.collect import collect_profile, merge_profile_reports
```

`project_release` requires every owned scoring task's explicit inference
binding. It preserves the original frozen task in the runtime request. Only
the scheduler dependency view changes: selected-fit tasks have zero final
dependencies; I4 tasks have their four exact final component dependencies.
External training dependencies are retained separately and sealed.

Production `run_gpu.py` accepts `--worker`, `--run-spec` and
`--run-spec-sha256`. Its original portable host guard runs before project
metadata/native imports; disabled77 refuses before probes. The final run
spec has schema `portable-final-reviewed-run-v1` and exact pins for:

- `training_run_spec`, `training_go`, final `release` and `readiness`;
- `global_decision`, `global_contract`, and derived `training_seal`;
- an optional reviewed `continuation` accounting for all retained STOPs.

The source manifest and final release/readiness are bound in the **same host
GO view used by that host's training**. No second user action is claimed.
No actual GO files are supplied here.

### Relation to one command per GPU

The existing portable training command remains the training entrypoint.
The final integration supplies its **separate final-scoring command** after
the global terminal barrier. It does **not** yet implement one combined
train-through-final command. Prepared combined orchestrationv3 sequences the existing TRAIN and this
FINAL command around the real offline barrier. Its actual process/restart
qualification remains pending; do not advertise runtime readiness from
source preparation alone.

Training supervisors must run detached on their servers so a Mac disconnect
does not stop them. All training outcomes must first become terminal. The
immutable local shards must then be exchanged and merged offline. That is a
real later synchronization step. No continuously connected Mac or online
coordinator is required. No complex waiting/orchestration layer is added here.

## Actual dispatcher reuse

The successor Engine retains the original launch/PID custody, run loop,
two-child ceiling, GPU/worker lease paths, STOP/drain behavior and terminal
failure retention. TEST results have their actual schemas and `TEST_export`
receipts; they are never relabeled VALID. A missing component creates a
scalar missing I4 outcome without a child, model forward or fallback.

The backend inherits actual admission/probe/identity/poll/wait/reap methods.
Its minimal `start` successor retains the original locked
reservation→Popen→identity-commit transaction. There is one Popen owner.
The fixed child entry reuses original lexical interpreter validation and
allows only this package's source-pinned final scalar observer.

The final phase uses the **same host ledger root and HOST.lock**. It preserves
original OWNER, claims, STOPs, archives and exact old STATE bytes. Only an
empty, fully sealed training ledger may transition to final ownership. The
old training loader refuses the final STATE ownership hash. An intermediate
crash before STATE replacement blocks final startup for review; the old
loader does not read the new marker, so do not claim that marker alone blocks
training. The already-terminal inventory and frozen prohibition on new
training resumes remain in force throughout this transition.

Campaign-wide validation happens once at supervisor startup. Each actual
startup reservation gets a pinned supervisor-preparation receipt and one
task capsule. The child validates that committed reservation, its own scalar
training chain and zero/four I4 dependencies. It never repeats full campaign
qualification. Frozen scorer v2 still reads its original roster/seal JSON and
does the actual complete selected VALID reconstruction before TEST.

## Inference resource admission

Inference bindings use the reservation fields consumed by the actual host
ledger, without pretending that inference has a mature optimizer. Their
measurements are explicitly VALID-only. A representative class measurement
may support many seeds/task mappings when a documented conservative shape
bound covers each target's frozen TEST population. Seed IDs alone do not
create resource classes. `representative_basis` distinguishes the real
measurement from derived target bounds; reference cost is not reported as
that task's observed execution cost.

Actual wall/resources/failures are collected from the final invocation and
terminal records. Two jobs share a GPU only if both peer profiles are
qualified and the original host-wide VRAM/RAM/disk/CPU admission succeeds.
Serial profiles remain serial. There are no assumed resource figures here.

## Offline global contract

`offline-final-global-contract-v1` binds `training_ownership_sha256`, the
path-independent `role_semantics` for all 27 presets, and every nonempty host:

```text
hosts[host] = {
  training_release: exact original-host receipt,
  scoring_roster: exact original-host receipt,
  GO_view_template: exact prepared local execution-view template receipt
}
```

Each template is `prepared-host-campaign-decision-template-v1`, with `host`
and `execution_view`. Its pending view has `user_explicit_GO=false` and
`human_instruction_evidence=null`. It omits
`parent_global_decision_sha256` and `final_execution.global_contract` because
those are later parent authorities; including them would create a hash
cycle. The global real decision binds the frozen contract hash and the
actual instruction receipt. The actual local view inserts only those parent
authorities/evidence and sets true GO; all other fields must equal the
globally frozen template. All host views are derived from the same human
decision, with no invented second authorization.

`build_host_shard` seals every local terminal outcome and embeds only exact
JSON source text with its origin receipt. `merge_host_shards` authenticates
copied bytes without opening foreign paths, verifies a complete disjoint
6,730-task partition and equal role semantics, then derives a local seal
compatible with frozen scorer v4 (unchanged v2 role/GO/roster/seal APIs). Its global outcomes include failures and
no-selected-state cases; own selected tensors stay with their physical owner.

## Scalar reporting and costs

`collect_profile` has a separate TEST normalizer. It does not call the
VALID-only `extract_runtime` or `extract_i4` helpers. It reuses accepted
`report()` for immutable cohort pairing, M1/body0 identity, diagnostic bank
identity, unresolved conditions and descriptive ten-cohort intervals. All
expected rows are supplied, including explicit missing placeholders.

Pass the accepted TRAIN collector's complete attempt cost ledger via
`training_cost_ledger`. If absent, the report says TRAIN cost is unknown.
Every final START/observer/terminal cost is retained. Observer wall and the
overlapping native invocation wall are not summed twice. Missing crashed
work is never inferred as zero. The original report's `TEST_access=false`
describes metadata reporter access; `scored_role=TEST` and
`TEST_tensor_access_by_collector=false` make the role distinction explicit.

`merge_profile_reports` selects each row only from its immutable physical
host and reuses accepted `report()` on the complete merged row set. It moves
JSON only. No survivor-only intervals, TEST-selected arms, tuning or retries
are introduced. These outcomes remain exploratory, not held-out confirmation.

## Remaining admission requirements

Metadata checks are not runtime qualification. Before execution, retain:

1. Actual role/provider contracts for27presets and qualified Python/source
   closure, with the existing source reproduction caveats.
2. Representative actual inference measurements, conservative role bounds
   and compatible pair profiles on each authorized physical worker.
3. Actual native selected-state reconstruction evidence, including BUDDY
   context and private BatchNorm state compatibility.
4. Reviewed source/phase/STOP/export custody gates and the pending real
   campaign GO; future77 also needs explicit route qualification.
5. A minimal combined entrypoint and actual one-command-per-GPU launch review.

No remote calls, models, data/checkpoint/TEST reads, forwards, optimizer
updates, new fits or Git operations were used to prepare/check this package.
