# Disabled master plan assembler

12 October 2026. `compile_master_plan.py` consolidates **already compiled immutable disabled packets**. Owned files are the assembler, `tests/test_master_plan.py`, and this document. It does not execute a model, read graph/target arrays, connect remotely, launch workers, create approval receipts, create phase-closure receipts or grant GO. Root supplies the common frozen controller and performs allocation/source/resource/task QA.

## Input index and readiness

Use a project-relative JSON index with schema `disabled-master-input-index-v1`. Each compiled packet declares a unique ID and exact `path`, `bytes`, `sha256` receipts for its PLAN and SUMMARY. An optional canonical `plan_hash` must agree with SUMMARY's hash when that field exists. Optional `coverage` receipts preserve CSV/Markdown coverage/blocker documents. Example shape (replace receipts with actual metadata hashes):

```json
{
  "schema": "disabled-master-input-index-v1",
  "packets": [
    {"id": "native", "plan": {"path": "NATIVE/matrix/PLAN.json", "bytes": 123, "sha256": "ACTUAL_SHA256"},
     "summary": {"path": "NATIVE/matrix/SUMMARY.json", "bytes": 456, "sha256": "ACTUAL_SHA256"}},
    {"id": "link", "status": "pending", "blockers": ["sealed final TEST custody", "final bank/evaluation endpoints"]}
  ]
}
```

Pending rows may bind a `manifest` receipt; their source/data blockers and prospective formulas remain visible, but **unmaterialized forecasts are not added to registered fit totals**. The current link TRAIN/SELECT qualification does not supply final TEST custody. Native resource qualification remains incomplete until its complete measured roster is retained. The molecular four-study/24,560-fit disabled packet is an input scope, not a migration or ALL249 claim.

`--mode dry-run` is the default. It verifies metadata receipts, canonical study hashes, schema, disabled/no-GO flags, run assignments/dependency DAG, known optimizer/body semantics, bank ancestry, and required reference cohorts, then prints readiness. Invalid inputs are listed and return exit2. Duplicate study IDs across packets, conflicting immutable source/data bindings, and ambiguous identities fail; nothing is silently deduplicated. Run IDs are scoped by study, so legitimate repeated names in different studies remain distinct `study_id/run_id` identities. Repeated run IDs within a study fail.

The assembler reads only PLAN/SUMMARY/study/coverage/evidence metadata and frozen controller code. It preserves source/data path/bytes/hash declarations from each original contract without opening or rehashing graph or target-array files. The admitted common execution controller rechecks its bound files. This distinction is saved explicitly; metadata validation is not new provider/heldout-array qualification.

## Common frozen controller

Preparation requires `--controller-runtime` outside the live `training_pipeline_v2_20261012` package. Root copies the **latest** `engine.py`, `metrics.py`, `spec.py`, `registry.py`, `partition.py`, `run_study.py`, `run_endpoint.py`, `launcher.py`, and `phase_closure.py` there after current QA. That includes the root's namespace-isolating dispatcher and latest validation operation-count handling.

Every rewritten spec binds every common-controller file individually, plus its original frozen packet source/data bindings and input identities. It retains each exact adapter factory path; the new dispatcher imports that adapter's own frozen relative dependencies in its namespace. A molecular classifier or metrics module cannot silently resolve from the common/native package. The assembler checks that the supplied dispatcher has the namespace construction, but this source check does not substitute for root execution QA.

Rewriting changes spec hashes and global plan identity. No original packet approval or gate receipt is transferable. Models, seeds, HPO domains, objective/optimizer recipes, budgets, selectors, role authorities and adapter paths stay intact. Optional resource overrides change only existing reservation/estimate fields; no outcome, source initializer or learning decision is edited. Existing sources with conflicting bindings for one physical path fail instead of being relabeled. There is **no automatic cross-study fit/source reuse**; exact existing within-study member0 identities remain reused and count once.

## Typed registered counts

Counts come from actual unique run contracts. Original packet summaries and coverage are retained in full for audit rather than numerically summed into incompatible columns.

| Contract | Atomic run group | Paid optimizer trajectories | Physical slow bodies | Actual prospective step calls |
| --- | ---: | ---: | ---: | --- |
| Native coupled untied4 | 1 | 1 | 4 | One per declared optimizer-update unit |
| Coordinate synchronized historical untied | 1 | 4 | 4 | Four per full-graph source cycle;2000 epochs means8000 calls |
| Native shared/factor/original4 | 1 | 1 | 1 | One per declared optimizer-update unit |
| Separately fitted I4 | 4 member fit groups | 4 | 4 | Each member's own exact budget |
| Molecular or link epoch trajectory | 1 | 1 | 1 | Unresolved until source minibatch/tail count is known/measured |

The master separately reports atomic fit groups, paid optimizer starts, physical slow-body instances, multi-optimizer cohorts, coupled multi-route groups, registered epoch/update units, known registered model step calls, unresolved epoch fit groups, search selectors, exports, CAL fit jobs/updates, evaluation jobs, served banks/routes, genuine I4 banks and exact member0 reuse references. An epoch job is not misreported as one optimizer update per epoch. If any registered epoch trajectory has unresolved minibatch updates, the complete optimizer-call total is **null**, with a known subtotal and explicit unresolved run roster. Unknown costs are never zero-filled.

Genuine I4 exports require four distinct one-route/one-optimizer fits, body IDs0–3 with body0 first, unique actual initialization seeds, common learning/data/selection contracts apart from declared RNG/resource fields, and explicit fit dependencies. A one-group historical untied bank cannot pass this check. Member0 reuse counts only when the same exact fit ID also supplies a declared one-member bank. Unsupported source/warm-start/continuation lifecycles fail until their exact typed source adapter contract is added, rather than silently claiming zero or ordinary fresh fits.

All totals are **registered maxima, not executed work**. No model update, scalar update or predictive score is produced by assembly. Input executed/scored/approved flags are rejected when importing an input as disabled work.

## Placement and measured/provisional cost scopes

`--resources` supplies positive byte capacities for exactly `allocation/gpu0`, `server77/gpu0`, `server77/gpu1`. These are planning bounds, not live hardware proofs or permission for remote execution. Two slots per GPU are retained. Every whole study remains one dependency group; original multi-study groups and cross-study barriers join connected studies onto the same GPU. Missing parents, repeated assignments and cycles fail.

Source reservations and estimated seconds are retained by default. Optional `study_profiles` classify memory/time separately using `kind` in `measured`, `measured_proxy`, `provisional`, `mixed`, `unknown`, a plain `scope`, and evidence for measured claims. A profile may override specified runs only:

```json
{
  "worker_capacities_cuda_bytes": {"allocation/gpu0": 123456789, "server77/gpu0": 123456789, "server77/gpu1": 123456789},
  "study_profiles": {
    "ACTUAL_STUDY_ID": {
      "memory": {"kind": "measured_proxy", "scope": "maximum-shape no-update cases plus declared Adam storage; live concurrency recorded", "evidence": [{"path": "RESOURCE/MANIFEST.json", "bytes": 123, "sha256": "ACTUAL_SHA256"}]},
      "time": {"kind": "provisional", "scope": "measured backward/forward proxy with unmeasured optimizer and serialization allowances"},
      "runs": {"ACTUAL_RUN_ID": {"reservation_mib": 8192, "estimated_seconds": 1000}}
    }
  }
}
```

Without explicit classification, memory is `unknown` and time `provisional`, while the original profile/evidence/status is retained verbatim. Supplying a label does not qualify an unmeasured component. Worker reports include exact typed counts, serial estimated GPU seconds/days, reservation maximum, the largest two registered reservations, capacity feasibility, resource scopes and pending live headroom qualification. The master never divides serial cost by two and calls that measured two-slot throughput. The launcher still checks aggregate active reservations and current external free memory at every admission; a pair that does not fit runs serially.

Optimizer kernels, checkpoint serialization, disk capacity/I/O, scalar endpoint time, and concurrency assumptions remain visible in the input profiles/summaries/evidence. The master retains the native resource projection and molecular declared estimates when attached as metadata coverage/evidence; it does not convert those estimates into actual training cost. A source no-update memory admission is not qualification of complete optimizer or serializer behavior.

## All-packet gates and references

Per-packet gates are replaced by new master `whole_search`, `whole_comparative`, `whole_calibration` gates. Their source cohorts include **every materialized input packet**, across actual assigned hosts:

- All search fits and successful frozen selection endpoints must close before any comparative fit.
- Every comparative fit, including matched references and independent member fits, must complete before bank export, CAL or evaluation.
- Every registered CAL job must complete before evaluation.

Required hosts are derived separately from each gate's actual source cohort, so a host with no CAL jobs is not asked to invent a CAL proof. The root collector uses exact current master run-contract hashes and complete host run rosters. A single packet or host closing cannot release the others. Failed search attempts remain reported while selection must succeed; failed required comparisons or CAL banks do not close their global phase. This assembler creates no gate directories or receipts.

Every named candidate/reference cohort is checked within its study and registered seed/split roster. Missing competent references block preparation. No similarly named cross-study/source result is substituted. Unsupported original GNNM molecular/binary references, absent author replication, missing final link custody and unported method families remain in preserved comparison/source/data/coverage blockers. A supported subset closing does not establish ALL249 migration, confirm an unsupported comparison, or authorize scores for an unmaterialized pending packet.

## Outputs and commands

`--mode prepare` requires controller, resources and a fresh `--output` outside the live package. It emits `PLAN.json`, typed `SUMMARY.json`, `READINESS.json`, compressed rewritten studies, and `WORKER_COSTS.csv`. SUMMARY binds the canonical new plan hash. Original packets remain unchanged. All outputs say scientific launch disabled, human GO required, approval/GO receipts absent, and ALL249 coverage false. Host bindings default to pending live verification; optional supplied metadata is preserved without a new live-verification claim.

CLI runs only on the verified allocation before project metadata access:

```sh
python -B training_pipeline_v2_20261012/compile_master_plan.py \
  --project-root ALLOCATION_PHASE_ROOT --packet-index MASTER_INPUT_INDEX.json --mode dry-run

python -B training_pipeline_v2_20261012/compile_master_plan.py \
  --project-root ALLOCATION_PHASE_ROOT --packet-index MASTER_INPUT_INDEX.json --mode prepare \
  --controller-runtime FROZEN_MASTER_CONTROLLER --resources MASTER_RESOURCES.json --output MASTER_DISABLED_run01
```

Local standard-library fixture command:

```sh
python -B -m unittest tests.test_master_plan
```

Fixtures check four-object accounting, unknown epoch totals, receipt changes, duplicate IDs, missing references/dependencies, global source cohorts spanning packets, common-controller bindings, pending link custody, exact I4/body0 accounting, and disabled outputs without approvals/phase receipts. They do not import Torch, execute adapters or open provider/target arrays. Root must perform actual common-controller/resource/whole-plan QA before presenting the concrete roster and costs for explicit human GO.
