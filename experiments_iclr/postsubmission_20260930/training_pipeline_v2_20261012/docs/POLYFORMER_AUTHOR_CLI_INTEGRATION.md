# Explicit author CLI integration proposal

Owned files: `author_cli_spec.py`, `run_author_cli.py`,
`compile_polyformer_author_study.py`, `tests/test_author_cli.py`, this document
and author preparation/handoff files. Shared `spec.py`, `registry.py`,
`launcher.py`, `compile_master_plan.py` and `phase_closure.py` are root-owned
and have not been edited here. Existing immutable packets remain unchanged.

## Contract and worker

The compiler emits one eight-run study with **kind `author_cli_group`** and
**stage `author_cli`**. Each run is one unchanged CLI process with ten
sequential repeats. Run fields retain scheduling identity/config/parents,
remove `adapter`, and add `cli` containing exact frozen source/raw receipts,
private cwd/basis/processed/quarantine paths, executable, argv, environment
and runtime versions. There are no fake model adapters or checkpoints.

The native budget is exactly:

```json
{"unit":"optimizer_updates_per_repeat","policy":"author_native_validation_patience",
 "maximum":2000,"patience":250,"repeats":10,"resume":false}
```

`author_cli_spec.validate_author_spec` checks this budget, native three-role
protocol with no CAL, selector, sequential RNG/split seed contract, confined
paths and explicit whole-master gates. The compiler compares every argv and
command order to the unchanged literal source script. `load_author_spec`
rehashes source/private-dependency/raw bytes without decoding graph/targets.

`run_author_cli.execute` verifies allocation hostname/sole GPU UUID before
project access, checks current runtime version metadata and empty original
private basis/processed paths, then verifies GO and the exact master HPO
closure. It uses real `Registry.claim/event/finish` transactions. An owned
`register_author_spec` writes the same actual SQLite schema through a strict
author validator until root integrates shared validator dispatch. It never
changes or monkeypatches the existing model validator. Duplicate, terminal or
already-running records cannot be launched again.

The child has a new process group and real PID/start-ticks custody, finite
wall deadline and retained START/RESULT receipts. Its literal args remain
unchanged; private dependencies and existing CUDA library paths are child
environment overrides. Signals/deadlines terminate this owned process group.
The source's native early stopping is untouched. No resume/reseed/restart is
offered for an interrupted ten-repeat group.

All raw stdout/stderr go exclusively to the predeclared packet quarantine.
The streaming parser captures only source TRAIN/VAL accuracy and loss plus
repeat/epoch markers. TEST tokens are skipped without conversion or storage
in public metadata. Rounded source diagnostics are not an exact selected
checkpoint or selected epoch. The worker never parses final TEST aggregates.
Public progress/result/registry records retain safe TRAIN/SELECT, PID,
deadline, exit/failure, wall cost, raw-file size/hash and observation counts.
Traceback/exception text from source output is not copied to public results.

Costs distinguish eight CLI groups,80 declared repeat trajectories and
160,000 maximum optimizer updates. Each fully observed epoch proves one
completed native optimizer update and four source eval forwards. Successful
complete ten-repeat observation supplies an exact actual update count. For
an interrupted/unobserved group, completed-epoch counts remain a known lower
bound and the actual total is **null**, with the unknown flag explicit. No
unobserved repeat or update is zero-filled. GPU peak usage is explicitly
unmeasured. Prep has executed0 actual author updates.

## Exact shared-core additions proposed to root

1. **`spec.py`**: add `author_cli_group` to `KINDS`. At the beginning of
   `validate`, dispatch only a nonempty all-author roster to
   `author_cli_spec.validate_author_spec`. This keeps native three-role and
   native-policy contracts separate from standardized four-role/full-budget
   model contracts. `load` can retain `bind_source`; author worker additionally
   verifies per-run provider raw receipts. `Registry.register` then works
   unchanged for author contracts. Mixed author/model runs in one study should
   fail; master combines separate studies.

2. **`launcher.py`**: dispatch `author_cli_group` to `run_author_cli.py`,
   `model_fit` to `run_study.py`, and supported cheap kinds to
   `run_endpoint.py`. Use the existing argv shape, visible `cuda:0`, registry,
   output and approval paths. The launcher log may contain only the wrapper's
   safe metadata; source output stays in the declared quarantine. Author
   memory reservations and finite group estimates use normal slot admission.
   No failed author group is automatically recovered or relaunched.

3. **`compile_master_plan.py`**: add `author_cli_spec.py` and
   `run_author_cli.py` to the frozen common controller file roster; add
   `author_cli: author_cli_group` to `STAGE_KIND`. Retain `cli`, native budget,
   three roles, source args and original quarantine identities when rewriting.
   Set source `worker_entrypoint` to the newly frozen common author worker and
   bind its bytes, preserving original packet worker provenance. Add a
   separate branch in `typed_counts` using `author_counts`; do not pass author
   groups through `fit_semantics`, bank geometry, model `full_budget` checks,
   scalar CAL or endpoint counts. Report group8, repeat80, maximum updates
   160,000, maximum TRAIN forwards160,000, maximum validation/TEST
   forwards640,000, unknown stopped horizon and0 executed preparation work.
   Existing model-fit totals stay separate from author repeat trajectories.
   Retain group-level resource scopes; current estimates are provisional.

4. **Whole HPO gate**: add `author_cli` to `whole_search.required_for_stages`.
   Rebuild `all_packet_source_run_keys` from **all materialized master** search
   model fits and search-selection endpoints, and derive required hosts from
   that full cohort. The author worker independently compares this declared
   cohort to every current plan study and verifies exact copied host proof
   run lists, states and contract/result hashes. All search fits may be
   terminal; every selection must succeed. A packet-local closure is rejected.

5. **Whole author gate**: add `whole_author` with
   `source_stages=["author_cli"]`, the complete master author group roster,
   exact required hosts and no inference from one completed CLI group. In
   `phase_closure.write_host_receipts/collect`, permit `failed` only for
   `author_cli_group` in this gate (alongside existing failed-search policy).
   Preserve every failure. Include `terminal_roster=true` and
   `all_successful=all(states==complete)` in host/global proof summaries.
   Closure means all admitted author groups have terminal cost/custody
   receipts; it does not assert all succeeded. Raw TEST-bearing outputs become
   readable only after this exact whole-author closure. The worker's
   `require_global_author_gate(...,"whole_author",...)` supplies a strict
   metadata-only authorization check for a future result reader. Do not use
   the general registry export or tail source logs to read partial outcomes.

6. **Source acquisition gate**: current frozen author source/raw preparation
   performs0 source model fits and needs only static byte/readiness checks.
   Do not invent a completed zero-job source phase. If root later admits a
   method requiring paid upstream/source fitting, add an explicit typed
   source worker/schema with its real source objective, labels, optimizer
   objects, budget/stopping/RNG/artifacts and source run IDs. A `whole_source`
   gate must then cover all such materialized source jobs and precede any
   dependent search/comparison/author stage. Unsupported source lifecycles
   remain blocked until that exact adapter/worker exists. A cached archive or
   literature score cannot stand in for a paid source fit or gate proof.

The standalone author PLAN intentionally has an external-master HPO
requirement and no actual HPO source cohort. It cannot satisfy the worker's
whole-master HPO check or launch by itself. Master integration produces a new
plan/spec identity; no old approval or closure is transferable. Preparation
creates neither GO nor phase-release receipts.

## Regression gates

`python -B -m unittest discover -s tests -p test_author_cli.py -v` runs seven
standard-library engineering tests: disabled eight-group compiler/counts and
argv-tamper rejection; explicit native kind/noCAL; TEST-token omission;
real subprocess/SQLite ownership and completion; duplicate claim rejection;
wall deadline/failure with unknown partial costs; complete master HPO cohort
and failed-selection rejection; whole-author terminal-failure closure.
Synthetic subprocesses print fixture diagnostics only; no numerical model,
real graph/target data, scientific score or approval is constructed.

Root integration must additionally test shared-validator admission/rejection,
launcher dispatch, mixed standard/author master counts, all-packet HPO coverage,
common worker source bindings, copied author host failure proofs and continued
standardized gate behavior. Root should retain actual preflight receipts
against the frozen allocation scheduler packet. Full author data/basis/model
execution remains untested until qualified admission and human GO.

This is source-based code-path replication under recorded Torch/PyG/DGL/
seaborn differences. It makes no bitwise, published-task environment, published
score match or manuscript acceptance claim.
