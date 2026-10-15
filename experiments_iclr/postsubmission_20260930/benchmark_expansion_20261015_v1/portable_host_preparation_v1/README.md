# Portable host preparation v1

This is a local source successor for a staged rollout. It provides a functional allocation command path through the existing durable dispatcher, portable runtime identity guards, and explicit descriptor-field rebinding. **77 remains disabled and is not ready to run.** No host was contacted or probed, no native job ran, and no campaign GO was created.

## Implemented allocation path

`run_gpu.py` sets the exact host-config hash and GPU UUID mask, preserves the lexical `.venv/bin/python`, and delegates to `expanded_dispatcher_v2`. The v2 child currently sets the sole allocation index `0`; the portable guard accepts that mask only after verifying the sole declared allocation UUID. Future GPU1 still uses logical `cuda:0` behind its own exact UUID mask.

The existing dispatcher retains its two-child ceiling, reciprocal pair compatibility checks, resource admission, leases, permanent claims, STOP handling, failed-attempt records and reviewed same-fit resumes. Two is a ceiling: an incompatible or unqualified pair remains serial. A fixed output/claim namespace prevents this wrapper from silently switching to a fresh output tree.

The allocation runtime copies derive from **runtime_v4 for node/molecular, runtime_link_v8 for link, and runtime_v2 for I4**. Ten modules are byte-identical. The other six change only the host guard or CLI root/field checks. All training, selection, evaluator, checkpoint and export functions are AST-identical to their parents. The standalone portable interpreter entry preserves the exact lexical executable, resolved binary, prefix/base-prefix and `pyvenv.cfg` checks; the initial allocation path continues through the already qualified v2 interpreter entry.

`prepare_allocation.py` is the concrete metadata preparation step. On the authorized allocation it pins the exact existing compiler3 summary, verifies the skeleton/bindings/candidates, replaces only runtime entrypoint receipts, adds this source/config closure, and emits `generated_allocation_v1`. All 6,460 job/science hashes, 270 I4 dependency sets, placement and output-relative names remain intact. Existing permanent campaign claims block a new namespace. It carries no previous admission forward and compiles a blocked release until root attaches actual qualification/resource/reporting authorities and human GO. It was source-checked here, not executed on the allocation.

After root reviews the guard delta, prepares the metadata generation, attaches the actual remaining authorities, and materializes the pinned run specification from `ALLOCATION_RUN_TEMPLATE.json`, the single command from the allocation repository root is:

```sh
.venv/bin/python -B experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/portable_host_preparation_v1/run_gpu.py --worker allocation/gpu0 --run-spec experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/portable_host_preparation_v1/reviewed/allocation_run.json
```

The source-only preparation command from that same root is:

```sh
.venv/bin/python -B experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/portable_host_preparation_v1/prepare_allocation.py
```

Neither command was executed here. The run template deliberately has no invented release, readiness or GO receipts.

## Exact relocation boundary

The runtime descriptor fields eligible for relocation are enumerated in `descriptor_binding.py`:

| Field | Required invariant |
| --- | --- |
| `callback_module`; `session_transform.path`; `extension_factory.path`; `session_extension.path`; `link_policy_module`; `source_rng_emulation.audit` | Root-confined location; referenced source bytes and SHA-256 stay identical |
| `data_root`; `recipe.data_files.{graph,train,valid,test}` | Same provider bytes, role/split population and existing data receipts; no tensor access by rebinder |
| `recipe._repository_root`; `recipe._redesign_root`; optional `recipe.redesign_root` | Exact declared repository/phase mapping; no mounts or symlink emulation |
| `method_settings.context_manifest` | Same immutable context-manifest bytes and hash; no context regeneration or substitute seed |
| `expanded_contract`, `source_plan_entry`, `descriptor_preparation` | Preserved byte-for-byte as scientific identity/provenance, including historical path strings |
| task ID, scientific hash, method/optimizer/seed/body/controller/roles/dependencies | Unchanged; only the host-bound runtime job hash changes when runtime paths change |

The rebinder parses path components and edits only these fields. Any unclassified absolute runtime field or path traversal is rejected. It supports separate repository and phase mappings, since the future 77 repo-relative layout need not match allocation. Runtime guards additionally resolve real paths to reject an escaping symlink. Destination source/data identity is an admission step; this offline utility never certifies it from a path rewrite.

Release-level pointers also need explicit host binding: skeleton/source-descriptor receipts, source closure, runtime/interpreter/environment receipts, qualification/resource/cost/GO/readiness receipts, output/claim/lease/STOP roots, resume/export receipts, and the static group-to-worker assignment. They are **not** rewritten by the JOB helper. The existing v2 equality check (`job == original`) must become equality to the exact reviewed field-wise host-bound descriptor in a later multi-host compiler, retaining both canonical and bound job hashes. Existing claims/checkpoints must not be moved or reset by rebinding.

## Finite remaining migration for 77

1. **User host reauthorization, then actual verification.** Fill the two disabled host configs with observed hostname, the complete ordered GPU UUID list, lexical venv environment and source/data/runtime/resource receipts. The proposed `postsubmission_research_20260930` subtree under the authorized 77 repository is a staging design, not an observed layout. Create a normal repo-local layout after review; never mount, symlink or modify a host to imitate allocation absolute paths. Do not copy allocation venv prefixes as 77 environment evidence.
2. **Bind the complete finite inventory.** On authorized custody, inspect all 6,460 canonical descriptors with the explicit field audit, verify destination source/data bytes, write immutable bound descriptors, and retain canonical-to-bound job hashes. Unknown runtime paths block that descriptor. Preserve scientific contracts and all native source hashes. The local checks here cover three saved engineering descriptors, not the full remote corpus.
3. **Freeze one global ownership manifest.** Partition complete cohort groups, keeping each I4 export and all four native bodies on one physical worker. Every one of 6,730 IDs has exactly one owner. Verify disjointness/exhaustive coverage across three worker projections; retain all permanent claims. Never repartition an already claimed fit without a separately reviewed custody/resume decision. If allocation work starts first, only verified never-claimed tasks are eligible for future reassignment.
4. **Implement one shared 77 host resource ledger.** The current `campaign_launcher_v1.probe_system` assumes one allocation GPU, and Engine passes only its own GPU's active children. A host-wide flock must serialize check/reserve/spawn/commit across both 77 supervisors, with durable startup reservations and PID/UID/start-time reconciliation. GPU memory is reserved per actual UUID; RAM and disk reservations include both workers. Each GPU admits at most two reciprocally compatible profiles. Crash/orphan uncertainty blocks admission until reviewed reconciliation; it does not release a reservation optimistically.
5. **Add global claim and STOP custody.** Do not assume allocation and 77 share a filesystem or that independent local claim directories prevent cross-host duplicates. Pin the single ownership manifest and worker projections; add a reviewed cross-host STOP/claim transport or coordinator with failure/partition behavior that stops new admission and drains owned children. Retain immutable receipts and the same-fit-only resume rules. This step is not implemented by `run_gpu.py`.
6. **Compile the host-aware dispatcher successor and admit it.** Replace allocation-only guard/worker/root/closure checks using the already implemented host binding, use a host-specific resource backend from step 4, bind global ownership/STOP authorities from step 5, and preserve lexical interpreter validation. Qualify the actual host environment and resource envelopes in the authorized scope. Bind a real human GO to the exact global release and per-worker readiness. Only then remove the deliberate 77 scheduler block in the launcher.

The prepared eventual command interface, from `/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git` under the proposed reviewed layout, is:

```sh
.venv/bin/python -B postsubmission_research_20260930/benchmark_expansion_20261015_v1/portable_host_preparation_v1/run_gpu.py --worker server77/gpu0 --run-spec postsubmission_research_20260930/benchmark_expansion_20261015_v1/portable_host_preparation_v1/reviewed/server77_gpu0_run.json
.venv/bin/python -B postsubmission_research_20260930/benchmark_expansion_20261015_v1/portable_host_preparation_v1/run_gpu.py --worker server77/gpu1 --run-spec postsubmission_research_20260930/benchmark_expansion_20261015_v1/portable_host_preparation_v1/reviewed/server77_gpu1_run.json
```

These two commands are intentionally blocked by the current source, before probes, until that finite migration and authorization are complete. The portable runtime guard itself supports a declared verified future host; the shared-host scheduler is the remaining source component.

## Checks and scope

`check_source.py` performs source/AST and compact metadata checks only. It verifies wrong-host/device rejection, disabled77 rejection before any probe, exact allocation identity rebinding, a node/molecule/BUDDY relocation round trip, rejection of unknown/escaping runtime paths, unchanged scientific functions and lexical venv command construction. `CHECKS.json` does not claim native execution, 77 readiness, full-descriptor portability or resource qualification.

No existing compiler, dispatcher, runtime, claim, benchmark result or scientific contract was edited. `MANIFEST.json` pins this source package and the inspected parent sources for root review. No Git operations, remote connections, native jobs, tensor reads or hardware probes were performed.
