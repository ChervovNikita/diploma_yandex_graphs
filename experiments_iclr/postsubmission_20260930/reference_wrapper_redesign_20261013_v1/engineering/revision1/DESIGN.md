# Reference-first shared-wrapper campaign: engineering handoff

## Scope

This successor retires the proposed v29 80,850-fit campaign from future launch. It prepares fixed native backbone recipes and only the explicitly chosen shared-backbone wrappers; it has no hyperparameter search, SELECT-driven configuration choice, paid teacher compression stage or inherited historical state. The code here is a **disabled metadata/compiler prototype**, not a finished scientific runner or an authorization to train. Root owns the source roster, exact final count, source/data packet preparation, source-faithful runtime adapters, final manifest and GO.

No model or dataset was instantiated, no tensor arrays read and no remote operation performed during this task. Existing sources were read locally. Do not invoke old v29 launch commands after this redesign.

## Two procedures must stay distinct

### Native paper-recipe track (`native_replica`)

Use the official source model, constructor settings, preprocessing, official split, negative/query protocol, loss, optimizer grouping, native scheduler, native phase handoff, evaluation cadence, stopping rule, tie behavior and checkpoint eligibility. A source fixed-epoch cap must remain fixed; a source patience stop must retain its source patience. Changing the split, loss reduction, schedule duration or cap invalidates the paper-recipe replication claim. Published tables may require many runs or a particular evaluator; preserve that requirement or mark the attempt as an explicitly reduced-seed check.

Native M1 is one actual native fit. I4 is four separately initialized, optimized and selected native fits, each with its own complete state/RNG and source controller. Their selected predictions are averaged only after each finishes. If the single reference has exactly the same initialization, dropout/negative sampler streams, graph/split, source settings, loss and selection as I4 member0, reuse that exact fit. Do not train a fifth identical body. Do not train four ordinary networks jointly or select their epochs using pooled I4 validation.

A wrapper run under the same native controller is a *new wrapper comparison under the source recipe*, not a replication of the native paper's original model. Its parameter installation, loss/reduction and extra parameters must be stated. Original GNNM adaptation similarly does not imply the original GNNM paper was rerun; original-five datasets are excluded.

### Longer convergence track (`long_convergence`)

Train cold models from scratch under the same native model/optimizer/loss settings, with a prospectively declared longer cap and modest patience. Include matched M1 and I4 controls under that same longer controller as well as the shared GNNM/wrapper. Differences from the source controller are a finite explicit `schedule_adaptations` list; these results are never labeled paper-recipe replication.

Early stopping becomes available only after the declared minimum horizon and all essential source gates complete, for example native local-to-global transition, source optimizer warmup, and native scheduler warmup. This prevents stopping before a source's main architecture phase starts. Checkpoint eligibility and validation metric/ties remain declared. The helper uses exact metric comparisons; no score-tolerance campaign is added. Diagnostics should report selected step, completed step, stop reason, best-step distance from cap, late TRAIN/validation curves, and whether cap was reached without satisfying patience. Hitting the cap is not proof of convergence.

Do not silently extend a fixed-horizon scheduler. For OneCycleLR, changing total_steps changes even the early trajectory: either use an explicitly declared longer OneCycle recipe and compare all methods under it, or define a preserved source schedule followed by a declared tail schedule. For cosine/polynomial decay, specify the same distinction. Never run a native scheduler beyond its valid horizon, reset moments at an invented phase, or mutate a source selector by forcing `full_budget`.

## Minimal runner architecture

1. **Immutable source recipe.** One recipe row binds source files/commit, native factory, exact constructor/config, official provider/split/evaluator metadata, TRAIN/VALID/TEST role contract, source training contract, and source-supported seed policy. Official VALID remains validation: do not halve it into SELECT/CAL by default. Additional calibration, if scientifically wanted, needs a declared independent role/procedure and should not silently redefine the replica.
2. **Small fixed compiler.** `campaign_compiler.py` builds cold native body0–3 and explicitly listed wrapper fits for every declared recipe/split/cohort seed in each track. It has no model/data imports. Old adaptive keys (`selected_config_from`, HPO grids, etc.) are rejected. Strict training must equal the declared published contract. A new plan/output directory preserves all earlier decisions/results.
3. **Native execution adapter.** A source-specific adapter wraps the actual source's model/train/evaluator hooks and persistent optimizer/scheduler/RNG. It does not replace the source with a generic two-layer approximation. Wrapper installation preserves native nonlinearities, normalization, input/edge policy and bias ordering. The source roster must explicitly identify feasible factor sites. A universal string search and indiscriminate replacement of every `Linear` is insufficient.
4. **Transactional fit registry.** Reuse the existing registry's immutable registration, atomic claim, real process identity, terminal outcomes and explicit resume-after-owner-termination rules. Each I4 body is a separate fit record. No automatic retries or survivor-only averages. The successor registry uses a fresh output namespace and schema, not the v29 registry.
5. **Atomic complete checkpoint.** Reuse selected/final/recovery checkpoint helpers, including complete model, optimizer, source scheduler/phase state, mixed module modes, selectors and Python/NumPy/Torch/private-generator state. Restore the source's own selected state before any final prediction. Distinct source-native plain state must be captured; do not manufacture a cached source state during metadata preparation.
6. **Label-free bank + official evaluator.** Keep prediction export separate from target scoring; preserve full official query/negative layout and evaluator. An I4 export requires four distinct complete selected fit authorities, except that member0 is exactly the M1 authority. The wrapper export requires the selected state of its complete joint shared bank. No pooled epoch reselection and no heldout-guided wrapper selection.

The existing monolithic v29 compiler is not an appropriate entry point for this small campaign. Its source-family validators encode many obsolete candidate inventories and hardcoded caps. Introduce a new source-recipe schema and a narrow runner, while reusing audited state/ownership primitives.

## Concrete engine changes

Existing `engine.py::_validate_spec` hardcodes source-specific policies (HGT300, BSNN500, typed context200, complete-hop2000, plus special AMP clocks). Merely changing a `maximum` field will fail these guards or mislabel altered source recipes. Add two explicit generic policies in a **new runtime copy**:

- `source_reference_v1`: validation proves the complete run contract equals the pinned native source recipe; adapter's source selector/stop callback remains authoritative. Fixed native procedures ignore an invented patience; source-native AMP attempts versus committed optimizer maps remain separately reported.
- `source_long_convergence_v1`: finite cap, explicit schedule adaptations, source-native clock and validation cadence, and declared warmup/transition stop gates. Wrap selection/stopping only where the source adapter explicitly permits it. Native training/optimizer/scheduler internals remain source-owned.

The trajectory loop should continue to call `train_step(unit)` with the source clock (`epoch` may contain many minibatches), record actual optimizer maps/attempts/AMP skips, and call the source evaluator at its native cadence. Preserve source-defined local-best model/Adam handoff. Do not demand that every source epoch perform exactly one optimizer update. The validation routine must not optimize model parameters. The completed state saves the source schedule and gates so a legitimate resume cannot restart warmup or reuse stale selector counters.

`convergence_policy.py` is a metadata-state helper for the longer-track selector only. It supports ordered source metrics (e.g. AUROC then lower NLL), first/latest ties, checkpoint0 eligibility and stop gates. It must not replace a strict native source callback with different semantics. It resets pre-gate patience so misses before a required phase cannot trigger immediate stopping when the phase opens. Root must freeze whether later phase transitions reset best/selector state according to that source; this helper does not invent such resets.

## Scheduling and the three GPUs

Assign an entire recipe/track/split/cohort (four independent native fits, wrappers and their final evaluator) to one worker. This avoids scientific state transfer dependencies between hosts and preserves exact M1/body0 reuse. Worker roster: `allocation/gpu0`, `server77/gpu0`, `server77/gpu1`; maximum two live jobs each, six in total. A second job starts only when combined admitted VRAM/RAM/disk reservations fit the verified host capacities. Other users' jobs remain external occupancy; do not stop them or alter host settings.

The pure `scheduler_policy.py` performs no hardware calls or process launches. Its admission checks count active slots, require complete dependencies and exact registry `registered` state, and require declared resource reservations. Existing launcher functions can supply live host capacity and exact owned process usage. Preserve one-GPU route verification before allocation operations, and require restored current 18.77 hostname/UUID/runtime bindings before its future deployment. Preparation is allocation-only; there is no 18.77 access in this handoff.

Use deterministic longest-planning-weight cohort placement with eligibility/capacity constraints; relative source costs may initially be transparent planning estimates. Counts do not demonstrate balanced GPU hours. Replace weights with observed complete-job time only under a later declared scheduling amendment; this can change placement, not scientific settings, seeds or selection. Endpoint jobs close completed cohorts before starting more fits. No Git rendezvous is needed to train independent host-local cohorts; source and immutable plan are synchronized once, and reports are merged by exact campaign/contract identity after terminal completion.

Future one-command interface per GPU:

```sh
./run_gpu.sh allocation 0
./run_gpu.sh server77 0
./run_gpu.sh server77 1
```

The wrapper must verify active campaign hash, human GO, host binding and runtime/source compatibility, then request that GPU under a detached single-host supervisor. Requesting another GPU must not create a duplicate supervisor or duplicate fit. The existing startup-owner lock/identity design is reusable. These commands are a proposed interface, **not enabled launch scripts delivered by this prototype**.

## Retire v29 without removing history

`RETIRED_PLANS.json` lists the exact old canonical plan hash. `require_active_campaign` refuses an approval for that hash and refuses a disabled or hash-mismatched successor. Root must install this retirement check in the published launch entry point and old-plan GO-admission path before calling the redesigned setup ready. Keep old files/results intact. No GO exists for v29; do not generate one during redesign. A policy file alone cannot block somebody directly invoking a historical script, so this handoff does not claim live old-entrypoint retirement was already installed.

## Compiler input contract and count

Each `reference-wrapper-roster-v1` row contains:

- `id`, `dataset`, `source_bindings`, `native_factory`, `native_fixed_config`;
- `official_split_bindings` keyed by split ID, and the official evaluator/query law;
- fixed `seeds`, `split_ids`, four `native_seed_offsets` (body0 offset0);
- `published_native_training`, and tracks `native_replica`/`long_convergence`;
- explicit wrappers with `id`, `factory`, allowed tracks, native-operator-preservation declaration and own training objective;
- source resource reservations, worker eligibility and transparent scheduling weights.

Per recipe/split/seed/track, exact fit count is **4 + number of declared wrappers in that track**. M1 and I4 exports consume those four native fits and do not add training. Strict and extended tracks cannot alias because cap/stopping/scheduler contracts differ; reuse requires full learning/initialization contract equality, not merely equal width/LR. The prototype conservatively declares separate cold trajectories between tracks. Root will compute the exact full study total from the frozen source roster; the engineering fixture's counts are not the proposed scientific count.

## Read sources / adaptation path

- [Existing engine/state/clock behavior](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/native_postfit_serving_ports_20261013_v2/cumulative_controller/runtime/engine.py:257>).
- [Existing launcher/admission and detached startup](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/native_postfit_serving_ports_20261013_v2/cumulative_controller/runtime/launcher.py:124>).
- [Existing registry/atomic claims](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/native_postfit_serving_ports_20261013_v2/cumulative_controller/runtime/registry.py:25>).

These are locally available cumulative-runtime source references, not a claim that a new executable successor has been frozen on the allocation. Root should cherry-pick narrow audited primitives into a fresh runner rather than editing immutable v29 files in place.
