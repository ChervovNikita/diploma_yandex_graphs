# Prospective four arm graph continuation design

This design compares full output cotangent support with TRAIN remasking, common descent and a topology null under one shared accepted initialization step and identical continuation rules. It recommends completing the twelve Squirrel fits first, then applying a frozen validation trigger to the equally frozen twelve-fit Photo followup. A fixed two-graph option is also specified. This packet contains a design and metadata bindings; it launches nothing and changes no running cohort.

## Blocks and scope

| Dataset | Native backbone | Full output | Private slice | Paired seed and split blocks |
|---|---|---|---|---|
| Squirrel | PolyFormer-Mono, configuration 0 | 2,223 nodes × 5 classes | `stem.S`, `head.R`, 512 coordinates | (17,0), (29,1), (43,2) |
| Photo | Polynormer-r, configuration 0, final global phase | 7,650 nodes × 8 classes | `stem.S`, `global_head.R`, 1,024 coordinates | (17,0), (29,1), (43,2) |

There are six development blocks and four arm slots per block, for 24 fits. Seed and split are paired; these are not nine cross-products or independent replications of seed and split effects. Each block's four arms share exactly one native warm donor. Architecture, configuration, schedules, support construction, alpha grid, topology-null seed and selection rules are fixed before comparisons.

## Source and warm reuse

`BINDINGS.json` binds the active Round17 v3 precision driver, training adapter and qualifier, sealed support and paired helper, native preparation v3, and six exact checkpoint descriptors extracted from warm FREEZE metadata. The old `study_v2` namespace binds the active v3 source/protocol. All six donors were chosen by the existing declared warm construction, without old arm outcomes or fitted outputs entering this design.

Reuse is limited to the declared common native states and source primitives. Create a fresh prospective warm binding bundle referring to byte-identical donors, keeping specification, preprocessing, named Adam state, aliases, phase and post-warm RNG. Use new in-memory restorations for the four arms. No old fitted checkpoint, selected logits, arm comparison, registry admission or report claim becomes new method evidence. Original warm/role/source artifacts stay read-only. Changing a donor requires a prospective revision before comparisons, with one replacement donor shared across all four arms of that block.

Squirrel warm construction is 50 native updates with the fixed last state. Photo is 200 local updates, an earliest strict validation-NLL local model/Adam handoff with RNG retained after all 200 updates, then 50 global updates with the fixed last global state. These predeclared source-validation choices are part of the common donor construction; they are not new arm selection.

## Native prerequisites and paired initialization

Root reports that exact Squirrel17 native preparation v3 passed full-output AD, shared-alpha construction and all installed member-output equalities, with zero optimizer updates and no validation/final labels. Its receipt is `graph_paired_native_qualification_cpu_gpu_root_v1/QUALIFICATION.json`. This establishes correspondence on that state, not predictive quality. Photo native support construction remains unqualified by this design. A Squirrel pass does not qualify the Photo global predictor or another warm state.

Before continuation on either backbone, require actual full homogeneous output order/shape/finiteness, native/K1/K4 warm equality, active seed+90000 VJP/JVP qualification, exact source and preprocessing fingerprints, and all installed member outputs matching the exact common closure. Qualify each exact warm state, including seeds29/43. Require the pinned dropout-off actual-warm native/K4 Adam one-step equivalence on disposable copies before transporting useful state. Photo qualification must use its final global predictive logits, full cross-node derivatives and 1,024-coordinate slice. No sampled or surrogate closure substitutes for the full graph.

Invoke the sealed paired helper once per block in its fixed order: `common_only`, `train_remasked`, `full_node`, `full_node_permuted`. TRAIN labels are the only initialization labels. Use canonical normalized topology and the genuine seed+80000 `Pi S Pi^T` control with feature/label order fixed; save the permutation vector and exact sparse equality. All arms use the same common gradient and first jointly accepted point in the fixed six-step alpha grid. Preserve every trial and geometry receipt. Signed and Gram diagnostics do not add acceptance gates.

Joint failure blocks every arm in that block: no per-arm fallback, replacement initializer, selected successful subset, extra alpha, restart or donor choice. Record four blocked arm terminals and keep the failed block in the planned denominator. Resource failures are separate: cause/context classification plus the observed closure's resource sentinel takes precedence over a helper acceptance or joint failure, including swallowed candidate failures. It blocks installation while preserving the full helper report. Scheduled sparse products or container member passes on an abort are not reported as completed work.

## State transport and continuation

Construct a fresh K4 copy for each jointly accepted arm; verify common warm equality, install only its admitted slices, and compare all four installed outputs with independent closure references. Transport optimizer state with the unchanged `named_alias_adam_private_bias_coordinate_transport_v1` primitive. Shared W/body states, aliases, inactive local head, learning rates, betas and steps are copied exactly. Private B first moments scale by 1/4, second/max-second moments by 1/16, epsilon and coupled decay by 1/4. New R/S have empty moments and inherit the native boundary-weight group options. All continuation parameters follow the pinned native/boundary training treatment.

Restore the same post-warm RNG **after** cloning, qualification, initialization and state transport for every arm. Match initial random/dropout state and policy; differing early stopping can consume different amounts of that stream. Native weights, logits and training gradients stay FP32. Retain the qualifier's FP32 per-example CE / FP64 finite-difference mean measurement and the sealed helper's established geometry arithmetic unchanged.

| Rule | Squirrel | Photo |
|---|---|---|
| Continuation cap | 1,950 native updates | 950 global updates |
| Stopping | Patience 250 from selected continuation epoch | Fixed global cap |
| Paid warm updates per donor | 50 | 250 |
| Fixed midpoint | Continuation950 / native-stage1000 | Continuation450 / global-stage500 / paid-update700 |

Training loss is the arithmetic mean member CE. Primary pooling is `softmax(mean raw member logits)`. Select among initialized epoch0 and every continuation update using source-validation predictor NLL; retain the earliest strict minimum, with no tolerance tie breaker. Replay the selected state and verify the pinned NLL agreement. Save selected model/named optimizer, member logits and selection metadata. Mean member probabilities are a secondary diagnostic from the same selected logits and never select a checkpoint. Midpoint absence from prescribed early stopping is recorded, not replaced by another epoch.

## Labels and comparisons

| Phase | Label access |
|---|---|
| Design author | Recorded descriptors only; no label bytes |
| Source AD, optimizer audit, initialization | Compact TRAIN pack only |
| Continuation and checkpoint selection | Compact TRAIN and source-validation packs |
| Final confirmation | Six exact compact pool packs only after the separate release gate |

Full features/topology and unlabeled role IDs are transductive inputs. Role IDs verify exact row order, disjointness and coverage; no A/B/D or pool label values enter construction, training, selection or the development trigger. Do not hash validation/final label bytes during author preparation. At future execution, verify validation bytes only when entering authorized continuation; do not recursively open role-freeze payloads or validation/final label arrays during TRAIN-only qualification. Full-node unlabeled predictions may be saved before release.

Primary contrasts are `NLL(full_node) - NLL(train_remasked)`, then the same difference against `common_only` and `full_node_permuted`; negative means lower NLL. Report every paired block, each graph's three-block arithmetic mean, and, if both graphs finish, an equal-graph macro mean. Accuracy and diversity/construction diagnostics are secondary. Six overlapping development blocks do not justify independent-node significance or a broad superiority claim.

## Frozen staging and heldout gate

Recommended mode is `staged_validation_gate`. Complete all three Squirrel joint-attempt records and all twelve arm terminals before any comparison; publish no interim rankings. If all twelve selected fits exist and all prerequisites passed, compute the three-block mean validation NLL for each arm. Photo runs only if full_node's mean is **strictly lower than each of the three controls**. This is a development continuation trigger, not evidence of superiority. A failure/deferred block is recorded, never excluded; missing selected fits make the trigger unevaluable rather than a score on a successful subset.

Squirrel is heterophilic and Photo is homophilic. A negative Squirrel trigger defers this Photo followup; it does not reject the full-support hypothesis on homophilic graphs. The remaining scientific alternative is a prospectively frozen test on Photo or other homophilic graphs with native support qualification and the same controls. It requires a separately declared scope rather than silently reinterpreting this stage. Resource unavailability likewise leaves that scientific alternative open.

The alternative `fixed_two_graph` mode must be chosen and frozen **before** any Stage1 comparison. It runs both predefined sets regardless of Squirrel validation signs, subject to native/source and resource prerequisites. In either mode, Photo schedules, donors, arms, contrasts and selector cannot be tuned using Squirrel results. State which mode produced the evidence and that Photo in the staged mode was conditionally pursued.

Final labels remain closed at the Stage1 terminal. Two-graph heldout release requires all six block attempts and all 24 successful selected fits, complete cost/error terminals, immutable source/warm/selection fingerprints and a frozen full comparison specification; then use one explicit once-only six-pack release. No validation-sign gate selects a best arm for final scoring: score all four frozen arms. Incomplete/resource-deferred or joint-failed studies retain their denominators and cannot open this confirmation gate. The original running cohort keeps its own all72 closure rule unchanged. Broad utility or novelty claims additionally require independent heldout evidence, strong native family/grid comparisons, wider graph coverage and honest accounting of exposed development families and conditional followup.

## Resources and later driver

`RESOURCE_PLAN.json` binds the permitted outcome-free terminal-cost estimate. Its continuation-only proxy is roughly **16 minutes for twelve Squirrel fits** and **18 device-hours for twelve Photo fits**, plus new paired qualification/initialization and other excluded work. Observed twelve-fit proxy ranges are about15.1–16.9 minutes and17.72–17.78 hours; these are scope-specific observations, not precision ETAs. Older operation-ledger sums are separately identified and supply memory observations. Root's new Squirrel17 correspondence assay cost14.54 seconds is a useful source-assay observation, not twelve-fit timing.

Serial execution holds one arm at a time. Planning free-memory screens are `max(8GiB, 2*prior graph peak reserved + 2GiB)`: approximately11.70GiB for Squirrel and69.90GiB for Photo from the older complete-source/audit receipts. They are conservative planning screens, not certified paired peaks; nested peak resets and failed/rejected work must be retained in actual ledgers. Planning budgets are600seconds per exact qualification block,600seconds per Squirrel fit and12,000seconds per Photo fit, with any enforced overrun classified as resource deferral. Probe only the declared device in the future execution process; never change another cohort's server, GPU or RNG settings.

The later single driver should read this compact binding bundle, restore donors, qualify exact states, invoke the paired helper, install/transport fresh arm states, run the pinned continuation primitive serially, then freeze a complete Stage1 comparison. Use one fresh study output root with per-block/arm receipts, refuse existing output paths and preserve input fingerprints. No new coordinator/registry framework or old phase entrypoint is needed. Incremental compute with reused donors and fully paid shared warm/source costs must both be reported; reuse is not a low-cost claim. Scientific merit is assessed by the defined comparisons and remains separate from today's device availability.
