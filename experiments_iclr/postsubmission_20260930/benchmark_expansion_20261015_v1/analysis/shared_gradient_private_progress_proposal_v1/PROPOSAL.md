# Shared gradient interference after private BN recovery

15 October 2026. Scientific design for root review; implementation and execution are pending review. One dataset, one native backbone, three paired seeds, at most nine new full fits. This proposal is separate from the 25 campaign recipes.

## Question and evidence

Does the actual shared Adam displacement damage otherwise useful routes, and can accounting for their native private progress improve the served predictor? The method remains a trainable shared backbone with small private wrappers. Gradient handling adds zero learned or serving parameters; the present boundary factors are an instantiation, rather than a requirement that future wrappers use BatchEnsemble.

MolBACE/GIN private BN has recovered ranking competence: selected VALID AUC averages 73.614%, versus native M1 73.321%, genuine I4 74.847% and shared-BN GNNM 67.778%. Its mean BCE is nevertheless 1.086, versus M1 0.779 and I4 0.890. At the same selected private-BN weights, coherent common mixture moments score 68.449% AUC, better than arithmetic tying's 59.060% but below own banks in all three seeds. These are development observations on the same 151 VALID graphs. Private BN improves the evaluation map and selection; under the inspected ordinary TRAIN BN contract, running-state ownership alone does not change the common mathematical TRAIN prefix.

Consequently, the remaining hypothesis concerns shared optimization after a normalization confound has been controlled. Neither the AUC recovery nor old negative gradient cosines establishes harmful shared learning. The previous SYNC outcomes tested separate Adam moments and synchronized weights; PCGrad/CAGrad/MGDA interventions remain untested in the retained GNNM record.

The completed nine-fit follow-up now closes the ingredient question: private BN alone scores 73.614% AUC, plus geometry 73.651%, plus input correction 72.186%, and both 73.028%, against I4's 74.847%. Geometry adds only 0.0366 points on average with mixed seed signs; correction lowers AUC in all three seeds. The within-private interaction is +0.806 points versus shared BN's -5.873, but the combined procedure remains below private BN with neither added ingredient. Moderated antagonism does not supply an ingredient gain. These complete results strengthen the case for a direct interference audit rather than another geometry/correction combination (`benchmark_expansion_20261015_v1/BN_GEOMETRY_COMPLETE_16.json`).

## Minimum decisive audit

Use all three existing private-BN selected states, epochs 92, 54 and 116 for seeds 101, 202 and 303. Restore their genuine parameters, own BN banks, native Adam state and RNG. Follow the next complete native TRAIN epoch on a disposable branch, covering all 38 eligible batches per seed. This is 114 charged shadow Adam transitions, with no retained training result or new model selection. It samples these selected states and their immediate continuation, not the original full learning history.

For each batch, retain the original four route forwards and source mean-own backward. Also collect each unscaled supervised route gradient before the step. Let shared parameters be theta, private route parameters phi_m, native proposed displacements d0 and Delta phi_m, and route loss ell_m. Record

```text
s_m = grad_theta ell_m dot d0
p_m = grad_phi_m ell_m dot Delta phi_m
t_m = s_m + p_m
```

Use actual native Adam displacements, including current moments and epsilon. The original mean loss supplies private gradients scaled by 1/4; the protected ell_m gradients above are unscaled. In this inspected model a route reads only its own private rows. A replacement wrapper with cross-route prediction dependencies would require the sum over every private block instead.

Measure finite changes on the same batch and dropout realization at O=(theta,phi), B=(theta,phi_plus) and D=(theta+d0,phi_plus). The two extra four-route forwards are in TRAIN mode under no_grad. They restore the same pre-forward RNG and banks for each trial, then reinstate the native post-forward banks and post-step RNG. They must not read eval BN, consume another loader batch, recalibrate banks or advance the live route counters. Report ell(B)-ell(O), ell(D)-ell(B) and ell(D)-ell(O). The finite shared effect is conditional on private progress; s_m is an old-state derivative, so these are distinct measurements.

Report every route and batch, per seed: shared ascent, private progress/ascent, predicted joint ascent, finite shared and joint harm, cosine/norms as secondary diagnostics, and whether private descent pays for shared ascent. Accumulate dot products in FP64 over the native FP32 coordinates. Do not treat the 456 route-batch events as independent experiments.

**Proposed conditional gate:** advance the interference intervention only if at least two seeds have at least 16 of their 152 route-batch events with all three quantities exceeding tau: t_m, ell(D)-ell(B), and ell(D)-ell(O), where tau=1e-4 + 1e-4*abs(ell(O)) nats. This is a prospective materiality screen, not a significance test. Repeated native forward reconstruction must meet that fixed tolerance before interpreting the screen. Also report compensated shared-ascent events; if absent, private-progress credit has no demonstrated target even if generic interference exists.

This gate uses TRAIN changes, with no new VALID predictions, candidate performance ranking, checkpoint choice or seed exclusion. All three seeds enter any subsequent comparison. A closed gate parks this particular interference explanation for the sampled state range; it does not prove that PCGrad cannot improve generalization for another reason. The gate and all diagnostic choices must be settled before the audit, rather than changed after seeing its result.

## Matched comparison if the gate opens

All cells use the existing native 5-layer width-300 nonvirtual GIN, dropout .5, batch size 32, mean graph pooling, original categorical inputs, four routes, own supervised BCE, private BN with shared affine parameters, and mean raw-logit serving. No geometry objective or input correction is added. Keep Adam lr=.001, betas .9/.999, epsilon 1e-8, zero decay; complete native epochs; maximum 1000 epochs, minimum 100, patience 50; whole VALID AUC each epoch, first strict maximum selected. There is no optimizer coefficient search.

| Condition | Shared operation | Private operation and state |
| --- | --- | --- |
| Native BN control | Original Adam on mean own BCE | Original native private Adam |
| PCGrad control | Standard randomized pairwise conflict projection on the four shared gradients, averaged before the native Adam map | Original private mean-own gradients; shared moments follow the PCGrad gradient |
| Shared-only projection | Closest displacement to the native shared Adam proposal, with every route's shared directional change nonpositive | Native private proposal and native mean-own moment transition retained |
| Private-progress projection | Identical projection, replacing zero shared-ascent budgets by the measured private-progress offsets | Same private proposal and moment convention as shared-only projection |

For PCGrad, project a route vector only against an original other-route vector when their dot product is negative; skip a zero comparison vector. Use an isolated seeded task-order generator, without consuming training dropout/loader RNG. Average the four projected vectors to preserve the source mean-loss normalization. This is the established PCGrad ingredient integrated with native Adam; its gradient geometry does not certify that Adam's actual displacement protects every route.

For both displacement projections, form A with rows grad_theta ell_m and solve

```text
min_d  .5 * ||d - d0||^2
subject to A d <= b

shared-only:       b_m = 0
private-progress:  b_m = -p_m
```

Use all shared native parameters, including BN affine parameters, with no favorable block selection. Four inequalities give an at-most-16-subset active-set solve. Use the same Euclidean metric, rank/KKT criteria and deterministic tie handling for both. No ridge, slack penalty, pooled-progress equality or finite-loss acceptance search is added. The applied correction must not exceed ||d0||; unresolved/infeasible or larger-correction cases use the native joint proposal and remain in all outcomes. The zero-budget projection is feasible at d=0, whereas private ascent can make the offset constraints infeasible. Fallbacks forfeit protection and must be reported.

One native Adam call proposes both blocks and its new moments. The projection replaces only the shared displacement; native private displacements, moments and step clocks are retained. No second Adam call applies the correction. Private updates are identical between the two QP rules at the same old state, but are not forced numerically equal across diverged training histories.

The decisive extension contrast is private-progress versus shared-only projection. Native versus PCGrad tests a useful known optimizer. PCGrad versus the QP rules also changes optimizer-state conventions and cannot by itself attribute a private-progress benefit. CAGrad would add a radius choice; FAMO tests delayed relative-loss weighting rather than this same-step offset question. Neither is another arm here.

Reuse the three exact original private-BN control fits and existing M1/I4 banks if constructor, initialization, batches, RNG, optimizer, controller and serving contracts match. This leaves three new methods times three seeds: **nine new full fits**. Do not silently add replacement control fits if matching fails; return that changed scope for review. Every new fit starts from the original paired initialization, not a favorable selected BN checkpoint. The audit states are diagnostic inputs only.

## Decisive outcomes and theory limits

Primary scientific contrast: private-progress versus zero-budget projection in selected VALID served AUC and BCE. Report all three paired differences and the complete native/PCGrad/M1/I4 table. A useful extension must improve served AUC over the zero-budget rule without worse mean BCE, reduce unnecessary corrections on compensated events, and show the corresponding change in actual shared/joint harm. Smaller corrections, more feasible steps or better TRAIN descent alone do not pass.

For a practical shared-wrapper advance, require mean AUC at least one percentage point above both reused private-BN and genuine I4, with positive AUC differences in all three paired seeds and no worse mean BCE than either comparator. M1 remains a competent reference. If only PCGrad achieves this, the result supports a useful established training ingredient; if private-progress matches PCGrad/zero-budget, its incremental utility is unestablished. Preserve adverse individual-seed BCE and mean/worst member AUC. A tradeoff remains a tradeoff. These three seeds on encountered VALID cannot establish general I4 superiority; any wider claim needs a separately frozen unused-data confirmation.

Record predicted contributions, constraint activation, infeasibility/fallbacks and correction size at every new optimizer step. At the first eligible batch of epochs 1, 10, 50, 100, 200, 400, 800 and 1000, if reached, also measure the same-view O/B/D finite changes without steering the update. Use the identical schedule for all new methods. Post-training diagnostics must not change the checkpoint or pool.

On accepted offset steps, the exact local statement is grad_theta ell_m dot d + p_m <= 0. It is only first-order sampled TRAIN nonincrease. Curvature, changed views, BN evaluation, native momentum carryover and future trajectory effects remain outside it. Neither the QP nor PCGrad creates useful diversity by theorem. Protecting every route's fitted risk may obstruct useful shared progress. The offset QP is established loss-halfspace projection after fixing private coordinates, not a new optimizer principle; its defensible contribution would be demonstrated utility of private-progress credit in this shared-wrapper setting.

## Implementation path and cost

The current source exposes the necessary hooks:

- `reference_wrapper_redesign_20261013_v1/smoke_20261014/ogb_port/session.py:230,248`: exact native BCE and four-route mean-own update; one native Adam owns shared and private parameters.
- Its `categorical_boundary.py:124,131`: route forwards and explicit private parameter tuple. Original native references supply the shared partition. BN banks are buffers, rather than learned/private gradient coordinates.
- `benchmark_expansion_20261015_v1/private_bn_state_v3/banks.py:74,121`: original BN module execution with route-owned banks and identity-preserving pointer binding. Trial TRAIN forwards must snapshot/restore buffer values outside active contexts.
- `benchmark_expansion_20261015_v1/molecular/session_extension.py:46`: an existing pattern for replacing only session.train_update. A new extension can attach private BN and override this update without changing native constructors, losses or serving.
- `benchmark_expansion_20261015_v1/portable_dispatcher_v3/runtime_node_molecular/adapters.py:294,313`: complete native batch/epoch coverage and persisted private loader RNG. Existing session state/load_state already serializes model, banks and Adam; PCGrad's isolated order-generator state would additionally need persistence.

A conservative implementation keeps the original mean.backward and adds four route-gradient collections from those same forward tapes. Thus one update costs five reverse collections, representing eight route backwards rather than the native four. Four collections do not each traverse all four routes. No extra training forward is needed outside the declared finite monitors. This avoids assuming a gradient-reduction optimization before its native equivalence is qualified.

The completed three BN fits cost **4740.974 seconds = 1.317 hours**, 464 epochs and 17,632 Adam maps; observed allocated CUDA peaks were 0.436–0.440 GB. This measured cost includes VALID/export and the then-current host contention. It is a reference, not a timing for the new methods.

| Work | Finite quantity |
| --- | --- |
| Minimum audit | 3 shadow epochs; 114 temporary Adam maps; 570 reverse collections; 456 native plus 912 monitor route forwards; 912 route backwards; zero VALID/TEST calls |
| Nine new fits if the native 149/149/166 epoch counts recur in each arm | 1392 epochs; 52,896 Adam maps; 264,480 reverse collections; 423,168 route backwards; 3.951 hours at the old native throughput before added VJPs/copies/solver work |
| Nine-fit hard 1000-epoch ceiling | 342,000 Adam maps; 1,710,000 reverse collections; 1,368,000 training route forwards and 2,736,000 route backwards, plus VALID/export |
| Sparse finite monitors | At most 9 fits times 8 monitor points times 8 extra route forwards = 576 extra route forwards |

If forward/backward time dominated and kernels scaled ideally, doubling backward arithmetic would place the nine-fit workload between one and two times the 3.951-hour native proxy. Launch overhead, contention, tensor copies and small-solver synchronization can exceed that proxy; it is not a wall-time bound. Measure the new path before reserving full-fit resources. At the hard epoch ceiling even the old native-throughput proxy is 25.54 hours before extra work, so cap-hit cost must stay visible.

For P shared FP32 coordinates, four route-gradient vectors occupy 16P bytes. Native parameter snapshots, actual displacement, correction and solver work add shared-vector storage; PCGrad also needs original and projected vectors. Use 60P bytes as a planning allowance for fifteen shared vectors, plus a complete parameter/optimizer/buffer snapshot for the audit and existing activation tapes. This allowance is not a validated peak bound. Actual allocation and step time need measurement; the old 0.440-GB peak is not a qualified bound. No additional native/I4 bodies or teachers are acquired.

## Retained evidence

Paths below are relative to the postsubmission research root.

- `benchmark_expansion_20261015_v1/pilot_20261015_bn_state_v1/complete_analysis_v1/{INTERPRETATION.md,SUMMARY.json}` and `benchmark_expansion_20261015_v1/resource_budget_draft_v1/BN3_COST_METADATA.json`: complete competence, proper-loss and paid-cost evidence.
- `benchmark_expansion_20261015_v1/analysis/bn_mixture_moments_completed_v1/INTERPRETATION.md` and `benchmark_expansion_20261015_v1/analysis/bn_train_semantics_source_v1/NOTE.md`: stronger common-moment control and TRAIN/eval distinction.
- `shared_member_gradient_conflict_primary_20261007_v1/NOTE.md`: retained CAGrad/FAMO scope and actual-displacement warning.
- `shared_adam_private_progress_projection_assessment_20261007_v1/NOTE.md`: offset QP, exact block-elimination ancestry, feasibility and local-theory limits.
- `graph_shared_gradient_routing_20261003_v1/REPORT.md` and `shared_core_attributable_progress_balancing_20261010_v1/REPORT.md`: retained PCGrad comparator and distinct FAMO-style feedback proposal.

No new primary-paper read, model/data/checkpoint access, implementation, remote operation or Git action was used to prepare this design.
