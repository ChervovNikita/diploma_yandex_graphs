# Current GNNM research state

Updated after the 23:12 UTC observations on 8 October 2026 (9 October in Moscow). **The goal remains active and unmet:** no new methodological extension has demonstrated superiority to capable single models and ordinary independent ensembles, and no fresh manuscript acceptance has been established. Original paper scores are unchanged. Complete history, failures and unsuccessful hypotheses remain in Git and `research_ledger.json`.

## Current hypotheses and live training

| Study | Verified live handles | Complete work | Next comparison |
| --- | --- | --- | --- |
| Molecular ensemble feedback, allocation | Parent 523400 / birth 6019318952 verified alive | 14/18 fits; O_7307 at 40/100 epochs, 23:12 UTC | Open all18 only after the fixed roster closes. Candidate I, mixture0.5 remain fixed. |
| Private graph attention,18.77 | Parent 3713404 / birth 1760762180 verified alive; both GPUs at 100% | 6/12 fits; relationJ seed 6101 at 1027/1100, seed 6203 at 437/1100, 23:12 UTC; four seed 6307 fits queued | Open all12 together with the complete original12 control union. |
| Frozen-native label posterior P0, allocation | Owned workers exited cleanly | All 3 seed blocks and 12 endpoints completed 1100 updates; all 12 selected states reconstructed | Closed negative: both co-primary gates fail; retain the fixed recipe and all failed contrasts. |
| Query-conditioned value gate, allocation | Parent540114/birth6031118260; child540120/birth6031118285 verified alive | Seed6101 at50/1100 updates,23:26UTC; three seeds/five banks fixed | Open all15 only after clean complete closure and selected-state reconstruction. |

Mol18 and relation12 observations are in `live_mol_relation_progress_root_20261009T2317Z/ALLOCATION_RESULT.json` and `gpu77_connection_recovery_v1/commands/graph_progress_root_20261009T2317Z/RECEIPT.json`; their authoritative observation time is 23:12 UTC. No partial scores were opened. Earlier observations remain preserved. A transport failure never restarts a scientific job. Other users' processes remain untouched.

### Private graph attention

The backbone is existing Polynormer, with four separate hidden-state paths and shared large learned transformations. Small local attention scorer banks and existing global query/key BatchEnsemble factors are private. Each method uses the same architecture, initialization, full WikiCS split0, width512, seven local/two global layers, two stochastic views,1100epochs and three paired seeds.

The comparison changes the gradient recipients. alphaF uses mean individual cross entropy everywhere. allJ uses an equal mixture of individual loss and probability-average ensemble loss everywhere. phiJ applies the mixture to internal dense factors. relationJ applies it to private local scorers and tied global query/key factors; shared transformations retain individual supervision. relationJ minus alphaF is the frozen primary contrast; allJ/phiJ are attribution controls.

The unchanged continuation rule requires positive accuracy signs in all3seeds, mean gain at least0.2percentage points, nonworsening pooled NLL and member-competence safeguards. The recipient group combines local and global parameters, so the study cannot attribute a gain specifically to local edges. Native local GATv1's static-ranking restriction remains. The own/pool mixture, private attention and gradient allocation have established ancestry; no new attention operator, novel risk or competence guarantee is claimed.

Source: `graph_relation_private_credit_source_20261008_v2/`. Benefit limits and an inactive projected-step safeguard: `graph_relation_private_credit_benefit_boundary_20261008_v1/NOTE.md` and `graph_relation_member_competence_projection_design_20261008_v1/NOTE.md`. The projection is not implemented or admitted.

### Molecular ensemble feedback

Mol18 covers all32901official TRAIN and4113development graphs,3paired seeds and100epochs. Six conditions are single, genuinely ordinary independently selected4, O,I,P,G. Candidate I sends ensemble feedback to internal factors while shared features/boundaries learn member supervision. The coefficient0.5, entire roster and complete-family opening remain fixed. Coexecution durations are recorded and do not measure intrinsic speed. No new partial outcome has been opened.

### Staged label posterior

The earlier additive label-correction screen is complete negative. The closed staged P0 recipe preserved a competent native predictor: restore an authentic own-selected native state, freeze it, capture representations/logits once, and train masked-label posteriors on permitted TRAIN labels. Prediction uses the fixed0.2native/0.8label-posterior mixture where one-hop anchor support exists, with exact native fallback elsewhere. No TEST labels enter any role. Four routes are compared with a capable nonlinear joint4head single, untied label routes on the same frozen backbone, and one path. Untied label routes are explicitly **not** an ordinary independent GNN ensemble.

The code is sealed at manifest `ab2be08e841293e95d23d2098c128e0c989087e6a6215101474ce00a2ea0a3e8`. Full-input qualification passed in11.31seconds, with2,258,632,704bytes peak reserved GPU and1,221,582,848bytes peak RSS. It checks masks, frozen native parameters, optimizer ownership, learned restore and cached reconstruction, but computes no development quality. Creation-time disabled seal remains unchanged; root qualification is separate. Complete selected-state serving reconstruction now matches all 12 endpoints: correct counts are exact and float discrepancies are within the declared 2e-5 tolerance. It took 20.95 seconds with zero native forwards or training. This validates serving, not generalization.

All serial seeds 6101/6203/6307 and all four banks completed the full 1100 updates. Owned workers exited cleanly. Both unchanged co-primary accuracy/NLL/member gates fail, so this exact linear one-hop recipe is closed. No extra seed, serving coefficient, epoch or favorable subset is selected to rescue it. The recorded resource limits remain 7200 active seconds per seed, 8 GiB owned GPU/RSS and 12 GiB fresh GPU headroom. This consumed split is exploratory, not unused confirmation.

## Completed evidence that changes the next action

Complete staged P0 development accuracy is 81.62685% for C4, 81.63949% for the nonlinear joint single, 81.65213% for same-backbone untied4, and 81.63949% for one path. C4 paired gains versus the joint single are -0.05688, 0, +0.01896 percentage points; versus untied4 they are -0.03792, -0.03792, 0. Mean pool NLL is worse by 0.00963 and 0.01084 respectively. Both co-primary gates fail.

At its selected endpoints, C4 repairs 2, 3, and 1 native mistakes and introduces zero new errors. Its gain over native 81.58893% is only 0.03792 points. Pooled-only rescues are zero in every seed. Selected label epochs are 14, 9, and 3 despite the complete 1100 update schedule. The combination preserves almost all native decisions and demonstrates no useful ensemble complementarity. Early selection alone does not diagnose an optimizer fault or show that every label correction must fail. Illustrative paired three-seed intervals describe fixed-split optimizer variation and selected development checkpoints; they do not establish graph generalization or remove selection bias.

The original label-correction screen finished all3native trajectories and12bank endpoints. Mean development accuracy is81.5194% for shared4,81.4499% for joint single,81.5636% for same-backbone untied4, and81.5763% for one path. Shared4 fails both primary contrasts and NLL safeguards; no promotion.

A complete24recipe-by3seed Correct-and-Smooth reference selected one common author-autoscale DAD/DAD recipe, correction alpha1 and smoothing alpha0.8,50+50passes, TRAIN anchors only. Native81.5889% becomes81.8797%, with paired gains+0.3223,+0.4551,+0.0948points. This is known-method evidence, not our contribution. It has true zero probabilities on11,9,3nodes and infinite NLL under the declared normalization; no clipping or favorable reselection is added.

The fixed secondary 20% native / 80% normalized-C&S serving adaptation averages 81.78486%, above P0 C4. Its three 100-pass CPU recomputations took 6.59 seconds with no native forward, training or recipe selection. Raw known C&S scores, including infinite NLL, remain unchanged. The adaptation is a secondary serving control, not a new method or published C&S result.

Retrospective error analysis reproduces exact selected counts. C&S repairs/harm are39/22,49/25,34/29. It also repairs nodes outside one-hop TRAIN support. Of5274development nodes,2512lack a one-hop TRAIN-labelled neighbour,1733have exactly one visible class, and1029have multiple visible classes. In the one-class native-error cohort, the visible neighbour class equals truth only49/276,62/270,49/266times. Wider label reach is available structure, not proof of useful evidence.

For this bias-free linear label route, a single visible class supplies only a scalar times one fixed class-logit vector. Attention mass can change confidence but not the route's preferred label class before native mixing. This restricted algebra is not an impossibility theorem for the whole ensemble. A query-conditioned dense value gate could escape it, with established GNN-FiLM ancestry and an extra-feature-classifier confound. The gate15 source passed its four-update engineering qualification; the full15 scientific comparison remains unexecuted, with matched capacity and label-identity-erased controls fixed.

Evidence: `postfamily_label_information_decision_root_20261008_v1/NOTE.md`, `native_CS_selected_recipe_error_analysis_execution_root_20261008_v1/RESULT.json`, `label_support_capacity_diagnostic_root_20261008_v1/RESULT.json`.

## Genuine independent-ensemble reference custody

An independent source/metadata audit identified authentic ordinary Wiki24 independent4 banks6101/6203/6307: separate native models, initializations, optimizers, stochastic streams, own cross entropy and own checkpoint/local restoration. These differ from coupled independent4_contrastive and from same-backbone untied label heads.

Root revalidated all11checkpoint/closure/export/role bindings on the singleton allocation at22:05UTC. All3selected states are present and byte-exact.18.77 holds metadata copies but is not their payload custodian. No checkpoint was deserialized, numerical outcome recalculated or new fit started. Exact custody is in `wikics_independent4_current_byte_check_root_20261009_v1/RESULT.json`; the original source audit and custodian correction remain separate. These references will not be used to replace label-aware/matched-information controls.

## Other closed branches and literature memory

Complete Wiki24 recorded stronger average unit-factor contrastive members but poorer complementary correctness: shared pool81.6332% versus ordinary pool82.0440% on selected development. Context9 is closed negative, ROUTE minus COMMON mean-0.03160points. Direct12 final-head diagnostics supplied no useful gain. Complete original12 plus3canonical SupCon readouts show no supported contrastive accuracy gain: combined alignment/repulsion minus plain mean-0.0695points. Members agree on98.123–99.583% of decisions; pooled-only rescues are zero. These findings concern decision redundancy, not identical embeddings or an inferred training mechanism.

Canonical literature pointer is `literature_memory/CURRENT_SUPPLEMENT.json`, supplement28. The new follow-up expands one existing GOODIE appendix algorithm scope and one pinned bounded author-code scope; it adds zero new paper identities or full-paper/code-audit credit. In those inspected paths, GOODIE retains TRAIN anchors, joint feature gradients, combined-classifier CE and learned embedding aggregation. That bounded difference from our staged recipe is not novelty clearance. The query-value assessment reuses saved GNN-FiLM conclusions, without new read credit. Closest inaccessible graph-ensemble bodies remain unresolved.

## Publication and completion requirements

Closed P0, gate15 source/design, independent source review and qualifier are committed and verified pushed at `89f965bb7815987544b3fd8f6f73a29c88ad0260`, on 8 October at 23:19 UTC. The last verified 18.77 sync remains `7bad36d59373edcb26392640c462dcf2d8f346c9`; root will synchronize the next complete qualified-owner source separately. Historical staged execution source `4bcae2cca71835b70fe8479d52e31738739361c6` and all prior source/results remain retained. The owned staged P0 family is closed negative. The gate15 owner is committed and verified pushed at `77c7343a3a37fde318f3e5d2bedbac2140541ed2` and has begun its fixed scientific comparison. The next both-server sync remains pending.

A positive pilot still requires published competitive comparators, competent singles/ordinary independent ensembles, appropriate capacity/view/objective/information controls, unused split/task confirmation and uncertainty. Papers/reviewers will receive supported claims only. Fresh paper reviewers use the supplied skill, immutable complete evidence, no author history and no requested verdict. No PDF compilation, sudo, GENLINK or scientific use of the relay allocation is authorized.

## Complete-family decision and prospective sources

The complete P0 decision and aggregate summary are in `staged_posterior_complete_decision_root_20261009_v1/NOTE.md` and `SUMMARY.json`. All 12 selected states reconstructed; raw tensors and predictions remain on the allocation. No TEST truth or unused confirmation was scored. Genuine ordinary GNN banks remain a separate reference from same-backbone untied label heads.

`query_conditioned_value_gate_full15_callable_source_20261009_v1/` is the callable implementation of the prospectively frozen query-value proposal. It applies a 64D query gate initialized to the identity to the existing label aggregate and includes matched gated joint single, four fully untied gated routes, one-path secondary and permanent label-identity-erased C4. The five banks require all 15 endpoints at the original three seeds and 1100 updates before opening. The two co-primary checks and the separate erased-label qualification retain the frozen accuracy/NLL/member thresholds.

Every numerical factory/driver defaults to refusal and the creation-time seal remains disabled. One full-input engineering qualification passed in 19.97 seconds: four discarded updates across all five banks, identity/RNG checks, 24 nonzero gate-gradient/change checks, four private U4 Adam owners, permanent erasure, zero-message/native fallback and learned/cache restoration. Peak reserved GPU was 2,275,409,920 bytes and RSS 1,264,807,936 bytes; no development score or native training was computed. Serving/cache discrepancies stayed below the practical tolerance and need no parity investigation. Exact proof/adoption are in `query_value_gate_complete_input_qualification_root_20261009_v1/`. The adapted full15 owner started scientific training on the singleton allocation at 23:25 UTC. At 23:26 UTC its actual child had completed 50 updates across all five banks. Scientific execution source is `77c7343a3a37fde318f3e5d2bedbac2140541ed2`. No partial quality comparison has been opened. Established GNN-FiLM/BE ancestry and the extra feature classifier confound remain; masking message labels does not erase TRAIN supervision already encoded in native H. This proposal cannot reinterpret or rescue the closed P0 screen.

`label_two_hop_shared_kernel_prior_design_20261009_v1/REPORT.md` remains a distinct inactive source-only design for two applications of the same feature kernel. Exact two-edge support need not contain one-hop support; saved radius-two reach is not its measured reach. It requires aligned two-hop controls, fresh qualification and complete-family accounting of extra graph work. No implementation, execution, benefit, measured cost, novelty or quality claim follows from that assessment.
