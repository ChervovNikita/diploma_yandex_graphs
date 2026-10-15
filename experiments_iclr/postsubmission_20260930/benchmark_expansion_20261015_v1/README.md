# Current state — 15 October 2026, three-GPU preparation

The latest molecular comparison is complete: 45 unique fits and 39 prediction
banks. Structured private learning improves pooling but weakens the individual
routes; its primary criterion failed, and a reliable advantage over the
independent ensemble remains unestablished. [Complete results](analysis/structural_private_risk_complete_31/INTERPRETATION.md).

The user substituted server 18.51 for occupied 18.77. MacLink and authenticated
SSH to 18.51 work. Host `olig` has two A100 80 GB GPUs; the dedicated allocation
has the third. The published source revision is synchronized, and the exact
PubMed inputs and native source dependencies have transferred successfully.
A matching Python/PyTorch/PyG runtime and three disjoint eight-fit PubMed queues
are being checked. The intended assignment keeps each seed's native four-model
ensemble and all four shared-model variants on one GPU. Check the dated
[connection and queue record](planning/three_gpu_sync_20261015_v1/README.md)
for actual qualification and launch status. Prepared queues are not running fits.

The 6,460-fit campaign remains separate and requires user review and GO.
Three dormant campaign assignment lists are being prepared; unsupported
historical allocation-only execution bindings are not relabelled as ready.

The NCNC selected-checkpoint replay passed. Composed BUDDY completed its
1,585-update native epoch, then failed in fresh selected serving because a
required native dataset argument was omitted. The checkpoint is preserved.
The narrow source repair and zero-training replay are ready; the training epoch
will not be repeated. The earlier PubMed queue stopped at its BUDDY prerequisite
without launching a scientific fit. Its failure record is retained.

## Earlier research decisions and context

# Current research snapshot — 15 October 2026, release 25 preparation

We are testing trainable routes through one shared native GNN, with small
parameter overhead. The target is better predictions than the native single
model and a genuine independent ensemble. **That target remains unachieved.**

The latest completed MolBACE/GIN comparison has residual-only routes at
74.518 ± 2.660% VALID AUC, versus I4 at 74.847 ± 0.500%. Parameter overhead is
1.0183%, but probability loss is worse and averaging adds no AUC benefit.
The [full result](pilot_20261015_residual_only_private_bn_v1/complete_results_v1/INTERPRETATION.md)
preserves all paired outcomes, uncertainty, costs and exposure limits.

The running six-fit study tests supervised structural responsibilities against
uniform and shuffled assignments. At 17:30 UTC, two fits were complete and
the third was running. No partial quality was inspected. The older GRU worker
is the other GPU worker; only the dedicated allocation is in use.

The native PubMed/Cheb extension has passed three full-graph CPU lifecycle
checks. Its seven queued GPU checks will measure executable behavior and
production cost after the six-fit study. The proposed 24 scientific fits have
not launched. A separate pairwise-ranking source package now includes shared
variants, independent native controls with the same auxiliary loss, and a
TRAIN-only support inspector. All four CPU lifecycle checks passed, with eight total training updates and twelve whole-VALID passes. Its proposed 21 fits have not launched.

Two additional scoped primary reads establish prior shared/private rankers
and task-specific AUC risks. They narrow the novelty claim; they do not supply
evidence that our particular same-task ensemble improves predictions.

The 6,460-fit campaign remains unlaunched and requires user review and GO.
Concrete remaining work includes composed BUDDY qualification, final selected
VALID serving coverage, and measured three-GPU placement and budget. Passed
native BUDDY, NCN and molecular checks remain accepted and will not be repeated.

- [Research direction and next decisions](research/CURRENT_DIRECTION_20261015.md)
- [Actual pairwise qualification](PAIR_QUALIFICATION_SUMMARY_25.json)
- [Pairwise implementation](research/molbace_pairwise_competence_v1/README.md)
- [Closest ranking prior work](research/pairwise_competence_prior_followup_v1/NOTE.md)
- [Composed BUDDY handoff](analysis/composed_buddy_qualification_handoff_v1/HANDOFF.md)

The dated material below preserves the earlier study decisions and results.

---

# GNNM benchmark preparation and bounded research

The goal is a learnable ensemble with one shared native GNN and better
predictions than the original shared baseline and ordinary independent
ensembles. We preserve each reference backbone and isolate changes to the
wrapper, private corrections, objectives and prediction aggregation.

## Completed molecular study

The official MolBACE scaffold split and native OGB GIN were used for three
paired seeds101/202/303. The model has5 layers, width300, full atom/bond
features, mean readout, dropout0.5, and Adam with LR0.001/WD0. Each cohort trains
four independent native models plus base GNNM, class alignment with a variance
floor, nonlinear private input corrections, and their combination. The single
baseline is independent member0; every model selects its own checkpoint.

| Method | Mean VALID AUC ± SD (%) | Mean BCE |
| --- | ---: | ---: |
| Native single | 73.32 ± 1.17 | 0.779 |
| Independent ensemble | 74.85 ± 0.50 | 0.890 |
| Base GNNM | 67.78 ± 4.39 | 0.719 |
| Alignment + variance | 71.17 ± 2.66 | 0.551 |
| Private input correction | 71.76 ± 3.91 | 0.941 |
| Both ingredients | 69.28 ± 1.80 | 0.882 |

[All24 fits and the complete analysis](pilot_20261015_molbace_v1/complete_analysis_v1/REPORT.md)
are now available. Private corrections improve GNNM in3/3 seeds but do not beat
the independent ensemble on average. Geometry improves AUC in2/3 seeds and
mean BCE; combining the ingredients gives a smaller AUC improvement. This is
exploratory evidence from selected validation on one split. It does not establish
statistical significance, novelty or transfer. All outcomes are retained.

Training lasted149–210 complete epochs per fit, under minimum100/cap1000 and
patience50 after the minimum. The study records3,680 epochs and139,840 updates,
zero failed fits,7.81 elapsed hours, and7.75 summed worker hours. No fit hit its
cap. Selected checkpoints span epochs2–160. Every curve was checked against
its recorded first-best AUC selection. Validation stopping is not proof of
optimization stationarity. The curves stay on the allocation; compact results
and reproducible analysis source are committed.

The [three-fit private-BN follow-up](pilot_20261015_bn_state_v1/complete_analysis_v1/INTERPRETATION.md)
also finished. Each route keeps its own running statistics while all learned
native affine weights remain shared. Mean selected VALID AUC was **73.61 ± 1.06%**,
up 5.84 points from base GNNM and 1.23 points below I4. All three seeds improved
over base GNNM. Mean BCE increased to 1.086, so the ranking improvement came with
a worse probability-loss score. Training used 464 complete epochs and 17,632
updates, with no cap stop. This is a known normalization control, not a novelty
claim. Different selected epochs prevent an inference-only causal explanation.

The [complete route diagnosis](analysis/complete_route_diagnostic_v1/REPORT.md)
uses saved predictions from all 24 parent fits. Mean member AUC was 73.41% for
I4, 62.01% for base GNNM, 64.24% with geometry, 66.07% with private corrections,
and 61.00% with both. The combined method had fewer ranking mistakes shared by
every route but weaker individual predictions and a weaker average than either
ingredient alone. Increasing diversity is therefore insufficient in this study.
We need to preserve route quality as well. These are selected-validation
diagnostics, not independent confirmation results.

The [BN route follow-up](analysis/bn_route_diagnostic_v1/REPORT.md) reconstructed
all three selected checkpoints without training. Mean member AUC reached 72.63%,
with a 0.99-point gain from averaging. Each seed improved mean member AUC over
base GNNM while remaining below its independent control. This motivates checking
geometry and corrections under matched private normalization before rejecting
their combination. The nine-fit follow-up is now complete. All seeds and all 15 reused factorial
controls are in the [complete results](pilot_20261015_bn_geometry_correction_v1/complete_results_v1/INTERPRETATION.md).
New fits used 1,373 full epochs and 52,174 updates, or 4.27 summed worker-hours;
all stopped by patience after 149–178 epochs. Nine new route collectors and
all 30 prediction banks were analyzed with zero TEST access.

With private BN fixed, geometry reaches 73.65 ± 1.64% AUC, correction
72.19 ± 1.41%, and both 73.03 ± 1.61%, against private BN alone
73.61 ± 1.06% and I4 74.85 ± 0.50%. Geometry's earlier gain becomes +0.04
points, and correction's becomes −1.43 in all three seeds. The interaction
changes sign to +0.81, but the combination still loses 0.59 points to neither
ingredient. Mean BCE is 1.122/0.816/1.030, respectively; preserve these tradeoffs.

Geometry under private BN lowers mean member AUC from 72.63% to 72.19%, while
raising the averaging benefit from 0.99 to 1.47 points. I4 gives 73.41% member
AUC and 1.44 points from averaging. The near-tie therefore does not demonstrate
better individual learning. No combination is promoted as a native-I4 win.

The [completed fixed-aggregation comparison](analysis/complete_fixed_aggregation_v1/REPORT.md)
now includes all three seeds and all completed methods, including I4. Averaging
probabilities raises base GNNM AUC from 67.78% to 69.01%, while I4 reaches 74.90%
under the same rule. Private BN reaches 73.47% with probability averaging versus
73.61% with the original logit average. No fixed readout closes the ranking gap.
Probability averaging reduces correction-model BCE from 0.941 to 0.638, showing
why probability loss and ranking should both be retained. This uses saved selected
VALID predictions, seven banks per seed, zero new fits or forwards, and 7.81 CPU
seconds. It changes no original score or proposed primary serving rule.

The [fixed-weight normalization assignment analysis](analysis/bn_bank_assignment_v1/REPORT.md)
compares own, tied and cyclically exchanged TRAIN-derived banks at the same
three selected checkpoints. Mean AUC is73.61% with own banks, 59.06% with tied
banks and 57.98% averaged across the three cyclic assignments. Own versus tied
improves AUC in all three seeds, but one cyclic assignment improves one seed;
BCE does not uniformly favor own over tied. This is a development diagnostic,
not an independent-ensemble win. It used 300 CPU VALID forwards, zero training
updates and 30.82 seconds. A [scoped prior-art inspection](literature/switchable_norm_scope_v1/READING.md)
found separate running statistics with shared affine weights in the author
implementation of universal slimmable networks. This strengthens the reason
to treat private BN as a necessary control rather than the claimed novelty.

The [stronger common-mixture control](analysis/bn_mixture_moments_completed_v1/INTERPRETATION.md)
adds between-route mean variation to the average saved variance. At the same
weights it reaches 68.45% mean AUC, versus 59.06% for the arithmetic tie and
73.61% for own banks. The gap therefore shrinks from 14.55 to 5.16 points;
own banks remain ahead in all three seeds. BCE and Brier favor the mixture in
one seed and own banks in two. This used 120 CPU VALID forwards and 25.76 seconds,
without training or TEST. The mixture is a saved-EMA approximation; it is not
exact pooled-history recalibration. The full report preserves the earlier
controls, every seed and the limits of this fixed-weight comparison.

The [normalization design note](research_next_decision_v1/DESIGN_IMPLICATION.md)
identifies a decision to resolve before committing the large compute budget.
Original GNNM stays unchanged. If private BN is adopted for an extension, its
mechanism controls must use the same policy, and private-BN GNNM must remain a
reference. The completed nine-fit study informs that decision; the exact per-architecture
policy migration remains under review. The current
roster is unchanged; three seeds on one molecular setting do not justify a
universal normalization rule.

The [BatchEnsemble batching inspection](literature/batchensemble_bn_batching_v1/READING_AND_DESIGN.md)
adds a concrete implementation explanation. Sequential routes use their own
TRAIN batch statistics but update one shared running history. The inspected
vectorized CIFAR BatchEnsemble implementation instead normalizes a combined
route batch. These are different training rules when native BN is present;
concatenating routes is not an equivalent speed optimization by default.
Separate running state has prior art, and this derivation does not attribute
all of the observed AUC gain to one cause. The
[TRAIN-semantics note](analysis/bn_train_semantics_source_v1/NOTE.md) adds an
important limit: with matched parameters, batches, RNG and route order, changing
only running-buffer ownership leaves native TRAIN batch-normalized outputs and
updates mathematically unchanged. It changes evaluation state and can change
checkpoint selection/stopping. This is a conditional source argument, not a
numerical equivalence experiment or evidence of improved TRAIN gradients.

A [focused context-target analysis](literature/context_target_novelty_followup_v1/REPORT.md)
clarifies another proposed direction. Routes learn relationships derived from
different observed graph contexts while sharing the native predictor. Existing
multi-view and molecular-similarity methods supply clear ancestry. The common
versus split-target comparison also changes target entropy and can change
whether the target is representable by cosine similarities. A lower auxiliary
loss or a win over the common-target arm alone cannot prove useful prediction
specialization. These limits and the exact analytic example are retained before
the broad experiment; no new method or accuracy result is claimed.

[Native molecular label support](molecular/RARE_CLASS_SUPPORT_NOTE.md) also
shows why a cross-task port needs more than a callable loss: with the unchanged
batch size32, only33.81% of random MolHIV batches have two distinct positive
molecules. We will report actual auxiliary support without changing the native
sampler or treating these probabilities as model results.

## Full proposed design

The [exact method coverage table](benchmark_review_map_v1/README.md) shows all
25 recipe definitions, their controls, and where each task includes them.

The [expanded roster](expanded_study_design_v1/README.md) contains6,460 proposed
physical fits across7 datasets,8 architecture families and27 fixed reference
presets. Every compatible preset has separate source-horizon reproductions,
matched native M1/I4 and original GNNM. M1 reuses I4 body0; each independent body
owns its weights, optimizer, RNG and checkpoint selection.

| Task | Datasets | Reference architecture families | Proposed fits |
| --- | --- | --- | ---: |
| Node classification | CiteSeer, PubMed, Coauthor-CS, Coauthor-Physics | PolyFormer, source Mono/Cheb presets | 1,920 |
| Molecular graph classification | ogbg-molbace, ogbg-molhiv | GCN, GCN-VN, GIN, GIN-VN; source dropout0/.5 presets | 3,840 |
| Link prediction | ogbl-collab | NCN, NCNC, BUDDY | 700 |

The total includes270 source-reference fits,1,080 independent native bodies
and5,110 shared-method fits. The25 recipe definitions include original GNNM,
necessary controls and known ingredients; they are not25 novel inventions.
The24-fit pilot and3-fit BN follow-up are separate development studies.

The candidate families test task geometry, persistent structural training
responsibility, common versus route-specific factual context, nonlinear private
input corrections, training of the ensemble prediction, learned aggregation,
and private native BN state. Four selected combinations test interactions.
Exact rules, controls and native hyperparameters are in the roster and
[NATIVE_SETTINGS.json](expanded_study_design_v1/NATIVE_SETTINGS.json).
No hyperparameter search or original-five paper rerun is included.

## Execution readiness

The [fourth compiler version](expanded_dispatcher_v2/HANDOFF_v4.json) prepares
**all 6,460 source descriptions**, with zero remaining source/context blocks.
It updates 100 BUDDY source bindings and reuses 6,360 descriptions, preserving
the scientific configurations. The 270 paired M1/I4 groups retain separate
native bodies. These are preparation counts, not completed training runs.

Real-data checks cover the additional NCNC and BUDDY objectives, interactions
and private BN, including BUDDY production input construction at a nonzero seed.
The source-equivalent BUDDY recipes inherit scoped coverage; they are not
misreported as individually executed checks. All failed attempts remain saved.

The production runtime has passed two complete node epochs with recovery and
selected predictions. Its molecular counterpart also passed two complete epochs
on CPU: 76 updates, four whole validations, three constructions and two exports
in 45.76 seconds. The NCNC successor passed two complete epochs on CUDA:
34 updates, four whole validations, three constructions and two exports in
181.32 seconds. These checks cover interrupted epoch-boundary recovery and
finalization from the selected checkpoint. They are execution evidence only.

The [launch preparation](expanded_launch_preparation_v1/HANDOFF.json) accounts
for 6,460 fits plus 270 I4 prediction exports. Exports require no training.
The dispatcher implements durable work claims, restart handling, completed-run
skipping and at most two resource-compatible jobs per GPU. Its 38 subprocess
checks passed. The current source descriptions have 6,730 callable execution
candidates. The BUDDY provider now handles the shared model's null body label
correctly; this changes metadata only. Native training functions are unchanged.
The final NCN and BUDDY lifecycle checks are prepared, with respectively 34 and
3,170 optimizer updates, four whole validations and two existing prediction
exports per case. They remain unlaunched while both permitted GPU slots are
occupied. MolHIV resource measurement and final resource assignments are also
pending. Earlier node, molecular and NCNC lifecycle evidence is reused.

The [portable runner](portable_dispatcher_v3/README.md) has a single command
interface per GPU, permanent task ownership and shared host resource accounting.
Its actual allocation metadata generation now succeeds for all 6,730 tasks.
The complete 6,460-descriptor audit preserved every scientific configuration.
Two earlier preparation failures are retained: a host-config pin mismatch and
an omitted native-callback path declaration affecting link jobs. The coherent
repair uses the same path rules during preparation and execution; all 16 native
runtime modules remain byte-identical to the preceding portable package.

The [reporting rebind](benchmark_reporting_portable_rebind_v1/README.md) also
completed against that actual generation. It preserves all 6,190 expected
prediction rows and the native/data/controller/evaluator identities. The build
used no models or datasets. Full generated files stay on the allocation;
[generation](PORTABLE_PROJECTION_RUN_13.json) and
[reporting](REPORTING_REBIND_RUN_12.json) receipts record their exact locations.
These are successful metadata builds; native/resource admission remains pending.
The default ownership puts all tasks on the allocation for preparation and is
not the final efficient three-GPU assignment. Both future 18.77 workers remain
disabled.

The full campaign has not launched. Only the dedicated allocation is authorized;
the two future 18.77 workers are disabled and have not been probed. The measured
costs cover only a subset of the proposed settings. A campaign-wide completion
time is not yet supported, and two concurrent jobs do not imply twice the
throughput. Resource measurements, extrapolations and missing timings will be
listed separately in the review package.

The [resource budget](resource_budget_draft_v1/README.md) gives exact workloads
and measured-versus-unknown costs for all 27 presets. The matched training
minimums alone require 237,211,000 optimizer steps across the proposed fits.
Shared steps include several routes, so this is not a FLOP or elapsed-time
estimate. The proposal remains unchanged. A staged approval option would run
one cohort across all settings and methods first (646 fits plus 27 exports),
then the remaining nine cohorts (5,814 fits plus 243 exports). This is a review
option, not permission to prune methods after seeing favorable results.

## Statistical design and launch

The [reviewed reporting protocol](benchmark_reporting_protocol_v1/README.md)
keeps paired seed/split differences, complete outcome counts and descriptive
95% paired bootstrap intervals within each preset. Missing or unresolved pairs
withhold the complete-preset interval. No significant-win label or cross-preset
winner substitution is permitted. The [initial exposure ledger](benchmark_reporting_protocol_v1/EXPOSURE_LEDGER.json)
records verified historical Physics TEST exposure; the other six histories
remain unknown. Current development uses TRAIN/VALID with TEST scoring closed.

Training a source-reference model does not itself verify the paper's reported
TEST metric. The final review package must also specify selected-checkpoint
scoring with the original split, evaluator and task-specific test inputs. That
[scoring source](final_selected_scoring_v4/README.md) is now implemented and
reviewed: native role inputs, each own selected state, learned aggregation,
member outputs and independent-ensemble exports. It checks complete selected
VALID reconstruction before TEST. Actual role metadata and dispatcher/reporting integration are complete. Three
genuine molecular states passed whole-VALID CPU reconstruction. Cross-task
native numerical/resource qualification remains incomplete. This adds
no training fits; no validation number will be presented as a reproduced test
result.

A [runnable metadata reporter](benchmark_reporting_implementation_v1/README.md)
now preserves all planned pairs, failures and capped outcomes, and generates
1,170 declared contrasts across 27 presets. Nine focused metadata checks passed;
the full compiler registry now retains all 6,190 expected predictor rows,
including same-bank diagnostics. The empty campaign report retains every
missing outcome and withholds all 1,170 intervals. Selected-export and cost
integration remains source-only until its real lifecycle check completes.
Its three-seed pilot extraction does not receive a ten-cohort interval.

The broad matrix is exploratory. The [confirmation proposal](confirmation_protocol_proposal_v1/README.md)
would select at most one method/setting using development VALID, then freeze
20 fresh paired cohorts for a separate120-fit confirmation study. It is a
proposal, needs its own budget and review, and is not silently added to6,460.
Its exact sign tests concern repeat-training win probability, not mean effect
or universal transfer. Historical test exposure must be recorded honestly.

The giant campaign has not launched and still requires the user's review and
explicit GO after the code, comparisons, convergence rules and resources are
ready. Only the dedicated allocation is currently authorized. Original paper
scores remain unchanged. Fresh manuscript review follows substantial supported
paper progress; no acceptance verdict is requested or promised.

- [Scientific rationale](RESEARCH_SYNTHESIS_20261015.md)
- [Literature conclusions](literature/SCOPED_CONCLUSIONS.md)
- [Cross-Batch Memory and rare-class supervision](literature/xbm_support_followup_v1/READING_AND_DECISION.md)
- [Retained method catalogue](catalogue/CATALOGUE.md)
- [Selected prior-result index](prior_result_index.json)
- [Current continuation state](SESSION_STATE.md)

## Actual evaluation-population preparation, 15 October

The CPU preparation now produces native role contracts for all **27 presets,
99 preset/split proofs and 43 distinct populations**. Node IDs retain all ten
literal source splits; molecular IDs retain native batch/skip behavior; Collab
retains all 46,329 positive and 100,000 negative TEST queries in native order.
The work accessed TEST population metadata, including declared label/container
access needed by original providers. It computed no predictions or scores.

The first attempt completed six datasets but stopped on the missing Collab
TEST sibling. The original file was recovered from the retained official archive
only after matching its TRAIN/VALID bytes. The second attempt exposed OGB's
NumPy-to-tensor split conversion; the declared successor now uses that exact
native conversion. Both failures and costs remain. All six completed providers
were reused. The successful completion took 11.79 seconds; the two preceding
supervisors took 65.59 seconds combined. Peak observed provider RSS was 4.00 GB
for Coauthor-Physics. These are CPU population costs, not inference estimates.

[Compact execution evidence](POPULATION_COMPLETION_V2_RUN_14.json) identifies
the complete remote index and output manifest. Full arrays stay on the server.
No dataset/model/training setting or original paper score changed.

The repaired final scorer preserves original producer/checkpoint identities,
then attaches BUDDY VALID/TEST operation ownership explicitly. Native OGB split
conversion is retained. Three genuine molecular selected states passed CPU
whole-VALID replay in 23.26 seconds: zero optimizer, TEST or CUDA calls. This
covers M1, original GNNM and private BN at their actual producer scope, not
all current-runtime states. See [CPU replay evidence](CPU_SELECTED_VALID_RESULT_15.json).

The coherently bound one-command sequencer passed eight actual orchestration
checks in 4.29 seconds, using three standard-library fixture children. Restart,
failure, duplicate prevention, readiness recovery, offline delivery and ledger
handoff were exercised without models or network access. The first driver's
wrong-config negative check failed before native work; its corrected successor
and failure remain. See [execution evidence](FINAL_ORCHESTRATION_V2_RUN_15.json).
Native CUDA resource/serving checks and final placement remain pending. The
campaign has not launched.

The [quality/diversity note](literature/quality_preserving_diversity_followup_v1/REPORT.md)
finds no justification for adding another repulsion loss now. A pointwise rule
requiring every binary route to beat its own mean prediction would force equality
at zero slack. The existing matched private-correction comparison, own-member
loss versus half own-member/half pooled loss, already tests the useful question.
Its ancestry and possible learner-collusion failure are recorded; no new paper
read, new method, pilot, or predictive advantage is claimed by that analysis.

The actual population contracts now also compile against the current reporting
registry into all **6,730 final-stage task descriptions**, with 6,190 expected
predictor rows and zero added training fits. This 0.65-second metadata build
performed no inference. The full 14.36 MB roster stays remote; see
[execution receipt](FINAL_POPULATION_ROSTER_RUN_14.json). The combined launcher
successor also passes a focused interrupted-readiness recovery check. Final
runtime/source binding is complete; native resource qualification remains pending.

## Scientific directions retained after the controls

Two scoped primary reads of RIDE and SADE are recorded in the
[shared-competence review](literature/shared_competence_primary_followup_v1/REPORT.md).
Distinct learning targets can support useful routes, but repulsion alone does
not guarantee competence; binary prior offsets can preserve population ranking.
These readings add no recipe or claimed novelty to the current roster.

The [gradient-history clarification](literature/shared_gradient_evidence_clarification_v1/NOTE.md)
separates completed optimizer synchronization and saved-state gradient diagnosis
from unimplemented gradient-surgery proposals. No completed PCGrad, CAGrad,
MGDA or GradNorm intervention was found in the audited record. Whether actual
shared Adam steps harm competent routes remains an open hypothesis. Private BN
changes inference running state; it does not settle that training question.

## Research continuation after the completed factorial

The [history shortlist](research/quality_history_shortlist_v1/REPORT.md) audits
native versus equipped/factorized controls and preserves successful and failed
transfer outcomes. It was written before interpreting the complete factorial;
its old correction gain is qualified by the completed matched-BN result above.
Native-initialized additive routes with optional BE factors remain an untested
mechanism, not a conclusion from equipped-I4 comparisons.

The [shared-update proposal](analysis/shared_gradient_private_progress_proposal_v1/PROPOSAL.md)
first diagnoses actual Adam displacement and private progress on three complete
TRAIN epochs. Root authorized implementation of that finite diagnostic, without
VALID/TEST scoring or a new full training grid. Any later projection comparison
must show prediction utility beyond known PCGrad/projection controls; a local
first-order bound alone is insufficient. A minimal residual-only comparison
using exact completed controls is also being prepared.

The MolHIV GIN-VN whole-epoch resource check started at 14:01 UTC on the allocation.
It measures 1,029 fixed updates and two full VALID passes, without TEST. Its
one-epoch score is an execution check, not scientific efficacy evidence. The
older private-GRU acquisition remains running. Neither job starts the campaign.

## 15 October update: completed gradient diagnostic and residual-only pilot

The matched-BN factorial is complete. A follow-up using the actual Adam
updates found 0/456 qualifying harmful-interference events; finite shared and
joint updates lowered every measured member loss. Negative gradient-pair dots
were insufficient to diagnose harmful training updates. See
`analysis/shared_gradient_shadow_completed_v1/REPORT.md` for scope and costs.
The next frozen three-fit candidate removes BE and retains native-initialized
additive routes with private BN. Its exact CPU lifecycle passed two optimizer
updates and three full VALID passes, without scientific selection. Neither
this candidate nor gradient projection has an established accuracy advantage.

MolHIV resource measurement passed:1029updates,582.446sTRAIN,613.996soverall.
At this one setting,100TRAINepochs imply16.18worker-hours before validation;
there is no whole-campaign ETA. The two NCN/BUDDY lifecycle jobs remain queued.
The large6460-fit campaign still requires user verification and GO.

## Live status — 15 October2026,14:47UTC

Release17 was pushed as8b9f0a7d3cb3e631ded98222d6858da8b614e08f.
The qualified residual-only three-fit study is running serially; seed101 reached
53epochs/2014updates. It adds1.018%learned parameters. No partial accuracy claim.
Its owner is846365/start6088444359,firstworker846371/start6088444385.
The olderGRU remains the only otherGPUworker. No thirdGPUworker is admitted.

NCN's queued lifecycle failed before model construction/update on the exact
engineering-only parent-descriptor path. Failure and claim are preserved;
portable_runtime_authority_v4 is being prepared. BUDDY remains unexecuted.
Generic normalizationV2 source fixes two real checkpoint ownership problems
found inV1; focused actual-parent scalar regressions pass. Both versions are
preserved. Genuine native/runtime/scorer integration still awaits qualification.

## Continuation — 15 October, 15:06 UTC

The residual-only pilot has completed seed101 (186 epochs, patience stop), and
seed202 is running. All three fits must finish before the planned joint analysis.
The selected-VALID addition is ready: it will collect only three new banks and
reuse the completed30-bank analysis, retaining39 unique training fits overall.
The prediction rule and all five declared comparisons stay fixed.

The NCN parent-descriptor repair passed42 runtime metadata checks,17 driver
checks and guarded verification of166 actual source receipts, both parent JSONs
and the preserved zero-model-work failure. Native NCN/BUDDY qualification still
awaits a compatible free worker slot. The residual pilot and older GRU job occupy
the two allowed slots; no third worker is started. These checks add no scientific
accuracy evidence.

Next scientific work is a convergence audit of the36 completed fits and a
focused review of task-responsibility and learned-aggregation mechanisms. These
will guide a finite representative pilot after the residual-only outcome. The
full6,460-fit campaign remains unlaunched and awaits user GO.

## Completed convergence audit and normalization integration

The [36-fit convergence audit](analysis/completed_molbace_convergence_v1/FINAL_INTERPRETATION.md)
replayed every recorded selection decision. No fit hit its cap; all stopped by
validation patience while TRAIN loss kept falling. Final 50 mean AUC improved
in 16/36 versus the preceding 50; BCE worsened in all 36. No last 50 record is expected
under patience and is not independent evidence of stationarity. Additional epochs
could produce later recovery, but a blanket extension is unsupported by these
observations. Original outcomes and scores remain unchanged.

[Composed normalization runtime](composable_normalization_runtime_v1/README.md)
now includes the actual producer, checkpoint restore, selected serving and two
finite native qualification commands. Root source checks and an
[independent code review](analysis/composable_runtime_independent_review_v1/REPORT.md)
found no actionable blocker. Actual molecular and BUDDY execution is still
pending; neither source checks nor this review establishes native success.

## 15 October, release24 preparation: PubMed native execution

All three residual variants passed full-graph CPU update/restore checks.
`PUBMED_CPU_SUMMARY_24.json` links all outcomes and costs: six optimizer updates,
nine whole-VALID calls, initial native parity and fresh restore. Actual learned
parameters are592435 native plus8192 private. These are engineering results,
not training quality. GPU execution/cost and a separately admitted24-fit pilot
remain pending. The code is published in release23.

The current MolBACE six-fit study continues under its frozen protocol. The
separate pairwise-competence note proposes a task-ranking objective, including
matched independent-ensemble objective controls; it launches no new runs and
changes no ongoing study. See`research/pairwise_competence_synthesis_v1/NOTE.md`.

The24-fit PubMed source package and isolated production-cost probe are now
implemented, reviewed and source-checked. Actual allocation preflight validates
all24 descriptors without a model or fit. A bounded engineering queue is
prepared to wait for the current six-fit study, then qualify CUDA behavior and
measure production costs; its launch receipt determines actual execution.
The24-fit pilot itself remains unlaunched and unbudgeted.

## Release29 source preparation

The frozen normalization migration generator preserves the proposed fit count
and native definitions while replacing 2,890 candidate contracts with the
reviewed private-normalization composition. Actual metadata generation and
downstream reporting/dispatch binding remain separate from campaign admission.
The inherited composed scorer now has a narrow typed-extra-state successor.
The learned-private-noise [proposal](research/stochastic_private_adapter_audit_v1/NOTE.md)
is retained as a distinct quality experiment for preparation. No new quality
result or large-campaign GO is claimed.

## Release30: actual preparation and private-noise qualification

The normalization migration, reporting registry and dispatch plan were generated
on the allocation with zero model work. The current roster remains 6,460 fits
and 270 I4 export tasks; 2,890 fit descriptors use composed normalization.
All 27 reporting profiles cross-check against the complete dispatch roster.
Actual generation took 34.73, 49.02 and 55.11 seconds respectively. The initial
migration attempt failed before creating outputs because two decision files
were absent; the dependency repair and failure remain recorded.

The fixed- and learned-private-noise implementations both passed two genuine
CPU updates, fresh VALID restore and next-noise replay. This is engineering
evidence only. New stochastic training remains unlaunched; the prepared
PubMed transfer comparison remains the next scientific priority.

No massive-run admission follows from metadata success. Composed BUDDY, NCNC
selected serving, runtime reporting witnesses, custody review and measured
resource placement still require completion. All 18.77 execution is disabled.

## Release31: complete structural comparison and measured transfer costs

The six-fit structural study and selected-state analysis are complete: 39 banks
from 45 unique fits, no warnings, no TEST. The frozen primary criterion failed.
Structured AUC 74.469% improves on shuffled 72.112%, but remains just below
uniform 74.518% and I4 74.847%. Averaging benefit improves from −0.043 to +0.455 points,
while mean member AUC falls from 74.560% to 74.014%; repairs and harms nearly cancel.
See [complete interpretation](analysis/structural_private_risk_complete_31/INTERPRETATION.md).

All seven PubMed CUDA lifecycle/cost checks passed. The frozen 24-fit study
remains unchanged. Conditional TRAIN+VALID projections are 10.08–12.16 hours
at 2,000 epochs and 50.34–60.73 hours at 10,000, before checkpoint I/O/setup. The
resource candidate is closed until root admission; no full-campaign GO follows.

NCNC selected CUDA replay passed in 12.54 seconds: one construction/restore,
four encoder forwards, eight predictor calls, one whole VALID pass and zero
TRAIN updates; saved logits and same-bank mean match exactly. The composed
BUDDY check is running separately. Runtime checks are engineering evidence.
