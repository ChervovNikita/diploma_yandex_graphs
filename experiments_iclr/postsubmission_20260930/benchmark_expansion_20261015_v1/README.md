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
their combination. The nine-fit follow-up started on the allocation at 09:36 UTC on 15 October. It
tests geometry, corrections and both together under private BN, with the same
three seeds and 15 completed factorial controls. It runs one new fit at a time.
No completed-family result is available yet. The scope and reporting criteria
were frozen before launch; it remains separate from the 6,460-fit campaign.

The [completed fixed-aggregation comparison](analysis/complete_fixed_aggregation_v1/REPORT.md)
now includes all three seeds and all completed methods, including I4. Averaging
probabilities raises base GNNM AUC from 67.78% to 69.01%, while I4 reaches 74.90%
under the same rule. Private BN reaches 73.47% with probability averaging versus
73.61% with the original logit average. No fixed readout closes the ranking gap.
Probability averaging reduces correction-model BCE from 0.941 to 0.638, showing
why probability loss and ranking should both be retained. This uses saved selected
VALID predictions, seven banks per seed, zero new fits or forwards, and 7.81 CPU
seconds. It changes no original score or proposed primary serving rule.

[Native molecular label support](molecular/RARE_CLASS_SUPPORT_NOTE.md) also
shows why a cross-task port needs more than a callable loss: with the unchanged
batch size32, only33.81% of random MolHIV batches have two distinct positive
molecules. We will report actual auxiliary support without changing the native
sampler or treating these probabilities as model results.

## Full proposed design

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

The [third compiler version](expanded_job_binding_v1/HANDOFF_v3.json) prepares
**all 6,460 source descriptions**, with zero remaining source/context blocks.
It creates 100 descriptions and reuses 6,360, preserving the 2,480 predecessor
contracts. The 270 paired M1/I4 groups retain separate native bodies. These are
preparation counts, not completed training runs.

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
checks passed. The new source descriptions still need to be attached to the
final qualified runtime and resource assignments. Remaining source-policy and
callback coverage is being consolidated from existing evidence.

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

