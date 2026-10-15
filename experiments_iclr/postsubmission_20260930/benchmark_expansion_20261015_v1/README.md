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

The [three-fit private-BN follow-up](pilot_20261015_bn_state_v1/README.md) has
started after the parent completed. It shares native affine weights and isolates
route running buffers, reusing all parent controls. No complete BN result is
available yet. Another previously running GRU acquisition still occupies the
other reserved training slot. No third GPU worker is admitted.

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

[Source binding audit](expanded_binding_audit_v1/README.md) accounts for all511
method/preset entries. At that audit's freeze,494 entries had prepared source and
17 needed new native ports, representing4,940 and170 shared fits. Prepared
source does not imply a qualified training run; later port results are recorded
separately. This audit is a snapshot, so new code does not silently change it.

Completed checks include native node/molecular/NCN updates and checkpoint
restoration, the8-case NCNC CPU bridge, node/molecular interaction checks, and
all40 native node dataset/split providers (covering80 Mono/Cheb bindings).
Provider checks construct no model and score no test labels. BUDDY's native
year2007 context and all8 native CPU variant checks passed. Four NCN interaction
recipes also passed their finite update/VALID/restore checks.

Runtimev3 has a real complete-epoch/interruption/stop-resume qualification.
Its metadata issue is preserved, and runtimev4 repairs that reporting and adds
same-call selected-bank mean export. Runtimev4 has separate source and saved-tensor checks. Its real CiteSeer/Cheb
CPU gate now passed2 complete TRAIN epochs, interruption/recovery, own selected
actual/mean export, and stopped recovery with no additional training. Molecular,
link and GPU scopes remain separate qualification work. Frozen running sources are not modified.

The remaining work includes NCNC/BUDDY context and interaction ports, BUDDY
private BN, native full epochs and selected-serving checks, CUDA resume where
relevant, measured costs/resources, and the expanded executable descriptors.
The predecessor2,480 descriptors pass metadata validation; that does not qualify
the proposed6,460-fit matrix. The three-GPU scheduler exists, but77 deployment
and measured three-GPU assignments remain pending authorization and readiness.

## Statistical design and launch

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
- [Retained method catalogue](catalogue/CATALOGUE.md)
- [Selected prior-result index](prior_result_index.json)
- [Current continuation state](SESSION_STATE.md)
