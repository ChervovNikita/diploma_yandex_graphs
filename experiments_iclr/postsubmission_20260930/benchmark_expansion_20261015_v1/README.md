# GNNM benchmark expansion and bounded research

The goal is a shared-backbone ensemble with better predictions than base GNNM
and a competitive ordinary independent ensemble. Most native weights remain
shared. Every modification needs an explicit mechanism, matched comparisons
and attribution; no current extension is declared a new successful method.

## Common comparisons

Every suitable source preset retains these comparisons:

1. A separate native single-model run with the reference paper/release horizon
   and selection protocol, to check the published baseline. Any unresolved
   source setting or protocol adaptation is disclosed.
2. A matched single native model and four genuinely independent native models,
   with separate weights, optimizers, RNG streams and checkpoint selectors.
   The matched single is member zero, avoiding duplicate training.
3. Base four-route GNNM around the same native model, with shared native weights
   and declared private input/output factors.
4. Explicit GNNM candidate methods and limited ingredient combinations. A
   node-only implementation is not a scientific reason to exclude molecules
   or links; their objectives need appropriate task targets and reductions.

The first [expanded scientific roster](expanded_study_design_v1/README.md)
proposes **6,460 physical fits** across7 datasets,8 architecture families and27
fixed source presets:270 source references,1,080 matched independent bodies and
5,110 shared-method fits. It preserves all predecessor rules and avoids counting
M1/body0 or exact mean-J aliases twice. This is a prospective design, with native
integration/resource/statistical gates still open; it is not ready or approved
for launch. The old2,480-fit proposal and retired80,850-fit launcher remain history.

## Running representative pilot

The frozen pilot uses the official MolBACE scaffold split, native OGB GIN,
5 layers, width300, full atom/bond features, mean readout, dropout0.5, Adam
LR0.001 and WD0. No native hyperparameter search is performed.

| Trainable method | Mechanism |
| --- | --- |
| Base GNNM | Original shared body and private boundary factors; mean native member losses |
| Graph alignment and variance | Class-balanced graph-prehead supervision and a conditional variance floor, inspired by SupCon/BCL/VICReg; native task loss retained |
| Private input correction | Per-route rank8 nonlinear residual after complete AtomEncoder output; orthogonal initial down projections, zero initial up projections |
| Both ingredients | Same input correction and graph objective together; measures interaction |

For each of seeds101,202,303, train four independent native bodies and the
four shared methods above: **3 × (4 + 4) = 24 physical fits**. The single-model
comparison reuses body0. This small study tests a mechanism and its interaction;
it is not an estimate of broad generality. The separate paper-reproduction
fits remain in the full benchmark and are not replaced by this extended pilot.

Every fit traverses whole TRAIN epochs and whole native VALID, with a 1,000-epoch
cap, a minimum100 epochs, and patience50 after the minimum. Caps without a
plateau are inconclusive for convergence. Each fit selects its own first best
VALID checkpoint. Test scores remain closed. The queue runs at most two total
own training workers on the authorized allocation, reserving a slot for an
existing older acquisition while it is live. No 18.77 work is performed.

The pilot was launched on the dedicated allocation. Raw server timestamps and
compact observations are saved under [pilot_20261015_molbace_v1](pilot_20261015_molbace_v1/).
No new prediction-quality result is claimed before complete paired runs and
selected-checkpoint analysis. All trials, failures and costs will be retained.

## Runtime and analysis

[runtime/API.md](runtime/API.md) explains complete epochs, source-specific
selection, independent checkpoint ownership and explicit epoch-boundary resume.
Native parameter identity and optimizer ownership are checked. Runtime/source
hashes bind checkpoints and selected predictions. There are no automatic
scientific retries. Running v1 retains its known stopped-resume limitation and
has no automatic resume; the separate v2 guard below preserves v1 unchanged.

The running v1 frontend supports node and molecular fits. Prepared
[runtime_v2](runtime_v2/API.md) adds explicit source-specific link dispatch,
endpoint-aligned I4 exports, and finalization of already-stopped checkpoints
without another TRAIN epoch. It does not silently migrate v1 checkpoints.
Its actual model/resume/link qualification remains pending. A separate allocation
audit verified the narrow omitted OGB TEST-iterator RNG effect without accessing
TEST data; this is not whole-paper trajectory reproduction.

[Native link integration](link_native_integration_v1/README.md) and
[link policy](link_policy_v1/README.md) now define real source epoch boundaries
and selectors. The selected-state inference source is prepared for NCN, NCNC
and BUDDY, including source-specific role graphs. Native numerical, data/cache,
resource and final runtime qualification remain open. No link fit or TEST score
is created by these preparation checks.

## Additional concrete methods under preparation

| Candidate | Purpose | Current evidence |
| --- | --- | --- |
| Route-private BN running state, shared affine | Test whether using one running-state bank for different routes harms evaluation | [v3 source](private_bn_state_v3/README.md); seven allocation CPU checks and an actual GIN TRAIN-batch forward/gradient audit passed, zero optimizer steps. No predictive fit yet. |
| Structured private responsibility | Test endpoint-correlated training credit against a raw-marginal-matched query control | [v2 source](structured_responsibility_control_v2/README.md); eleven synthetic CPU checks, including the complete native NCN batch shape. Native integration remains pending. |

The BN candidate preserves every learned native parameter and adds only running
buffers. Its first two prepared versions and the real Torch2.1 API failure are
retained. The responsibility correction fixes a variance mismatch in the earlier
control; group normalization still changes its marginal law and is explicitly
part of the comparison. Neither ingredient is declared novel or successful.

Per-route selected-checkpoint analysis runs separately from training and changes
no selection. [Observation12 and its partial diagnostic](pilot_20261015_molbace_v1/PARTIAL_SEED101_DIAGNOSTIC.md)
record four completed seed101 fits. Graph alignment/variance has selected-VALID
AUC0.71062 versus base0.62784; the input correction has0.67729 and nativeM1
has0.73846. Both candidates have higher BCE than base. The combined fit and
remaining seeds/I4 bodies are unfinished; no full-cohort benefit, novelty or
confirmation claim follows. Complete curves and all costs remain preserved.

The proposed statistical policy also needs revision before the full launch:
ten trials with exact two-sided sign flips and Holm across226 contrasts cannot
reject any cell at alpha.05. The [audit and prospective options](statistics_proposal_v1/AMENDMENT_PROPOSAL.md)
preserve the full comparison matrix. A [specific global-primary option](statistics_proposal_v1/GLOBAL_PRIMARY_OPTION.md)
would test a different, explicitly dataset-weighted question; it is a proposal,
not an adopted rule or a significance claim.

## Evidence and decisions

The portable input-adapter package now has two explicit methods: BE plus private
nonlinear input corrections, and four private corrections around one shared
native body with BE factors removed. The latter remains a four-route ensemble.
Actual MolBACE/GIN and full-graph CiteSeer/Cheb CPU checks passed, including
updates, complete VALID and fresh-session checkpoint restoration. These are
execution checks, not trained accuracy results. See
[portable input readiness](qualification/portable_private_input_v2_READINESS.md).
The original v1 repository-binding defect and its source remain preserved.

The [conditional aggregation recipe](conditional_aggregation_v1/README.md)
prepares mean, global-weight and prediction-conditioned weighting under the
same half-own/half-served-pool loss. They add0/4/20 learned parameters. Actual
MolBACE/GIN CPU update/VALID/restore checks passed; other native ports, CUDA and
final serving are separate gates. Positive weights and entropy/disagreement
inputs do not guarantee better predictions or calibrated confidence.

[Structural class responsibilities](qualification/structural_class_responsibility_v1_READINESS.md)
prepare atom-count/degree quartiles within each TRAIN class and an exact
within-class permutation control. Every route retains positive supervision
and equal full-TRAIN class mass. Shared weights receive ordinary own risk;
private factors receive weighted risk. Both molecular variants passed actual
CPU update, whole-VALID and checkpoint checks. Node integration is being added;
the endpoint-correlated link field is a separately defined hypothesis.

A separate [three-fit BN follow-up](pilot_20261015_bn_state_v1/README.md) is
frozen and its supervisor is waiting for successful completion of the current24
fits. It reuses all three exact base-GNNM and independent-ensemble control
cohorts. It starts one new child at a time, without changing the parent queue.
A CPU-only analysis watcher will report every paired selected-VALID difference
after all three candidates finish. No BN quality result is available yet.

The [campaign supervisor](campaign_launcher_v1/README.md) has a concrete finite
source-job interface and eighteen passing synthetic scheduling checks. All2,480
[predecessor descriptors](job_binding_preparation_v1/REPORT.md) now pass metadata
validation, with actual pinned source/data paths and exact I4/M1 ownership.
Those checks neither execute a native epoch nor qualify the expanded roster.
Measured resource profiles, full-runtime coverage and prospective confirmation
remain open;18.77 is not currently authorized. Three-GPU readiness is not claimed.

- [Mechanisms and current scientific rationale](RESEARCH_SYNTHESIS_20261015.md)
- [First-seed prediction diagnostic and its limits](pilot_20261015_molbace_v1/FIRST_SEED_DIAGNOSTIC.md)
- [Task-method scientific audit](method_matrix_scientific_audit_v1/REPORT.md)

- [Exact frozen pilot](pilot_20261015_molbace_v1/SPEC.json)
- [Molecular objective and scoped readings](molecular/README.md)
- [Link-specific candidates and integration limits](link/README.md)
- [Literature-to-experiment decisions](literature/SCOPED_CONCLUSIONS.md)
- [Reviewed method catalogue and all historical dispositions](catalogue/CATALOGUE.md)
- [Current execution progress](PROGRESS_20261015.md)
- [Selected prior outcome index](prior_result_index.json), not an exhaustive
  experiment census or a new paper-reading count

The pilot's primary comparison is paired VALID ROC-AUC against base GNNM and
independent4, with BCE, competence, common errors and costs. Geometry is a
mechanism diagnostic. Three seeds on one split do not establish statistical
significance. The final benchmark needs paired uncertainty and predeclared
heldout confirmation, rather than judging success by separated standard
deviations or by the best observed seed. Original paper scores stay unchanged.

## Newly completed native bridges

- Node class alignment and alignment+variance passed actual fullTRAIN/VALID/restore
  CPU checks on CiteSeer/Cheb, with the fixed512-node auxiliary support recorded.
- Structural responsibilities and their exact classwise permutation control
  passed the same native node lifecycle; the earlier guard failure is preserved.
- Molecular alignment-only now has both a preserved one-class lifecycle check
  and a separate active mixed-class check with nonzero shared/private gradients.
- [NCN native extensions](link_expansion_ports_v1/REPORT.md) passed8actualCPU cases:
  mean/global/conditional aggregation, unit/query/endpoint responsibility, and
  BE+residual/residual-only. Native65536positive+65536negative updates and complete
 160084-query VALID were retained. NCNC/BUDDY and full-runtime work are separate.
- [Fixed context target source](context_affinity_v1/README.md) and exact task
  definitions prepare common versus split information targets; tensor semantics
  passed, while native task ports are being completed.

These are execution and mechanism checks, not additional trained accuracy
results. The latest reviewed source addition was pushed and exact-ref verified
as`eb6ddd1c58789a88261d81976c65323ce425b6a2`. Subsequent work is recorded under
a new source addition, preserving all prior releases and outcomes.

Observation13 records the completed seed101 combination: VALID AUC0.68352 and
BCE0.51682, versus alignment-alone AUC0.71062. The combination does not improve
on its strongest ingredient in this seed. Other seeds and I4 remain unfinished;
this result and its costs are retained without changing any frozen setting.
