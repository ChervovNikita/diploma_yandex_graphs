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

The frozen 13 October proposal remains intact: 7 datasets, 8 architectures,
27 fixed paper/source presets, 270 separate source-horizon reference fits and
2,210 matched fits (2,480 total). This is the predecessor design, not the final
expanded count. New candidates and task ports are being specified. The old
80,850-fit launcher remains retired. The expanded giant campaign has not been
approved or launched.

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
scientific retries. A later resume-finalization guard is still required for
failure after a stopping condition but before final export; v1 is preserved.

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
no selection. Native and base-GNNM exports have completed. The latest saved
observation has two completed fits, no failures and the graph-objective fit
training. There is no complete paired cohort or candidate advantage yet.

The proposed statistical policy also needs revision before the full launch:
ten trials with exact two-sided sign flips and Holm across226 contrasts cannot
reject any cell at alpha.05. The [audit and prospective options](statistics_proposal_v1/AMENDMENT_PROPOSAL.md)
preserve the full comparison matrix. A [specific global-primary option](statistics_proposal_v1/GLOBAL_PRIMARY_OPTION.md)
would test a different, explicitly dataset-weighted question; it is a proposal,
not an adopted rule or a significance claim.

## Evidence and decisions

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
