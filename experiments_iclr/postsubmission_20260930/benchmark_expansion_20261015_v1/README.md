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

The current full-epoch frontend supports node and molecular tasks. Link
objective definitions, query-order capture and gradient helpers are implemented
in [link](link/README.md); native full-epoch link integration and numerical
qualification remain pending. Standalone synthetic tests do not count as native
model results. Per-route selected-checkpoint analysis is being added separately
so it does not change the running fits or checkpoint selection.

## Evidence and decisions

- [Exact frozen pilot](pilot_20261015_molbace_v1/SPEC.json)
- [Molecular objective and scoped readings](molecular/README.md)
- [Link-specific candidates and integration limits](link/README.md)
- [Literature-to-experiment decisions](literature/SCOPED_CONCLUSIONS.md)
- [Selected prior outcome index](prior_result_index.json), not an exhaustive
  experiment census or a new paper-reading count

The pilot's primary comparison is paired VALID ROC-AUC against base GNNM and
independent4, with BCE, competence, common errors and costs. Geometry is a
mechanism diagnostic. Three seeds on one split do not establish statistical
significance. The final benchmark needs paired uncertainty and predeclared
heldout confirmation, rather than judging success by separated standard
deviations or by the best observed seed. Original paper scores stay unchanged.
