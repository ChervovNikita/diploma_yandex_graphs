# WikiCS unit-factor attribution, VJP recomputation (disabled v2)

Preserve v1 and its waiting qualifier. The present packet changes execution to
fit a shared host; it requests no host change or action against other jobs.
No new model/data/epoch framework is introduced, and no numerical work ran in
source preparation. Actual fullgraph v2 qualification is required before fits.

## Same fixed12 scientific design

Question: does Wiki24's exploratory +0.506pp unit-factor gain reflect supervised
alignment (A), residual route contrast (R), or their joint effect (C)? Fresh
plain P(0,0), A(.05,0), R(0,.05), C(.05,.05) fits use seeds6101,6203,6307.
Seeds6101 then6307 use GPU0/a998; seed6203 uses GPU1/8ced. P,A,R,C run in order
within each seed; at most one owned fit per GPU. All12 fresh fits use v2 on77,
including P/C references. Historical scores cannot replace those references.
This is exploratory attribution using Wiki24 development seeds again, with no
grid, novelty/confirmation claim, extra seeds, tuning, early stop or retry.

The sealed public v2 library/core/native sources are unchanged. The same
per-session objectives facade calls active original functions unchanged and
zeros inactive A/R terms. Shared four-member unit factors, architecture, Adam,
all580 TRAIN labels, original two own-loss views, maximum512 auxiliary targets,
.05 coefficients and .2 temperature remain fixed. Unchanged public train.main
runs1100 epochs:100 local then1000 global, original joint local transition,
strict-first complete-development selection, no early stop. Development is the
5274-object union of official split0 validation/stopping masks; no TEST labels.
Source/data/equality bindings are in SOURCE_BINDINGS.json.

## Chain rule and stochastic trajectory

For fixed parameters theta and member/view RNG draws r, let
Y = {f_(v,m)(theta;r_(v,m))} contain logits and representations for two views and
four members. The complete original joint objective is L(Y). Its gradient is

    dL/dtheta = sum_(v,m) [Df_(v,m)(theta)]^T * (dL/dY_(v,m)).

First call the original Session.forward twice under no_grad, preserving member
streams before and after these8 shadow forwards. Detached output leaves retain
only [4,580,10] logits and [4,580,512] representations per view. Evaluate the
exact original own/active auxiliary functions, scaling and object selection on
those leaves once. One autograd.grad computes all output cotangents, including
cross-member/class-centering dependence. Unused representation cotangents in
plain are zero; no loss/gradient scaling is added during replay.

Restore pre-shadow streams. Replay view0 members0..3 then view1 members0..3
using model.member_forward and the original fork_rng/member-stream protocol.
Each graph immediately receives one backward with both logits and representation
cotangents. Autograd sums their shared paths. No parameter update, gradient
clear or Adam step occurs between these8 VJPs. After the full accumulated finite
gradient, perform one original Adam bank step and the original finite-state
check. Replay member RNG endpoints must equal shadow endpoints exactly, so only
the original two stochastic views advance the persistent streams. The pinned
WikiCS LayerNorm/GAT model has no BatchNorm/mutable buffers; installation checks
this restriction. Saved/restored streams and selectors keep their public APIs.

The mathematical chain-rule objective is preserved. Floating-point accumulation
order and kernels can differ; there is no bitwise author parity or numerical
equivalence claim. Qualifier prediction differences are diagnostic and have no
small-rounding rejection threshold. Exact RNG endpoints, finite actual gradients,
Adam/update/reload and full population work are required engineering checks.

## Charged resources and admission

Each update costs16 training member forwards (8 shadow +8 replay),8 parameter
reverse collections,1 output-cotangent collection and1 Adam bank update.
The fixed12 fits cost13,200 updates,211,200 training member forwards and105,600
member VJPs:105,600 additional forwards over v1. Extra compute is charged in
config, progress, completion and snapshots. Complete checks all counters.
Serving uses the original public forward/pool and is additional evaluation work.

Two-view retained outputs occupy9,688,320bytes; cotangents add the same (~18.5MiB
combined), plus bounded auxiliary-loss scratch. One backbone graph remains live
at a time. ~20GiB is a planning estimate for that graph, not a measured v2 peak.
Proposed owned GPU cap is32GiB (release range24–32GiB), fresh free admission36GiB,
with root-bound RSS and wall limits. Other shared-host jobs are preserved.
The external reviewed owner must enforce caps and own cleanup; this packet
contains no scheduler. Full cells retain32390s active +10s cleanup =32400s hard,
soft28800s, unless a separately reviewed source changes those bounds. Aggregate
108h hard safety sum is not an ETA and extra replay does not extend the horizon.

qualify.py runs all four conditions, local and global, with complete fullgraph
TRAIN/580 labels and finite5274-development serving. Its8 discarded updates
charge128 training forwards and64 member VJPs. It checks exact replay/shadow RNG
endpoints, source counters, finite actual backward/Adam, original CPU-mapped
save/reload, CPU byte RNG, CUDA parameters/Adam moments, and restored modes.
It records nonblocking replay prediction differences and actual GPU peaks.
Actual reserved peak must fit the released cap. Root must bind that complete
v2 qualifier receipt and actual resource evidence before releasing any fit.
Use the separate disabled release templates and command argv, existing venv,
empty PYTHONPATH and reviewed owned supervisor; never enable/edit this seal.

## Readout

First report paired fresh C-P on77, then A-P, R-P, C-A, C-R and interaction
C-A-R+P, by seed and mean/SD in pp. df2 exploratory intervals are descriptive:
three optimizer seeds on one graph are not independent graphs. If fresh C-P
does not reproduce the reference direction, report that limitation before
attribution; no rescue fits or selection. At fixed selected checkpoints report
member accuracy, member/pooled NLL/Brier and paired error flows against the same
seed's plain checkpoint. Route repulsion alone does not establish competence,
useful diversity or retained class information. Report development accuracy,
not published TEST accuracy; author-execution equivalence remains unqualified.
