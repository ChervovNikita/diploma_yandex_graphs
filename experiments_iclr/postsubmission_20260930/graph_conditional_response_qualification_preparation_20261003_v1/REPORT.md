# Conditional graph response: implementation qualification preparation

## Status and permission boundary

This packet prepares source and numerical qualification of the single extension
in `graph_contrastive_private_paths_quality_gap_20261003_v1`. The sealed proposal
is unchanged. The parent adopted source/numerical preparation only. This packet
does not freeze or launch a pilot, establish efficacy, certify novelty, request
resources, or change a canonical research ledger/status.

The core and ten numerical witness families are written. Only source syntax,
stdlib import structure and file custody are checked here. **No Torch import,
synthetic tensor execution, data/label/tensor access, training, numerical witness,
threshold tuning, or complete-data resource step has been performed.** Native
PolyFormer/BE coupling and DICE/FoRDE implementations remain unqualified. The
GENN access packet, literature indexes and immutable contrastive packet remain
unchanged. No new primary source scope was read by this packet.

`SOURCE_VERIFICATION.json` records exactly the permitted checks. Any future
numerical result must be a separate dated receipt; it cannot be inferred from
the existence of prepared code. The witness runner requires a separate explicit
authorization file before it can import Torch or execute any witness.

## Implemented preparation

| File | Prepared responsibility |
| --- | --- |
| `core/groups.py` | Fit/control TRAIN labels only, raw class/neighborhood/no-neighbor keys, fixed explicit cell membership and resolution gates |
| `core/response.py` | Per-member class probabilities, native-anchored two-probe response, within-group centering, exact normalized redundancy, frozen warm active set and inclusive energy bands |
| `core/guards.py` | Every member/view/global/supported control-cell CE reference and check; one three-view bundle reused for all cells and fit energy; work ledger outside rollback |
| `core/transaction.py` | Private-factor partition, full model/optimizer custody, one native AdamW proposal, complete-displacement backtracking, accepted full-moment commitment or full rejection/zero rollback |
| `prepared_witnesses.py` | Ten future opt-in numerical witness families; no automatic data, training, control fit or launch |
| `SOURCE_CONTROL_REQUIREMENTS.json` | DICE/FoRDE, functional-kernel equivalence, original-feature derivatives, permuted masks and modern backbone admission requirements |
| `UNRESOLVED_RULES.json` | Eleven explicit native rule/source/runtime gates; fixture choices do not resolve them |
| `RESOURCE_STEP_PLAN.json` | Later complete-data measurement plan; no measurements or resource request |

The core imports with stdlib alone. Torch imports occur inside numerical entry
points. Importing the core or witness inventory performs no numerical work.
The model and complete-node three-view forward adapters are deliberately absent:
the native source/site/cache rules are not qualified, and this packet must not
silently invent a PolyFormer port or graph-mask sampler.

## Response and fixed grouping

For each of four complete member predictors, the differentiable object is

\[
A_m(v)=[p_m^0(v)-p_m^+(v),\;p_m^0(v)-p_m^-(v)].
\]

Each `p` is the member's class softmax, before serving pooling. Probabilities
identify the class coordinate gauge: function-preserving hidden basis changes,
classifier-nullspace changes and additive class-common logit shifts preserve
the measured response, up to numerical precision. Positive response scaling
preserves normalized redundancy but is separately constrained by energy. The
code does not detach frozen common/stem/classifier maps from an objective path.
Frozen weights still transmit derivatives to intended private factors.

`raw_group_keys` uses each TRAIN target's class and whether at least half its
fit-labeled neighbors have that class. It returns a distinct no-fit-neighbor
key. Neighbors in control/forbidden roles contribute no labels. Its adjacency
must already have explicit source-qualified direction/duplicate/self-loop
semantics; this function does not choose those graph conventions.

The prose fallback rule is insufficient to identify a unique objective. For
example, a class fallback containing **all** class targets overlaps supported
fine cells, whereas a fallback containing only rare fine-cell targets gives a
disjoint partition. These choices change centering and equal-cell weighting.
`FixedPlan` therefore requires explicit fixed memberships, support checks,
overlap weighting, no-neighbor/terminal-global decisions and control duplicate
handling. No target is removed from native supervision. Native construction of
these memberships remains an explicit resolution and source gate.

For a fixed fit group, center responses across nodes and flatten:

\[
R_{m,s}=\operatorname{vec}(A_m-\operatorname{mean}_{v\in s}A_m),\qquad
U_{m,s}=R_{m,s}/\|R_{m,s}\|,
\]

\[
D=\operatorname{mean}_s\operatorname{mean}_{m<n}
\langle U_{m,s},U_{n,s}\rangle^2.
\]

Both group and unordered-pair means are equal-weight means. There is no epsilon
normalization. At the copied warm function, a group is excluded if **any** member
norm is below `1e-5`; norms equal to the floor remain eligible. The active group
IDs and membership fingerprint are frozen. Coverage is the unique union of
active fit targets, so overlapping cells cannot inflate it. Coverage below 50%
fails the mechanism screen. Current zero/nonfinite responses fail, rather than
changing the active set. Every active member/group must remain inclusively
within `[0.5, 2.0]` times its copied warm norm. No tolerance is added to these
guards. Shared group mean responses are not repelled.

The prepared candidate objective is native **mean member CE**, plus the mean of
two probe mean member CEs, plus `0.1 D`. It does not replace mean member CE with
CE of a mean-logit predictor. Native serving remains the proposal's mean raw
logits then softmax. TRAIN-control guards are supervision, not heldout evidence
or a validation/test competence guarantee.

## Functional-kernel attribution and comparator

The parent/literature agent identified repulsive function ensembles,
`arxiv:2106.11642v3`, as exact operator ancestry. On these same fixed measurements,
the homogeneous degree-two polynomial functional kernel

\[
k(U_m,U_n)=\langle U_m,U_n\rangle^2
\]

has equal-group/unordered-pair repulsion exactly equal to `D`. The prepared
kernel equality witness exercises this identity. A comparator using that kernel
on the identical response, grouping, update permissions, warm references and
guards duplicates the candidate operator. **It does not justify another fit.**
The remaining hypothesis concerns conditional graph evidence measurement and
private-only feasibility policy. This packet makes no new repulsion-operator
claim. General ancestry does not establish complete method duplication or
efficacy, and GENN overlap remains unresolved. Any genuinely different
source-informed functional-kernel recipe requires an explicit nonredundant
comparator/protocol decision before later execution; no arm is added here.

## Optimizer interpretation and rollback

“Corresponding moments” in the sealed proposal does not specify a unique scaled
AdamW algorithm. The prepared transaction explicitly requires adoption of
`scaled_displacement_full_adamw_moments_once` and `restore_only_no_guard`.
These are qualification interpretations, **not amendments silently applied to
the immutable proposal**.

Given the original gradients and optimizer state, call native Torch AdamW once,
capture its full proposal and try

\[
\theta_\eta=\theta_0+\eta(\theta_{\rm AdamW}-\theta_0),\quad
\eta\in\{1,\tfrac12,\tfrac14,\tfrac18\}.
\]

The scale multiplies the complete displacement, including decoupled weight
decay. At scale one the proposal is copied exactly. A nonzero acceptance retains
those scaled parameters and commits the **full unscaled proposed** optimizer
moment/step/Python state once. For ordinary AdamW's scalar notation,

\[
\theta_\eta=(1-\eta\,lr\,wd)\theta_0
-\eta\,lr\frac{\hat m}{\sqrt{\hat v}+\epsilon}.
\]

Scaling gradients would change moments and epsilon behavior, and is a different
operation. Scales below one therefore do not constitute an unchanged native
AdamW update. Rejection of all four nonzero proposals gives zero: restore the
original model/optimizer state, and make no extra zero guard in this explicitly
gated interpretation. It does not certify that an externally supplied infeasible
base is feasible.

This prepared zero policy also discards changed moments if the full proposal or
a scaled trial has a bitwise-zero parameter displacement, including rounding to
zero. A witness with zero learning rate and nonempty moments is prepared. That
case is included in the unresolved zero-policy question; no native interpretation
is silently selected from the sealed prose.

Snapshots include all model parameters, state_dict/extra state, persistent and
nonpersistent buffers, gradients, requires-grad flags, every module's mode,
the optimizer's complete Python state with original parameter identities,
CPU and already-initialized CUDA RNG state, and named audited external state
hooks. Rejected/zero/exception proposals restore those snapshots. Guard purity
is checked exactly, including RNG and external hooks. Native source audit must
identify mutable model/compile/cache state outside state_dict; absence of a
hook is not proof such state does not exist.

The prepared support is deliberately narrow: exact Torch AdamW, strided state,
unaliased private factors with nonmissing gradients, no unqualified optimizer
hooks, fixed parameter/module registration and all modules already in eval
mode. Guards may not create/remove/resize parameters, modules or buffers. Such
structural mutations are outside this transaction's qualified support. Extra
control optimizers/estimators need their own explicitly source-qualified full
custody. A generic source receipt is an admission seam, not a native qualification
certificate.

## Honest work accounting

The ledger is outside snapshots. It records complete member-view forward
attempts before the callback, retains failures, and records proposal/transaction
spans including snapshots, backtracking and rollback. Callback timings alone
are not synchronized GPU measurements. Timing events may nest and must not be
summed as independent elapsed times.

A complete evaluation bundle has three actual complete-node M4 callbacks,
equivalent to twelve member-view paths. Cell reductions and fit energy reuse
that bundle. A rejected full step followed by accepted half step consumes six
callbacks/twenty-four paths. Four failed nonzero scales consume twelve
callbacks/forty-eight paths. The restore-only zero interpretation adds no
forward. No cost is erased when parameters or moments are restored. Objective
forwards/backward, preprocessing, token builds, higher-order derivatives,
estimator work and hidden adapter retries require their own charged events.

No resource timings are claimed here. Before resource requests, qualify the
mechanism/source and measure a complete-data resource step using the plan.
The **33 base member-fit equivalents and 24 continuation banks** remain later
comparison estimates. They provide neither a measured budget nor launch
permission.

## Prepared witnesses and remaining work

The ten families cover label boundaries/resolution failure; common logit
shifts/member permutation; nonorthogonal compensated hidden reparameterizations
and classifier nullspace; conditional versus class-only grouping and exact
functional-kernel equality; warm exclusion/coverage and current zero/large
energy; supported and rare-group class-fallback competence; private autograd
through frozen maps; accepted scaled moments/state replay and full rejection;
impure guard/failed-forward rollback and costs; and original-feature cached-token
adjoints with higher-order gradient flow. Fixture trial rejections are expressly
staged structural tests, not naturally observed acceptance rates. Fixture models
are not competent modern backbones or efficacy measurements.

The original-feature witness checks the fixed-operator identity
`grad_X = sum_k (P^k)^T grad_T_k`. It uses an explicitly synthetic raw-logit
true-label score to test the adjoint. It does **not** qualify FoRDE's source score
or recipe. Repelling cached-token gradients without that pullback changes the
coordinate domain. Source qualification must use actual normalization/filter
adjoints and retain all member/target axes and higher-order work.

All numerical witnesses remain unexecuted. Eleven native resolution/source gates
remain open in `UNRESOLVED_RULES.json`. DICE/FoRDE source recipes, label-permuted
degree-matched masks, common active/reference semantics and complete modern
backbone coupling must be qualified before admission. Then measure the complete
resource step. No primary-read, pilot, efficacy or novelty conclusion follows
from this preparation.
