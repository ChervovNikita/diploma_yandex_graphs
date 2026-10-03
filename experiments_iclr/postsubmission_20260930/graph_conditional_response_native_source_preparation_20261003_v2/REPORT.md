# Native conditional-response successor: source preparation

## Status

This is a **new successor** to the sealed qualification preparation
`graph_conditional_response_qualification_preparation_20261003_v1`. That packet,
the immutable contrastive proposal, GENN access packet and literature indexes
are unchanged. The parent reports all ten old CPU engineering witness families
passed; this child neither executed them nor imports that result as native
PolyFormer qualification.

The parent explicitly resolved fit/control grouping and adopted the modified
AdamW transaction prospectively. Those rules are recorded in
`EXPLICIT_SUCCESSOR_RULES.json`; they do not rewrite the earlier proposal.
This successor prepares actual native source composition, graph/token/split
interfaces and scientific mask alternatives. **No model/module construction,
Torch/numerical import, tensor execution, dataset/label/checkpoint/outcome access,
fit, resource measurement, protocol freeze or pilot launch occurred here.**
Only source code, source metadata, stdlib syntax/import checks and custody hashes
were accessed. Five public pinned PyG code files were fetched, without import.
No primary paper was reread or new full-paper-read certification made.

## Architecture prerequisite correction

There is no verified pre-existing all-layer PolyFormer port in the inspected
sources. Existing modern ports provide boundary factors, full independent
members or cached-token MIMO. Their shared weights/factors do not establish the
proposed intermediate composition. Existing residual-model all-layer results
are not imported as evidence for it. “Existing intermediate private factors” in
the original proposal was a **prerequisite for the warm bank**, not a verified
implementation availability claim.

`native_source/adapter.py` now prepares a new attributed composition before any
warm bank. It preserves the pinned author class forwards and inserts R/S factors
around **every native affine map**, including stem/readout/head factors that will
be frozen in continuation. Every member retains its complete native token,
attention, LayerNorm, residual, FFN and head trajectory. Only final member logits
are stacked. This explicit sequential source reference makes no packing or speed
claim and has no implicit backend fallback.

The declared map algebra is `shared_affine(x * R_m) * S_m`, with all-one factors
for native parity and **no added private bias**. This is the local
`MemberFactorLinear` composition with its additive B fixed to zero. It openly
scales the shared affine bias by S. Edward2's inspected rank-one dense layer uses
a separate member bias after scaling the shared bias-free kernel; that is a
different bias convention. This is attributed implementation work, not an exact
TensorFlow port, new primitive, or novelty claim. A composition receipt is
required; recipe/competence admission is still open.

## Actual native source inventory

Author pin: `air029/PolyFormer@d390f39e88d0eaac80318fdc7704bd3bf3cf8b13`.
All retained class bodies are checked against the saved pinned author sources
with stdlib AST comparison. Only the outer module's import glue is changed to
a relative import; no native class forward is rewritten.

| Native maps/state | Warm composition | Stage B |
| --- | --- | --- |
| `lin1` F→H stem | Existing common map plus R/S | Freeze common map and factors |
| `lin2` H→H readout | Existing common map plus R/S | Freeze common map and factors |
| `lin3` H→C class head | Existing common map plus R/S | Freeze common map and factors |
| Every order's attention token MLP, two affine maps | R/S around each map, every layer | Only actual warm R/S train |
| Every layer's `W_Q`, `W_K` | R/S around each bias-free H→H map | Only actual warm R/S train |
| Every block FFN's `lin1`, `lin2` | R/S around H→d_ffn→H | Only actual warm R/S train |
| Attention/FFN LayerNorm | Common native parameters | Frozen |
| Learned head/order `bias_scale` | Common native parameter | Frozen |
| Fixed order-decay `bias` | Same native values registered as buffer before warm | Frozen, snapshotted |

The plain native attention `bias` tensor is absent from state_dict. Registering
its unchanged values before any warm acquisition makes device/dtype/state
custody explicit. Member selection and primitive native configuration are
captured by a named source hook. Every selected factor is restored in `finally`,
including native forward exceptions. All common/factor parameters and buffers,
gradients, modes, optimizer state and RNG use the unchanged copied core's
transaction. Structural mutation, reentrant/concurrent forwarding, unqualified
compile/hooks/monkeypatches and full-model-object pickle are unsupported. Rebuild
from the pinned source/config and use state_dict checkpoints.

The primitive hook also covers native GELU `approximate` and ReLU `inplace`
settings. Complete member factor storage is present before warm acquisition;
there is no post-warm factor insertion or boundary-factor reset in Stage B.

`site_inventory` requires exact map names and dimensions, so an added/removed
native map cannot silently escape factorization. For token MLP width
`t=int(H*multi)`, there are `3 + L*(2*(K+1)+4)` affine sites. Complete parameter
algebra includes every native bias, LayerNorm and attention scale, plus
`M*sum_sites(in+out)` R/S parameters. Actual constructed-module counts remain
unexecuted. `begin_private_continuation` adds no parameters; it freezes all
weights and boundary factors, enables only warm intermediate R/S, clears
inherited frozen gradients and sets all modules to eval. Objective forwards keep
autograd through the frozen maps; guards enter no-grad separately.

## Native recipe, normalization and split risks

Three concrete source risks prevent claiming ready native qualification:

1. **Amazon recipe availability.** The author loader recognizes `amazon-ratings`,
   but the pinned training CLI choices exclude it, and the pinned shell recipes
   contain no Amazon command. The old modern port's Squirrel constants cannot
   be inherited. `NativeSpec` has no recipe defaults and requires a receipt.
   Root must explicitly choose a source-informed adapted competent Amazon recipe
   before warm acquisition; author-native Amazon reproduction is not established.
2. **Unsafe native cache identity.** Author `load_base` keys its pickle by only
   dataset/base/K. Native and two removal views would collide. The successor
   bypasses that loader and exposes a complete identity with release, original
   features, native edges/attributes, removed units, fit role/labels, view, order,
   normalization source and runtime/precision fingerprints. It writes no cache
   or data itself.
3. **Split layout mismatch.** Pinned PyG2.7 HeterophilousGraphDataset transposes
   raw masks to `[nodes,splits]`. The retained author helper selects
   `train_mask[idx_run]`, incompatible with that layout. `train_roles.py`
   explicitly selects `train_mask[:,official_split_id]` after shape/release/hash
   qualification. No actual masks were loaded. The proposed precise stratified
   80/20 rule is floor(4*n_class/5), fixed hash order per class, remaining targets
   control, no forced minimum-one or seed retry; that split detail still requires
   prospective adoption.

The inspected mono path calls `gcn_norm` with native defaults, converts its COO
records/weights to SciPy, then forces the operator to float32. It constructs
`X,PX,...,P^K X` for every complete view. The successor retains that arithmetic
and rejects non-float32 X rather than silently changing it. PyG2.7 code is pinned
for `gcn_norm`, `to_scipy_sparse_matrix`, `add_remaining_self_loops`,
`to_undirected` and HeterophilousGraphDataset. Actual author environment was
Torch2.0/PyG2.3; source/runtime equivalence to the admitted runtime is still a
qualification gate.

For edge_index COO, pinned `gcn_norm` defaults to `improved=False`,
`add_self_loops=True`, `flow=source_to_target`; it uses destination degree and
adds/replaces exactly one loop per node when edge weights are absent. The author
loader/PyG dataset already makes the graph undirected. Masking therefore must
remove both reciprocal nonloop records, retain source loops, and rebuild the
normalization. Existing native weights/duplicates cannot be silently collapsed
or discarded. Prepared mask code stops on weighted, duplicate or asymmetric
nonloop records pending explicit qualification; it does not pretend those cases
were observed absent in Amazon.

Cached-token derivatives are not original-feature derivatives. The prepared
mono adjoint uses reverse recurrence to recover
`grad_X=sum_k (P^k)^T grad_Tk`, retaining all leading member/target axes and
higher-order autograd. SciPy normalization cannot propagate edge-weight
derivatives; those are explicitly unsupported. Feature gradients are valid for
this inspected fixed-operator construction, but the helper is **not** a FoRDE
score/kernel/recipe implementation. Full-data derivative memory/work remains
unmeasured and must be charged.

## Adopted grouping and optimizer rules

Fit groups are a disjoint partition: keep every supported raw fine/no-neighbor
group (32 targets); merge only rare raw groups within class; send unsupported
class rare-only unions to a remaining-rare-only global fallback. An unsupported
terminal global fallback stops, including when other supported groups exist.
Never overlap supported fit cells, drop targets or relax minima.

Control guards include complete global, all supported complete-class and all
supported raw fine/no-neighbor cells (16 targets), deduplicate identical
memberships and record aliases/coverage. Unsupported raw/class targets remain
globally guarded. A global role below16 stops. Objective mean-member native CE
retains all declared fit targets. TRAIN-control guards remain supervision.

The copied core uses the explicitly adopted **modified AdamW step**: propose
ordinary AdamW once; scale the complete displacement including weight decay;
accept the first feasible nonzero scale and commit the full proposed
moments/step/Python optimizer state once. Bitwise-zero or rejection restores all
state, with no extra zero guard. Costs remain outside rollback. No unchanged
AdamW or gradient-scaling claim is made. The copied core bytes are checked
against the old seal, so the parent's old generic CPU qualification is not
quietly altered here.

## Degree-matched label-permutation decision

The preferred original-strength null is **Alternative A**: one fixed
class-count-preserving fit-label permutation; preserve original per-view total
removed-unit counts and **every node's per-view removed degree**, while each
selected edge satisfies that once-permuted same/different category. Prepared
code builds the exact constraint instance, checks necessary local capacity and
independently verifies feasibility certificates. A source-qualified exact
undirected b-matching/MILP oracle and deterministic selected-solution policy are
still missing. The declared solution selector is lexicographic under a fixed
hash-priority edge order; it is not claimed uniform over feasible masks.
Infeasibility/timeout/unqualified solver stops admission, with no new permutation
or relaxed-degree repair. No actual feasibility was assessed.

**Alternative B** is a distinct exact-distribution null: match counts of removed
edges in each `(view,min full native degree,max full native degree)` stratum,
using exact degrees including the one native normalization loop. One fixed
permutation and hash-priority subset per stratum; insufficient capacity stops.
This preserves exact endpoint native-degree distributions and counts, **not
each node's removed degree**. Its complete stdlib sampler is prepared. It requires
an explicit prospective root decision and narrower falsifier interpretation; it
cannot silently substitute for A. No coarse bins, tolerances or weak stand-ins
are proposed. Keep original supervision/group labels, not auxiliary permuted
labels. Record identity/noninformative permutations without retries; their
admission disposition also needs a predeclared decision.

Cross-arm warm reference semantics still need an explicit scientific decision.
The recommendation freezes common conditional active IDs on original candidate
warm views and uses each arm's copied warm function on its own declared views,
requiring every common active member/group to meet the floor. An incompatible
permuted arm fails admission; no dynamic intersection. Class-only D has a
different partition, so normalization/energy matching cannot simply reuse
conditional group IDs. A possible matched policy is a common predeclared union
of conditional and class-only energy guards for every arm; it needs root
adoption and is not implemented as a hidden extra constraint. See the eleven
specific dispositions in `RULE_RECOMMENDATIONS.json`.

## Actual native checks prepared

Three opt-in CPU source-composition families are prepared in
`prepared_native_checks.py`, with no execution here:

- One-member and all-identical M4 native output parity, exact source map/count
  inventory and registered attention-bias custody.
- Complete member isolation, only intended actual intermediate R/S gradients
  through frozen native maps, dropout-off repeatability, exact state/RNG custody
  and member restoration after a native dtype failure.
- Exact native mono normalization/token equality, isolated nodes/self-loop
  handling, original-feature versus cached-token adjoint equality and
  higher-order flow into actual native private factors.

The numerical equality tolerances are predetermined engineering tolerances only;
they do not relax CE/energy guards. Old generic witness authorization does not
authorize these new checks. Native construction/forward qualification, exact
normalization/runtime replay and a competent Amazon recipe remain pending.

Strong source-informed DICE/FoRDE controls remain mandatory under the unchanged
prior requirements. No estimator/control implementation, weak replacement or
successful control qualification is claimed. The functional-kernel algebraic
equivalence from the parent prior packet still holds on identical normalized
responses; no duplicate fit is added. Once source/native engineering and rule
gates pass, measure a complete-data resource step before resource requests.
The old 33 base member-fit equivalents and24 continuation banks remain later
comparison estimates, not resource measurements or launch permission.

## Source-only verification and immutable seal

`VERIFY_PACKET.py` passed the permitted source-only boundary: 22 packet Python
files and 26 external Python files AST-parsed; six entry modules imported as
inventories under a numerical-package import blocker; all 58 unique external
bindings verified. The five copied core files are byte-identical to the sealed
v1 payload. Five native author classes and two preprocessing functions match
the pinned author ASTs. Six author files match retained SHA256 and Git-blob
pins, and all five newly retrieved PyG code files match their fetch receipts.
Raw native and library files were parsed and hashed, never imported.

Actual counts remain zero for numerical package imports, native check execution,
synthetic tensor/model construction, data calls, role construction, mask sampling
and feasibility calls. This is source/custody verification, not behavioral,
runtime, recipe, control, resource or efficacy qualification. The source review
also made fixture devices explicitly CPU and completed native activation
primitive custody; neither change was numerically exercised.

The binder carries the old preparation's complete sealed manifest, payload and
all nineteen inherited bindings, including original contrastive, GENN,
literature indexv31 and parent functional-kernel attribution custody. It hashes
those protected files without altering them or reading new paper/outcome text.
The sealer checks every verified payload byte against the recorded source
receipt, rejects stale receipts and refuses to overwrite an existing seal.
Final verification checks both the successor manifest and all external inputs.
