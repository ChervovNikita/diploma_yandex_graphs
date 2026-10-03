# Native conditional-response v3: preserved source and qualification preparation

## Status

This is a **new successor** to sealed native-source v2
`graph_conditional_response_native_source_preparation_20261003_v2`
(manifest SHA256 `797525334ddc17c4d60052b76c53c2c587eaea541676cccac3997462d54d69d0`).
The earlier `graph_conditional_response_qualification_preparation_20261003_v1`,
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
were accessed. Five pinned PyG code files are inherited unchanged from v2;
no source retrieval or network call was made for v3.
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

Recipe, cache and split/runtime conditions still prevent native admission:

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
3. **Scoped native split interface.** Pinned PyG2.7 HeterophilousGraphDataset
   transposes raw masks to `[nodes,splits]`. The pinned author `training.py`
   sends Amazon and roman-empire/minesweeper/tolokers/questions to
   `heter_fixed_splits`, which uses `mask.permute(1,0)[idx_run]`, equivalent to
   `mask[:,idx_run]`. `hetergraph_fixed_split` uses row indexing only in the
   filtered chameleon/squirrel branch; that custom loader retains raw masks
   without a transpose. V2's broad claim of an incompatible Amazon author helper
   was incorrect. The compact TRAIN-only successor `[:,official_split_id]`
   interface agrees with the author Amazon branch and still needs exact
   layout/release/hash qualification. No actual masks were loaded. The proposed
   80/20 floor(4*n_class/5), fixed hash order and seed policy remain unadopted;
   the new unequal-axis mask fixture does not select those scientific rules.

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

## Fresh critic and actual native checks prepared

The independent v2 critique is preserved and hash-bound as
`graph_conditional_response_native_source_review_20261003_v1/REPORT.md`
(SHA256 `2942036847c6f5bd30066462778c7035d2407196f6478b035c2ead4db1754192`)
and its `REVIEW.json`. Six source findings and their precise dispositions are
recorded in `NATIVE_QUALIFICATION_PLAN.json`. The critique is not authorization.

Eight opt-in synthetic CPU engineering families are now prepared, **all
unexecuted**. The original three families retain their narrower coverage:

- One-member and all-identical M4 parity, complete map/count inventory and
  registered attention-bias custody.
- One-member isolation, private-only native gradients, dropout-off repeatability,
  exact state/RNG custody and native dtype-failure member-selection cleanup.
- Original native mono/token and two-dimensional raw-logit adjoint checks.

Five added families prepare the missing conditions:

- All-site unequal nonunit R/S, nonzero shared biases, independent unwrapped
  author member paths using `W_m=S_m[:,None]*W*R_m[None,:]` and `b_m=S_m*b`,
  output/input/common/factor gradient comparisons, coherent member permutation,
  R/S isolation at every site and Stage B private parameter counts.
- Unequal node/split axes, source-consistent column selection, explicit
  reject-before-mutation reset behavior, and non-one-factor/changed-bias
  state_dict replay with explicit Stage B modes, permissions and receipt custody.
- An irregular reciprocal graph with an isolated node, source loops and paired
  same/other removals; independent degree/loop/operator checks, rebuilding every
  K+1 view, CompleteViewBank identity mismatch/invalidation and mutable-input
  snapshot detection/restoration.
- Actual complete native three-view objective/backward and CE/energy guards on
  supported synthetic groups; initialized actual AdamW moments; full and forced
  fractional acceptance, exhausted rejection, full displacement zero, impure
  guard and raised native forward. Exact snapshot assertions cover common/private
  parameters, buffers, gradients, modes, permissions, optimizer Python/moments,
  RNG, native configuration, nondefault member selections and separately audited
  fixture inputs. Real guard assessments precede fixed engineering rejections;
  no guard threshold changes. All proposals, oracle steps, objective/backward,
  attempted/failed guards and transaction spans remain charged. The inherited
  ledger's member-path field counts declared M4 attempt units for each complete
  view callback; it does not assert four executed trajectories after an early
  native forward failure.
- Leading `[4,5,96,3]` adjoint calls versus looped and coherently permuted
  references, K=0 and K=2, a separate nonsymmetric helper-only transpose fixture,
  and direct-X versus adjoint higher-order equality at every private R/S.
  The nonsymmetric fixture admits no directed scientific graph; synthetic
  raw-logit scores do not choose or qualify FoRDE.

The dispatcher additionally requires root's explicit native CPU authorization
and a receipt matching the actual Torch/PyG/NumPy/SciPy versions, source bytes,
imported normalization/loop/dataset source pins, declared compiled dependency
files and deterministic/thread settings. No installed numerical runtime was
imported or qualified here. Even a later passing receipt would not establish
Torch2.0/PyG2.3 parity, a warm Amazon recipe, strong controls or study admission.
Numerical equality tolerances are predetermined engineering tolerances only;
they do not relax strict CE/energy guards. Synthetic factor/gradient seeds,
removal lists and supported role sizes are fixture choices, not scientific
initialization, mask, role-split or reference-policy adoptions.

Post-wrap native reset is explicitly unsupported. MemberAffine now rejects
native reset calls before changing common/factor state; reconstruct per seed,
load state_dict and restore explicit permissions/receipts. Stage B still inserts
or resets no factors. The only adapter behavior change from v2 is that explicit
rejection; the source verifier checks this exact diff.

Strong source-informed DICE/FoRDE controls remain mandatory under the unchanged
prior requirements. No estimator/control implementation, weak replacement or
successful control qualification is claimed. The functional-kernel algebraic
equivalence from the parent prior packet still holds on identical normalized
responses; no duplicate fit is added. Once source/native engineering and rule
gates pass, measure a complete-data resource step before resource requests.
The old 33 base member-fit equivalents and24 continuation banks remain later
comparison estimates, not resource measurements or launch permission.

## Source-only verification and immutable seal

The permitted verification covers 26 packet Python files, 48 external Python
files and ten deferred entry modules under a numerical import blocker. It binds
all 92 unique external inputs: the complete v2 seal/payload, its 58 inherited
bindings, and both fresh critic artifacts. The five copied core files, adopted
rule JSON, unresolved mask alternatives, group/mask code and token interface
remain byte-identical. Five native author classes and two preprocessing
functions still match pinned ASTs; six author files retain SHA256/Git-blob pins;
all five inherited PyG source pins remain exact. An additional AST check confirms
both dataset branches, split expressions and the filtered loader's untransposed
raw masks. Raw native/library source is parsed and hashed, never imported.

Use Python 3.10 or newer for source verification. The system `python3` here is
3.9 and fails on the inherited union annotations before numerical import; the
bundled interpreter is recorded in the fresh receipt. No inherited core bytes
were changed to accommodate that interpreter. The source receipt records exact
bindings and no numerical packages, model/tensor constructions, native checks,
data/labels/checkpoints/outcomes, role/mask/feasibility calls, fits or launches.
This is syntax/custody/source verification, not behavioral admission.

The binder preserves the entire predecessor and its ancestors/decisions without
writing them, carries every inherited binding, and binds the fresh critic. The
sealer refuses overwriting a seal and rejects stale source receipts; final
verification checks the sealed successor and every external input. Any later
repair must use another preserved successor. All unresolved scientific/recipe,
control, actual runtime/data and complete-data resource gates remain open.
