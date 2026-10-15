# Fixed structural responsibilities within TRAIN classes

This candidate tests whether routes benefit from persistent structural tasks.
For molecules, routes emphasize different atom-count ranges within each
observed TRAIN class. The shared native backbone still receives the original
unweighted task gradient. Private BE factors receive the weighted task gradient.
All original supervision remains positive; the architecture, native loss
population, Adam options and mean-logit prediction remain unchanged.

The method does not use current losses to choose a winning route. It constructs
one table from the full observed TRAIN set and preserves that table on resume.
No validation or TEST label/count defines an assignment.

## Exact assignment and control

Within each class, sort objects by the chosen structural count and divide the
rank interval into four equal parts. A tied group crossing a boundary shares
its overlap fraction equally among all objects in that group. This avoids
arbitrary ID-based tie breaking and also handles class sizes not divisible by4.

Each object's quartile memberships sum to1. Each route's membership mass within
a class is exactly one quarter of that class's size. Weights are `0.5 + 2*q`,
so they range from0.5 to2.5, every route has the original total class weight,
and each object's mean weight across routes is1. Exact rational arithmetic
checks these properties before conversion to fixed float64 native loss weights.
An all-tied class gives uniform weights, with no redraw or invented distinction.

The permutation control moves each object's complete four-route weight vector
to another object within its class using one isolated, fixed permutation. It
preserves the actual vector multiset, route/class totals and each object's
total supervision. It randomizes the association with structural count; it
does not guarantee zero empirical correlation on every finite draw. The seed
is the fit seed plus7030031. There is no permutation search or redraw.

## Structural counts

For molecules, atom count is the number of rows in the actual source TRAIN
graph's full nine-field `x`, checked against its native `num_nodes`. All observed
TRAIN graphs participate; missing targets are excluded exactly as in native
BCE. Unobserved targets have an unused weight placeholder1. No graph, atom,
bond, categorical ID or input feature is modified.

The node rule is incoming degree in the actual declared native graph after its
source preprocessing, grouped by TRAIN class. Count directed edge records at
their target IDs, preserving duplicates and existing self loops. Add no loops,
coalescing, symmetrization or propagation-normalization edges in this helper.
Full public transductive topology can contribute to TRAIN degrees, while only
TRAIN labels and objects receive weights. The exact pure helper exists; the
native node training bridge is still pending and is rejected explicitly.

This is distinct from the link endpoint-correlated random-field candidate.
It tests semantic structural specialization rather than random endpoint
correlation. It adds no capacity/equipped baseline or internal-site search.

## Original native reduction and gradient credit

Let F be the mean of four original native per-route losses. W multiplies each
observed record's native loss by its fixed route weight, then applies the same
within-route reduction and mean across routes. The molecular reduction remains
the native mixed-batch observed-label BCE mean. Full-TRAIN class mass does not
license class-balanced BCE or per-minibatch renormalization.

Shared gradients come from F and private gradients from W, both evaluated at
the same pre-update parameters. There are two reverse collections and one
original Adam transition. This block update need not be the gradient of one
global scalar objective. Future shared gradients can change as private state
changes; shared F credit does not imply an unchanged shared training trajectory.

The bridge reuses the frozen source's private-responsibility dispatch branch.
Its historical dispatch label remains `boundary_private_fixed_responsibility`,
but no exponential table is drawn. Dedicated metadata identifies the new
structural rule/control. Logs retain unweighted F, private W and their semantics.
Native source parameters, original optimizer ownership/options and shared BN
behavior remain intact. The bridge rejects already extended wrappers.

## Evidence and readiness

`test_table.py` checks exact fractional masses, ties, whole-vector permutation,
source-order invariance, incoming-degree semantics and TRAIN-role rejection.
`test_gradient.py` checks the original block-gradient helper on discriminating
finite tensors: correct F-shared/W-private gradients from old state and exactly
one Adam step. Those tensor checks are not GNN experiments.

`qualify_native_cpu.py` prepares a separate bounded actual MolBACE check for
both variants: two transitions using the same32 real TRAIN graphs, complete
VALID, fresh-session checkpoint restore and a resumed second update. Source
freezing precedes execution; results and observed failures/work are preserved
outside the immutable package. No scientific fit or TEST scoring is inferred.

The molecular bridge requires a pinned `extension.py`, the final package
manifest SHA in `method_settings`, and explicit `recipe._repository_root`.
Checkpoint restoration checks exact source receipts, TRAIN IDs/classes/counts,
permutation and fixed weights. Original GNNM remains the required uniform-risk
comparison on new datasets. Original paper experiments and scores are untouched.

The scientific ancestry is the audit's C2 rule and the existing native block
updater. Novelty and predictive benefit are unknown; this package makes no new
literature-reading, benchmark-win or conference-acceptance claim.
