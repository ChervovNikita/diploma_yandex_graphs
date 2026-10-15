# Four explicit NCN interaction recipes

This source implements the reviewed `interaction_link_ncn_plan_v1`. It creates
one fresh, original NCN / ogbl-collab boundary session, assembles declared
tensor components directly, and owns a single state/update/validation proxy.
It does not chain standalone transforms. Original source classes, Parameters,
source optimizer groups/options, operators, minibatch edge mask, negative
sampler, four route/dropout streams and serving query order remain native.
NCNC and BUDDY are rejected by this port.

| Method | Components | Shared/private credit | Serving |
|---|---|---|---|
| `interaction_geometry_input` | BE + portable input correction + original endpoint contrast | shared F; private F+0.05A | Mean raw logits |
| `interaction_responsibility_private_pool` | BE + corrected endpoint responsibility v2 + private pool | shared F; private 0.5W+0.5L_mean | Mean raw logits |
| `interaction_input_meanJ` | BE + portable input correction + mean J | all-block 0.5F+0.5L_mean | Mean raw logits |
| `interaction_input_conditionalJ` | BE + portable input correction + conditional20 J | all-block 0.5F+0.5L_actual | Actual gated raw logits |

F includes all 65,536 positives and 65,536 source-sampled negatives per route,
using NCN's sum of group means. The task-specific I1 contrast is the original
link endpoint objective with private credit, temperature0.2/coefficient0.05,
up to1,024 existing anchors and16 existing endpoint-compatible opposite-group
alternatives. Unsupported anchors are reported; the source task rows remain.

I2 uses the corrected `link-resp-v2` semantic endpoint law, floor0.25 and
normalization within each route and original source group. This is not the
earlier responsibility hash. I1/I2 collect shared/private gradients at the
same old state before one original Adam step. Their logged private loss is
not a fictitious joint scalar objective. I3/I4 use one ordinary J backward.

The portable correction sits after the native64-wide logical input affine's
retained BE transform. Rank8/four disjoint QR frames fit this width; zero up
matrices give an exactly zero initial correction. Its4,096 Parameters join
the original encoder group atLR0.0082; factor seed is fit seed+901003.
Conditional20 uses the frozen detached confidence/JS descriptors and convex
raw-logit floor0.05 per route. Its20 Parameters join the original head group
atLR0.0037. No new optimizer group or extra native forward is added.

The observer authenticates every actual source task call before capture. A
prepended final-leaf prehook records preheads before final BE factors. Each
recipe retains the complete source update and exact epoch accumulator,
including the native dropped tail. Checkpoint identity pins the source,
ingredients, source data, seed, component/optimizer ownership, full native
model/Adam/sampler/global-and-private-RNG/modes/epoch state, and last actual
query plan or responsibility digest. Restore reconstructs the saved query
receipt with explicit device-colocated gathers and CPU authentication.

VALID uses the complete60,084-positive+100,000-negative native query order.
The same selected bank's arithmetic mean is returned/cached from the same
native call. Its zero-forward getter requires fresh VALID of the currently
loaded update, and the cache is invalidated before TRAIN or restore. The
generic runtime must separately bind and persist this sidecar to its loaded
selected checkpoint. This factory selects no epoch or checkpoint itself.

**Readiness:** local ingredient/algebra checks only at source freeze. No
native model construction, optimizer update, predictive result, convergence
claim or novelty claim follows from this source. A bounded real-data CPU
lifecycle gate is proposed separately and requires resource admission before
execution. CUDA lifecycle, two-worker resource admission, production selected
mean export and final TEST reconstruction remain pending. TEST is closed;
the comprehensive scientific campaign still needs user GO.
