# Source-only feasibility: accuracy-first GNNM graph views

## Finding and scope

The proposed comparison is implementable with an isolated graph-view driver and an untied bank wrapper while retaining the pinned per-member Polynormer operations. It is **not a drop-in use of immutable v6**. This note specifies future changes; it supplies no numerical qualification, resource admission, execution authorization, accuracy evidence or novelty conclusion. Only source text and file hashes were inspected. No datasets, tensors, checkpoints, servers or running studies were accessed.

The operative label/exposure contract is `AMENDMENT_LABEL_VISIBILITY_AND_EXPOSURE.md`; the original `DECISION.md` remains unchanged. `FEASIBILITY_SOURCE_BINDINGS.json` records exact file hashes, line ranges and excerpt hashes for the claims below. All 36 v6 manifest payloads matched their declared bytes and SHA256 before this note was written.

## Where a view acts

The recipe constructs ten local GAT layers and one global-attention layer, with width 512, input/local/global dropout 0.2/0.3/0.3, and 200 local followed by 2,500 global updates. [B01, B02]

Every forward executes the complete local loop **before** consulting `_global`. Each local layer consumes `edge_index`; global mode then feeds the accumulated local representation through global attention and the global classifier. Consequently, a deletion view changes neighborhood evidence in both stages. The local body continues receiving gradients during global training; only the local classifier is inactive. Local-stage inactive parameters are global attention, its input LayerNorm and the global classifier. A view-local representation cannot be cached across global updates. [B03, B08]

Global attention accepts only `x`. Its Q/K/V and node-axis reductions contain no `edge_index`: the deletion mask affects global predictions through their local input, not by restricting global attention to retained edges. Each member must retain its own complete local states and global reductions. [B04, B06]

Native graph preparation is `to_undirected → remove_self_loops → add_self_loops`; GAT receives that edge list with automatic self-loop insertion disabled. There is no precomputed degree-normalized matrix to replace. Views should delete both directions of selected non-loop pairs, retain every native self-loop and use the unchanged GAT neighborhood-attention operation. This resolves the original memo's generic “recompute native normalization” wording according to the actual implementation. [B05]

## Minimal future patch design

1. **Bind fixed views and TRAIN visibility.** Create separate protocol/role descriptors exposing every official TRAIN label for both CEs. Only pairs whose endpoints are in TRAIN may enter equal-class/different-class deletion categories. Freeze masks before fitting; bind split, ordered TRAIN IDs/labels, native edge identity, pair-selection rule, sampling seed, rounding and view hashes. The saved 10% probe choice needs an explicit denominator and eligible-edge counts. No VALIDATION/TEST labels enter view construction. The random-deletion null also needs a fixed count/degree-stratum matching rule. Source inspection establishes neither category sizes nor semantic strength.

2. **Use existing member forwards.** `forward_member(features, edge_index, member)` already permits different graphs, although the ordinary family `forward` supplies one graph to all members. A new driver can execute four complete native passes and four complete assigned-view passes per update, in the same declared order for every bank. Use `(1/4) Σ_m [CE_m(native, TRAIN) + CE_m(assigned_view, TRAIN)]`, one backward and one Adam step, with the native dropout policy enabled on every training pass. Retain the exact loss coefficients; do not average the eight CEs with an extra factor of one-half. Pair declared RNG streams where shapes permit and record graph-induced differences. Charge all passes. [B06, B09, B10]

3. **Untie the same parameterization.** The tied family shares the entire interior plus boundary W, with private R/S/B at stem and both classifiers. For the comparator, give each member its own copies of every shared parameter, including GAT weights, gates, normalization and boundary W, while retaining its corresponding R/S/B. Copy the initialized tied image so effective initial functions agree; avoid a post-wrap reset. Keep complete trajectories, persistent assignments, loss reduction, Adam defaults and one synchronized pooled selector/transition. Do not substitute unmodulated native copies or per-member CE rescaling. Shared-gradient aggregation versus separate copies is the intended intervention. [B07, B10, B11]

4. **Persist assignment and stage clocks.** Persistent banks give two fixed members each view. The shuffled bank uses the same two-per-view allocation at every update, balanced per member over each even-length stage (200 and 2,500 updates). Freeze schedule identity and actual-stage/update cursor. At the stage transition, restore selected-local model+Adam while preserving live end-local RNG and the actual schedule cursor; never derive assignments from rewound Adam steps. Checkpoints/replay need view descriptors, assignments, every core's stage flag and all paired RNG state. [B12, B13]

5. **Retain native deployment and selection.** Evaluate every bank on the full native graph after each update. Keep strict pooled VALIDATION correct-count improvement, earliest ties, one best record across stages and final restoration of the selected local-or-global stage. Pool class probabilities arithmetically across all four members. VALIDATION remains an exploratory development endpoint; any confirmation needs a separately frozen protocol and disclosed Amazon evaluation history. [B14, B15]

## Confounds that invalidate a purported matched comparison

**Label budget:** v6 deterministically reserves approximately one-fifth of each TRAIN class as control, requires prior-role receipt equality, and fits only FIT labels. Its primary endpoint is TRAIN-control NLL. The amendment uses all TRAIN and treats TRAIN scores as fit diagnostics. Reusing those roles, gates or current fits as decisive same-budget controls would violate the proposal. A separate role/protocol binding is necessary. [B16–B18]

**Selection opportunity:** conventional v6 independent fits each have their own selector and local transition; evaluation concatenates their separately selected checkpoints, potentially from different stages. They are useful disclosed references, but cannot isolate tying against a bank with one pooled selector. The new untied bank must share the proposed bank clocks and selector. Competent single/ordinary independent references also need the amended TRAIN budget. [B19]

**Implementation custody:** current construction admits only two kinds; core/primitives/inactive-gradient checks assume the existing layouts, and restore/replay enforces exact state inventories. The untied wrapper and eight-pass objective require corresponding isolated ownership, stage, shape, checkpoint and replay metadata changes. Existing v6 qualification cannot be inherited for those changes. Unequal pass/dropout exposure, changed initial functions or unrecorded schedule rewinds would confound the result. [B20]

The bounded conclusion is source feasibility with explicit protocol separation. Neither useful specialization nor a benefit from tying follows from this inspection.
