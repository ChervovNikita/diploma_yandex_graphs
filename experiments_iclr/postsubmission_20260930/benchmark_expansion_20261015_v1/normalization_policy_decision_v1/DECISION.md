# Proposed normalization policy after the completed factorial

**Recommendation:** use route-private running means, variances and counters
for every new shared candidate and its ingredient controls **where the native
model already contains tracked BN**. Keep native BN operators, TRAIN batch
statistics, affine parameters, optimizer ownership, route order and batches
unchanged. Do not insert BN into BN-free models. Keep original GNNM unchanged
and retain the existing explicit private-BN-only control.

This is a comparison-design recommendation, not a claim that private BN always
improves accuracy. It supersedes no source or result in this source-only review.

## Exact fit-count decision

The 25-recipe/27-preset source roster matches the saved **actual compiler4
family/track/method counts**. Every setting has ten cohorts.

| Scope | Presets | Action | Candidate fits changed |
| --- | ---: | --- | ---: |
| MolBACE, four native architectures × two dropouts | 8 | Private native BN state for 17 candidate/control recipes | 1,360 |
| MolHIV, same eight source presets | 8 | Same declared policy, without assuming a quality gain | 1,360 |
| Collab BUDDY | 1 | Same policy across 17 candidate/control recipes; preserve inactive BN branches | 170 |
| PolyFormer node presets | 8 | LayerNorm; no BN change or duplicate private-BN fit | 0 |
| Collab NCN / NCNC | 2 | LayerNorm/identity; no BN change or duplicate control | 0 |
| **Total candidate migration** | **17 BN presets** | **22 recipe IDs have affected applications** | **2,890** |

The other **3,570 fits** retain their scientific definition: 270 source
references, 1,080 native independent bodies, 270 original-GNNM fits, 170
private-BN-only controls and 1,780 BN-free candidate fits. Matched M1 still
reuses I4/body0. Thus **6,460 physical fits plus 270 export-only tasks can be
retained**, with no new normalization cross-product. All historical pilots and
their original state policy remain separately recorded.

Migrating only favorable ingredients would leave normalization confounded.
Migrate the randomized responsibility control, residual-only control, common
context control, pooled mean control and all interaction ingredients as well.
The four interaction recipes account for 680 changed fits; the other affected
recipes account for 2,210. Per-recipe counts are in `COUNTS.json`.

## Comparisons that must change

| Comparison | Consistent reference after migration |
| --- | --- |
| Candidate versus shared baseline on 17 BN presets | Existing private-BN-only fit, same preset/cohort; original GNNM remains a separate reported comparator |
| Candidate versus shared baseline on 10 BN-free presets | Unchanged original GNNM; no invented BN arm |
| Geometry × correction; responsibility × private pooling | Both ingredients and combination share the same normalization; private-BN-only is the neither-ingredient cell |
| Learned/global/conditional aggregation | Matched mean-pool/J and correction controls also use private state; same-bank fixed-mean diagnostics keep the candidate's own selected state |
| Native M1, I4, source references | Unchanged native training and each independent model's own native state |

The inspected compiler control-link logic would redirect **1,690 direct
symbolic G0 references** among the affected fits to the matched private-BN
reference for ingredient attribution. All 2,890 affected candidate comparisons
should expose that reference; retain G0 as a separate total-extension comparison.
Other ingredient links must resolve to their migrated same-policy variants.
This adds control links, not training fits.

## Source consequences

This is **not** an execution-time flag for frozen jobs. Create an immutable
successor defining the conditional policy and honest variant identity for the
2,890 scientific contracts. Update their method/registry callback bindings,
descriptor hashes, producer/state metadata, logical-bank/control links,
reporting identities and downstream source authorities. Keep the existing
roster/source generations and selected states intact. Global manifest rebinding
can touch additional metadata without changing those fits' scientific meaning.

The current molecular private-BN transform explicitly admits base GNNM only;
the BUDDY wrapper likewise requires a base, candidate-free native session.
Both therefore need explicit composition with existing candidate sessions,
preserving each loss and gradient rule. Own-route contexts must cover every
executed native BN call; checkpoints retain complete banks, and VALID/TEST
serving restores them read-only under the genuine producer identity. Existing
base-state inference compatibility does not qualify arbitrary compositions.
Never relabel an old shared-state checkpoint as a migrated method.

Reuse unchanged native/data/population evidence. Qualify only genuinely new
composition/state/serving branches; do not repeat all seeds or the full smoke
matrix. Actual migrated jobs, resource bindings, deployment and user GO remain
future work.

## What the evidence does not establish

The completed evidence is three seeds on one MolBACE/GIN/dropout0.5 selected
VALID split. It supports a fair normalization control, not universal gains on
MolHIV, GCN, virtual nodes, BUDDY, other losses or the campaign's newer portable
residual implementation. It establishes no significance, novelty, TEST transfer
or independent-ensemble superiority. Different selected epochs prevent a
fixed-weight causal reading of the complete training contrasts.
