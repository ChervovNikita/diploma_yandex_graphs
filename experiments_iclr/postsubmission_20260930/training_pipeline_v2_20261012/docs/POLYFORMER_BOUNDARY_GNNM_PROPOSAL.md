# Same-backbone boundary GNNM proposal for pinned PolyFormer

Status: **compatible source-based adaptation, not implemented or counted**.
The frozen standardized PolyFormer v1 and unchanged author CLI packets remain
immutable. This proposal does not execute a model or add a scientific fit/GO.

The separate native SAGE GNNM on the same node custody is a useful reference.
It does not supply a boundary GNNM whose shared body is PolyFormer. The latter
is feasible from the pinned source and needs its own finite contract.

## Exact sites and source laws

| Existing PolyFormer site | Proposed replacement | Shared remainder |
| --- | --- | --- |
| `lin1: Linear(features, hidden)` on `[N,K+1,features]` mono tokens | Native `BEBlock(features,hidden,is_first=True)` | All native attention/FFN blocks and `lin2` |
| `lin3: Linear(hidden,classes)` on `[N,hidden]` after `lin2`/ReLU/dropout | Native `BEBlock(hidden,classes,is_first=False)` | Original logits, with no extra classifier map |

These replace existing affine maps rather than adding layers. The native
`BEBlock` supports these leading dimensions through normal broadcasting and
`nn.Linear`. It supplies the actual member-private R/S/B law: first input R
Rademacher, later R and all S1, B0, bias-free shared Xavier W. No internal
PolyFormer map is factorized, and no private attention/message factor is
introduced. The existing `polyformer_internal_be` all-map geometry retains
author slow weights, R/S factors and shared ordinary biases; it is a distinct
method and cannot be renamed as this boundary model.

Use the already pinned original BEBlock source
`common_wrapper_native_source_20261010_v1/models.py`, SHA256
`07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303`.
Construct the pinned PolyFormer body first, replace `lin1`/`lin3`, and bind
that constructor order. Removed ordinary boundary tensors must be absent
from named parameters, optimizer groups and checkpoints. Other slow parameter
objects and source K/q/multi/heads/FFN/dropout decisions stay unchanged.

The original GNNM reduction is one own CE forward/member and separate CE/M
backward calls at the same old parameters before one optimizer update.
M4 serves `softmax(mean(raw_member_logits))`. Its matched M1 serves one raw
logit vector; genuine I4 also uses mean raw logits for this declared family.
Keep private route/view streams, own selectors, full2000 update horizons and
the prepared TRAIN/SELECT/CAL/evaluate roles. These RNG/role/horizon/backbone
choices are standardized adaptations, not unchanged published GNNM or
PolyFormer author protocols. The original eight author CLI processes are
separate and unchanged.

## Optimizer decision and finite domain

Recommended source-family contract: one **AdamW**, betas(.9,.999), eps1e-8,
weight_decay0 for every group, retaining PolyFormer's source-scoped joint
attention/ordinary LR pair and grouping. The LR grouping is an explicit
PolyFormer-backbone adaptation; original GNNM's optimizer family/zero decay
and member-gradient reduction remain visible. Zeroing both decay groups
collapses the original two decay choices, so the effective domain is
**40 configurations**, not80: five joint LR pairs × two widths × two depths
× two dropouts. Decay is locked0 and not searched as an ineffective axis.
Native source K/q/multi/heads/FFN/attention dropout stay locked per dataset.

An alternative standardized comparison can keep native grouped PolyFormer
Adam and its original decay-pair domain, giving80 configurations. That changes
the original GNNM optimizer/regularization law and must be labeled separately.
No duplicate ineffective zero-decay or factor-seed search cell is allowed.

## Necessary matched construction controls

Add two separately searched procedures for each of the four prepared datasets:

1. `polyformer_original_boundary_own`: one shared slow body, four native
   boundary R/S/B rows, source mean member CE reduction and mean-logit serving.
2. `polyformer_boundary_equipped_M1_own`: one slow body, one row of those same
   two native boundary blocks and initializer law, unscaled own CE. Derive its
   genuine I4 from four paid, independently seeded bodies and four own
   selectors; reuse the prospectively fixed body0 as M1, never the best member.

The matched M1/I4 retain the same source W initializer/private bias geometry
and optimizer as M4. This permits a sharing comparison without changing the
slow initializer or private boundary equipment. Existing competent native M1,
mean-four-dropout M1, all-map equipped M1/I4 and internal shared4 remain.

For S=22 existing splits and three comparative seeds, the recommended40-point
contract adds640 search starts,330 comparative starts,8 selection endpoints
and198 exports/CAL/evaluations each: **970 paid fits and1,572 records**,
**1,940,000 optimizer updates**. It adds66 genuine I4/fixed-body0 reuse banks
each and594 serving-route views. Potential combined totals would be174,458
fits,185,492 records,144 selectors,3,630 exports/CAL/evaluations each,
1,254 I4/body0 reuse banks each,10,758 serving views and348,916,000 updates.
These counts are prospective formulas and are not registered packet work.

Keeping the80-point grouped-Adam alternative instead would add1,610 fits,
2,212 records and3,220,000 updates, with the same comparative/export/I4
rosters. Root must freeze the chosen optimizer/domain before preparation/GO.

## Concrete implementation work

- Add a distinct boundary geometry/adapter that reuses pinned PolyFormer
  construction/preprocessing without calling `install_factors` on its body.
  Dispatch native boundary members in `forward_route`; keep the actual
  pre-`lin3` representation live and the source parameter groups correct.
- Extend mode/schema/provenance to bind source BEBlock, R/S/B site names,
  native Xavier/init law, members, source order and declared adaptation.
  Existing three-mode `polyformer.py` currently rejects this geometry and
  `mean_logits`; these are concrete missing execution paths.
- Extend the dedicated PolyFormer exporter to admit and preserve the declared
  mean-logit fusion for these banks while retaining strict own checkpoint,
  four paid I4 body IDs/seed/private-stream ancestry and target-free export.
  `node_endpoints.pooled_log_scores` already supports mean logits and the
  existing fixed post-pool scalar CAL/evaluation can be reused.
- Add source/init/dense-site/gradient tests for the actual two replacements;
  M1/body0 objective/state equivalence, four paid I4 selectors, restore and
  target-absent exports; verify source-family CE/M old-state backward law
  against the pinned native function and mean-logit versus mean-probability
  serving explicitly.
- Compile a fresh PolyFormer v2 reusing the four existing custodies by hash,
  append these finite controls, update comparison identities/actual counts,
  and qualify maximum shapes/resources separately. Preserve v1 and author
  sources. No scientific execution before whole-master gates and human GO.

Expected implementation/CPU qualification/preparation effort is about one to
two focused hours; GPU resource qualification is separate. This is feasible
work, not a permanent protocol incompatibility. No published-task, bitwise
environment, score-match or manuscript acceptance claim follows from it.
