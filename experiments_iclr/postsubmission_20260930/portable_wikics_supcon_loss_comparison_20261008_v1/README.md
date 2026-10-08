# Matched WikiCS cross-view and canonical SupCon losses

This public successor supplies two full native training conditions:

| Condition | Auxiliary loss |
|---|---|
| `class_full_cross_view` | Pinned original class-full opposite-view alignment, BotSCL Eq13–14 loss adaptation |
| `supcon_eq2` | Canonical SupCon Eq2: uniform same-label positives across both views, exact anchor excluded, positive average outside the log |

Both use mean-member two-view own CE plus `0.05` auxiliary loss, temperature
`0.2`, shared four-member unit-factor BE, the same pre-task-head representation,
all 580 own-CE labels, and the original fixed 512-node auxiliary panel. Both
views receive gradients. There is no projector, residual repulsion, cross-member
contrast or new recipient rule. Shared/private upstream parameters, including
stems, receive the auxiliary; the captured task head receives own CE only.
This is a native joint-CE SupCon loss adaptation, not its original projection
pretraining and frozen-classifier recipe.

## Source and data dependencies

The sealed `portable_internal_be_public_interface_20261007_v2` is the only
external code dependency. Its manifest and every payload are verified before
use. The bundled replay is byte-identical to public attribution and author
Wiki12 v2, SHA256 `1554e9c6b181d68be937596530e809d1133ea6a682faa04c155a533856156d73`.
The baseline facade calls the original loss directly. `losses.py` also contains
an independent callable cross-view implementation for semantic equivalence
checks and implements the new canonical loss.

Supply the pinned Polynormer `model.py` and complete numeric TRAIN/development
NPZ roles described by the public v2 package. Before Session construction,
this successor checks every ordered feature, edge, TRAIN ID/label and
development ID/label fingerprint against v2's public WikiCS pins. It reads no
TEST role. NPZ serialization may differ, but the six ordered arrays must match.
The complete graph has 11,701 nodes and 442,907 prepared edges; development
contains 5,274 nodes, the official split0 validation/stopping union.

Use an already prepared compatible Torch/PyG/OGB environment. No dependency
installation, model construction or data loading occurred while preparing
this source package. The inherited dependency list is in `requirements.txt`.

## Complete training entry

```sh
python portable_wikics_supcon_loss_comparison_20261008_v1/train.py \
  --condition supcon_eq2 --seed 6101 --device cuda:0 \
  --public-interface portable_internal_be_public_interface_20261007_v2 \
  --polynormer external/Polynormer/model.py \
  --train roles/wikics/train.npz --valid roles/wikics/valid.npz \
  --output runs/wiki_supcon_6101
```

Use `class_full_cross_view` for a fresh baseline only if root's frozen reuse
decision requires one. Seeds are restricted to 6101/6203/6307. The CLI exposes
no coefficient, temperature, horizon, projector, early-stop or resume options.
It retains the original Adam, 1,100 epochs (100 local +1,000 global), native
joint local restore, live member RNG streams, strict-first best complete
development checkpoint and mean class-probability serving. The exact original
`torch.linspace(..., steps=512, device=labels.device).long()` panel remains in
the byte-identical replay; it is not rebuilt by the loss module.

The original full driver is reused. Output-cotangent replay accumulates eight
member/view VJPs before one Adam update, with eight shadow and eight replay
member forwards per update. Complete execution charges 17,600 training and
4,400 development member calls. Source identity, active loss calls, actual
providers and inclusive execution costs are recorded. Canonical SupCon has
four times the score entries (16 MiB vs 4 MiB in float32 before buffers/backward)
and no additional backbone passes. Hardware/runtime/resource qualification
and scientific execution remain root-owned and pending.

## Anchor reuse decision before outcomes

`REUSE_QUALIFICATION.json` binds source, recipe, dropout streams, selector,
scaling, native source and data-preparation metadata without reading tensors,
checkpoints or pending outcomes. Its conclusion is **source/recipe eligible**
for the three already registered `alignment_only` Wiki12 anchors, conditional
on root freezing their exact execution/runtime/data identity and later passing
whole-family completion gates. This does not certify their completion or
numerical execution and does not select anchors by accuracy.

Reserve all three anchors together before outcomes are opened. Do not use
`combined` or the original `be_unit_contrastive` condition with both losses.
Do not rerun or replace original paper scores. If the static/execution bindings
fail, root must choose a fresh matched six-cell comparison before scoring; the
public caller default environment is not automatically equivalent to the
author runtime. A source-level loss equivalence is not bitwise trajectory proof.

## Frozen semantic fixture

```sh
python portable_wikics_supcon_loss_comparison_20261008_v1/test_semantics.py \
  --public-interface portable_internal_be_public_interface_20261007_v2
```

This CPU fixture uses literal small tensors and synthetic own-CE rows only.
It checks original cross-view value/gradient equality; canonical Eq2 against
an explicit anchor loop and analytic both-role gradients; distinction from
Eq3; singleton positives, self exclusion, normalization-floor finiteness,
absence of cross-member gradients, complete580 two-view own-CE scaling and
unchanged RNG. It imports only Torch and the original objectives module, never
a Session/model/dataset/checkpoint. Root will run it once in its existing CPU
environment; it was not run locally because no local Torch runtime is present.

Static AST/dependency/CLI checks passed locally. No fit, data conversion,
remote call, current outcome inspection or installation occurred. Three seeds
on the selected development graph remain exploratory. Literature v19 and
publication prior `0b853680` are the authoritative scientific context; this
source adds no novelty, superiority or original-paper-score claim.
