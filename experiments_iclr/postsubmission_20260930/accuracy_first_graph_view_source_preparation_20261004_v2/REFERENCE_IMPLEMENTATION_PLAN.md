# Missing full-TRAIN quality references: implementation plan

This is a separate prospective plan for root implementation. It adds no executable reference, fifth bank, experiment or release to the four-bank package. Keep its 10% semantic views and native operations unchanged. Accuracy of native-graph predictions is the objective; parameter saving is secondary. View diversification remains an attributed baseline/probe.

## Reference definitions

| Reference | Model and training objective | Training passes per actual update | Selector |
|---|---|---:|---|
| Modern view-augmented single | One ordinary unwrapped pinned Polynormer; `CE(native, TRAIN) + 0.5 CE(equal, TRAIN) + 0.5 CE(different, TRAIN)` | 3 | Its native-graph VALIDATION prediction |
| Conventional native single | One ordinary unwrapped pinned Polynormer; `CE(native, TRAIN)` | 1 | Its native-graph VALIDATION prediction |
| Conventional independent4 | Four separately initialized ordinary native fits, each with its own native CE, Adam and clocks; fixed probability pool of their selected predictions | 1 per physical fit; 4 total | Each physical fit selects independently; pool only after selection |

The augmented single's coefficients match the four-bank objective's native coefficient 1 and bank-level equal/different coefficients 0.5 each. Three versus eight training passes remains a disclosed compute difference; neither coefficients nor architecture make it a compute-matched bank. Native-only references intentionally receive no probe supervision. The conventional ensemble has four selection opportunities; it is a quality reference, while the existing synchronized untied bank isolates tying.

Use every official TRAIN label in every reference. The same split's fixed semantic masks, TRAIN role/label hashes, raw feature identity, native edge ordering and VALIDATION targets must match the banks. VALIDATION/TEST labels cannot construct views or enter updates. TRAIN scores are fit diagnostics. The prior v6 FIT-only physical fits cannot be reused as these all-TRAIN references.

## Small source addition

Implement a separate reference driver using the already byte-bound ordinary `native.Polynormer` class, without R/S/B or shared weights. Reuse the native constructor/device-reset order, width 512, ten local/one global layers, dropout 0.2/0.3/0.3, Adam 0.001/zero decay/default betas/eps, and the fixed 200-local + 2,500-global actual-update recipe. No HPO, freezing, response penalty, early stopping or new recipe selection is proposed. Keep all local layers active in global mode.

For the augmented single, run native, equal and different in that fixed order; backpropagate each CE times its coefficient immediately, then take one Adam step. All three forwards see pre-step weights and ordinary native dropout. Pair predeclared native-pass streams where compatible and record any shape/stream differences. The conventional fits use one native forward/backward and one step. Charge every backward, native validation pass, snapshot and restoration.

Keep one strict correct-count selector per physical fit spanning both stages, first candidate after update1, earliest ties. After actual update200, restore that fit's selected-local model+Adam while preserving its live end-local RNG and actual clock; set `_global=True`. After all2,700 updates, restore its overall selected checkpoint and exact stage, including a local winner. Adapt inactive-gradient checks to ordinary native head names. Store source/protocol/input hashes, device provenance, complete model/Adam/RNG/modes/gradients, selection and actual clocks; do not call immutable v6's role-gated driver with substituted labels.

## Initialization and pooling

Freeze three blocks `(split, block seed) = (0,17), (1,29), (2,43)`. The augmented single uses the block seed. Use v6's declared conventional member seeds `block_seed + 1009*m`: `[17,1026,2035,3044]`, `[29,1038,2047,3056]`, `[43,1052,2061,3070]`. Each fit has an independent stream and fresh native construction/reset. The native single may be an explicitly declared alias of conventional member0, saving a duplicate identical fit; never select the best member as the single. Thus this plan needs five physical fits per block: four native fits plus one augmented single.

Pool all four conventional members by arithmetic mean of class-softmax probabilities, with fixed membership and no temperatures/weights. Retain each member's separately chosen stage/update. Report all paired blocks, selected native-graph VALIDATION accuracy/NLL/macro-F1 and member competence, plus TRAIN diagnostics and complete costs. One favorable block or lower response correlation cannot establish quality. Competence and a worthwhile gain threshold must be prospectively specified before outputs; no improvement is assumed here.

## Implementation verification and limits

Before scientific use, test the single objective/gradient accumulation against the summed loss, head activity in both stages, full restoration and next-update replay, earliest ties/local final winners, full TRAIN visibility, native-single/member0 alias identity, independent selector separation and exact probability pooling. Small synthetic checks verify implementation only; full-shape/resource behavior and input-projection provenance remain separate requirements.

These references complete the missing quality controls; they do not create independent heldout evidence. VALIDATION remains selection-associated development evidence. Any later confirmation must freeze its own blocks/conditions/endpoints and disclose Amazon's consumed split/TEST history, without assuming an unseen TEST.
