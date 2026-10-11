# Native SAGE Packed Ensembles comparison source

Preparation only. One ordinary native SAGE PE4 γ1 point is fixed at **width 59 and 215,508 parameters**, under the current F BASE cap of 216,674. Width 60 has 221,560 parameters. Input/output maps, biases, private feature LayerNorm and all depth 2 SAGE/FFN parameters are counted. There is no E-budget variant or width search.

## Existing coverage and reuse

`COVERAGE.json` records the check made before this source was written. The transformer PE port passed seven synthetic CPU tests, with scatter fallback warnings; that is engineering coverage for other backbones. The older `efficient_paths` port supplies a genuine grouped SAGE equation, native-weight transfer and exact whole-model count. Its profiling is engineering evidence, and its width 44 channel budget, Xavier starts, sparse-operator interface and common packed RNG do not supply the current F comparison. The F learning contract explicitly deferred native PE γ1. The completed spectral PPI family concerns tied/selected/random/raw directions, rather than this comparator.

The outstanding scientific comparison is the complete three-seed PE point versus the same-role F shared bank and capable singles, with **complete full-width genuine factorized I4** retained as the decisive quality reference. The source adds only the PE arm; anchor admission and comparison remain root-owned.

## Implementation

`packed_sage.py` is a stdlib-only factory. Root supplies Torch and four fresh ordinary `models.Model` bodies using the existing F constructor/seeds. It copies their initialized values into grouped private affine/norm coordinates without another reset, and keeps no full native-body owner. All trainable coordinates are private. Native `MeanAggregation` alone is shared because it has no parameters. The factual directed edge list, duplicate multiplicities, self entries and zero-degree mean behavior remain unchanged. Relative to the width 128 native reference, α is 4×59/128 = 1.84375; the published construction's γ1 grouped maps remain valid.

States are `[node,member,59]`. One channel-packed native mean acts per layer; grouped dense maps retain the private neighbor/root/FFN/stem/head blocks. Feature LayerNorm uses each member's native fused affine operation. Each dropout call resumes the existing member stream. There are four nonlinear trajectories and 236 edge channels. This is a declared native Packed-Ensembles graph adaptation, without an exact author-code or timing claim.

`family_adapter.py` returns a subclass of the frozen F `Family`. Its inherited `fit`, metrics, objective, strict selector and model/Adam/stream restoration are unchanged. Only construction, four member-slice records and one-arm collection are added. Training uses mean own-member CE with every TRAIN row for every route; one AdamW optimizer and one coherent probability-accuracy bank selector match shared F. Members start from distinct native seeds. Full-width I4 keeps its four own optimizers and selectors; packed mean CE is not called exact unscaled-I4 Adam equivalence.

The fixed recipe is depth 2, FFN multiplier 1, LayerNorm, dropout 0.2, AdamW lr 0.001/decay 0, maximum 1,000 updates/patience 300, seeds 7301/7403/7507 and arithmetic probability pooling. Selection retains the earliest strict full-VALID accuracy maximum without an NLL tie rule. The selected output has F's `[4,VALID,10]` logits and `selected_VALID.npz` schema. `COMPLETE_PE_BANKS.json` closes only this arm and supplies no comparison verdict. Root can reuse the F reader's calibration/scoring functions after admitting complete PE banks and exact anchor custody; its fixed five-arm roster cannot accept PE automatically.

## Root integration

Load the existing F owner and these two source modules with the existing stdlib source loader, then call `family_class(owner, packed)`. The resulting class accepts the same F config/output/hash/seed-block arguments. The template has a null root admission, so construction rejects before the parent's numerical imports. There is no CLI, subprocess, scientific launch or admitted fit in this packet.

One future full actual TRAIN qualification should compare packed outputs/gradients to the four supplied native bodies, check count/optimizer ownership, private dropout replay and complete model/Adam/stream restore. Supply factual TRAIN inputs directly to `pack_native`; no toy suite or VALID scoring is prepared. A qualification pass and finite root supervision must be bound before acquisition. Temporary constructor/packing time is charged inside inherited acquisition; root must measure its memory peak too. Source preparation cannot forecast runtime or memory.

`PROTOCOL.json`, `CONFIG_TEMPLATE.json`, `ROOT_ADMISSION_TEMPLATE.json` and the source bindings make the point concrete. Static AST compilation, count arithmetic and file hashes are the only local checks. No numerical/model imports, data/prediction reads, execution, remote access, reviewer or canonical edits occurred.
