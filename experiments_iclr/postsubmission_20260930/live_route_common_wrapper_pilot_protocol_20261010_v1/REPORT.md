# One residual-SAGE pilot proposal

**Status: proposal complete; scientific protocol unfrozen, pending root review.**

Prepare one complete seven-arm, three-seed SAGE family. Ordinary and all-map factorized single/independent references have priority. Their competence remains unestablished until measured. No trainer, launcher, fit, model import, current partial-quality read, or TEST payload was used here.

## Fixed proposed screen

Use the existing residual SAGE family: stem Linear → dropout → GELU; two pre-LayerNorm residual SAGE/FFN blocks; concatenate ego and SAGE message before each two-layer GELU FFN; final LayerNorm/classifier. Proposed settings are H128, depth2, FFN multiplier1, dropout0.2, AdamW LR0.001/weight decay0, constant LR, full-graph TRAIN CE, and paired seeds0/1/2. These are the archive's **prespecified source default**, not a quarantined validation winner. Freeze one setting per method prospectively, with no grid. Root will bind exact common-wrapper maps, initializations, connector geometry, seed streams, and final criteria.

| Arm | Acquisition and selection |
|---|---|
| Ordinary native M1 | One ordinary body; unscaled own CE; own checkpoint |
| Genuine ordinary I4 | Four independently trained ordinary bodies, parameter storage, optimizers and RNG streams; unscaled own CE and own stopping/checkpoint for each |
| All-map factorized M1 | One factorized body with every reviewed eligible linear map; unscaled own CE; own checkpoint |
| Genuine all-map factorized I4 | Four independently trained factorized bodies; separate W/factors, optimizers/RNG and own stopping/checkpoint; unscaled own CE |
| Unchanged shared4 | Reviewed common wrapper with four complete routes; mean of four own CE losses, one accumulated update; pooled checkpoint |
| Equal-size separable | Exact matched connector allocation without route exchange; same shared4 loss/optimizer/init/selector |
| Exchange | Exact reviewed route exchange with the separable arm's parameter/storage budget; same shared4 loss/optimizer/init/selector |

I4 is pooled only after each body's own selected checkpoint is restored. It is not the historical ENS arm, which used mean1/4 loss and one pooled checkpoint. Cloning trained teachers is not independent acquisition. All-map factorization is not the native source's boundary-only TABM model: the latter factorizes only the input/output projectors. Its old configuration supplies no competence evidence for all-map M1/I4.

**Prospective horizon/selector:** at most1,000 updates; VALID after every update; save every first strict accuracy improvement, retain earliest ties; stop after300 consecutive non-improving updates. M1/I4 bodies use their own accuracy; the three four-route banks use accuracy of mean raw logits. No CE tie-break or probability-pooling selection is proposed. This differs from the common runner's cap5,000 and ten-step commit quirk, and from the archive's full1,000/no-stop accuracy→CE→earliest rule. Original scores remain unchanged.

## Data and evidence provenance

Use the **current authorized safe WikiCS split0 roles**, with final code/data identity bound by root. Saved role evidence records N11,701/F300/C10, TRAIN580, VALID5,274=`(val_mask | stopping_mask)[:,0]`, and graph processing `to_undirected → remove_self_loops → add_self_loops`:442,907 directed entries including11,701 self-loops. Preserve the existing feature tensor rather than substitute archive raw features. Trainer labels are projected TRAIN/VALID only. The TRAIN/VALID NPZ hashes and six ordered-tensor fingerprints are in PROTOCOL.json; these were read from saved metadata, not newly recomputed from arrays.

The old official split0 source used TRAIN580/VALID1,769, raw JSON features,431,726 symmetrized entries, and no explicit self-loops. Its published raw JSON hash agrees with current custody, but its data/selector population and feature tensor are different. None of its arrays, selected settings, scores, or timing is bound to this pilot.

The 432/54 archive audit establishes source/default/data/horizon/selector facts. It does **not** establish the actual scientific host authorization: the432-folder CUDA preflight is absent; matched54 preflight identifies only `cuda:0`; compact scope says host launch/GPU records were omitted. No concrete hostname/GPU-UUID authorization chain was found in the scoped local records. Seven-GPU scientific evidence is excluded by project instructions. All historical432/54 winners, quality, and costs are quarantined pending root adjudication. Earlier suggestions that those selected numerical results established competence are superseded here.

Readable references:

- [Common model source](../common_wrapper_native_source_20261010_v1/models.py), head `47c4af6228c52aa450ab8316efe5c7cfc33119a4`; models SHA256 `07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303`, identical to the archive model file.
- [Archive source protocol](../../validation_tuning_prepared/STUDY_PROTOCOL.md) and [FROZEN_STUDY.json](../../validation_tuning_prepared/FROZEN_STUDY.json): default and exact old data/selection law; numerical outcomes are not admitted.
- [Current safe-role provenance note](../WikiCS_normal77_official_role_provenance_scoped_note_20261008_v1.md), [role manifest](../learnable_internal_be_safe_role_export_execution_20261007_v1/metadata/outputs/wikics/ROLE_MANIFEST.json), and [public data pins](../portable_internal_be_public_interface_20261007_v2/PUBLIC_DATA_PINS.json).
- Platonov et al., [A critical look at the evaluation of GNNs under heterophily](https://arxiv.org/html/2302.11640v2), Table4/AppendixA; saved author commit `a431395582e929d88271309716bea4fe24ce6318` uses H512/dropout0.2/LR3e-5/1,000 steps and depth1–5 search. Paper Adam versus source AdamW/wd0; scheduler helper was not read. This supports the architecture's ancestry, not H128 WikiCS transfer competence.
- Mernyei & Cangea, [Wiki-CS](https://arxiv.org/abs/2007.02901), [official repository](https://github.com/pmernyei/wiki-cs-dataset). Retrieved DGL GCN/GAT presets do not establish an official residual-SAGE setting. README was fetched from `master`; train sources are commit-pinned. The tree response was truncated and unparseable; no whole-tree claim is made.

## Decision and cost

After all21 groups close, report per-seed pooled accuracy/NLL, each member's accuracy/NLL, correct-alternative coverage, losses during pooling, repairs/harms versus shared4, and actual acquisition/serving/storage/memory. An exchange gain over shared4 alone cannot distinguish factorization or added capacity: factorized M1/I4 and equal-size separable are essential. Weak references leave the competence question unresolved; no partial-family result chooses a new setting. Root will lock quantitative criteria before execution. This is encountered development evidence, not unused confirmation.

The roster contains39 body/bank acquisition units if I4 bodies are separately owned. At the1,000-step cap it requires at most66,000 training route trajectories plus66,000 VALID trajectories and selected-checkpoint readout; connector arithmetic is additional. Eligible completed PubMed private-hop18 took6,975.969s worker/7,011.153s owner, but its architecture, graph, horizon and overlap differ. It is a loose scheduling anchor only. No defensible SAGE wall-time estimate is established; fresh finite qualification must supply it. No legacy432/54 timing is used.
