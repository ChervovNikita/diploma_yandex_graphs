# Disabled native private hop-credit source delta

**Disposition:** one small source adapter and protocol, disabled by `ENABLED=False`. AST and selected-source hash checks pass. No numerical import, model construction, data/model/checkpoint read, fit, remote command, job or scientific source execution occurred. Existing proposal, original source and closed evidence remain unchanged. This packet is not a scientific admission, quality result or novelty claim.

## Exact native interface

The pinned author commit is `d390f39e88d0eaac80318fdc7704bd3bf3cf8b13`; exact inspected local bindings are in `SOURCE_BINDINGS.json`.

`utils.py:367` implements `mono_base`: apply native `gcn_norm`, convert through the pinned sparse path, append X, then append each `spmm` recurrence. K=2 therefore returns ordered `list_mat=[X,PX,P²X]`. `mymodels.py:40` stacks that bank on dim1, producing `[N,3,d]`, and applies lin1. There is no separate raw-X bypass outside `list_mat[0]`. Each block keeps the ordered token MLPs, K-indexed attenuation, learnable `bias_scale`, attention and FFN residual operations; the model then sums the three token embeddings and applies the native output maps.

One persistent zero tensor replaces the complete propagated input block in slot1 and/or slot2. Every bank retains slot0 X and all3 slots. These zeros do **not** remove slots, alter K or force later embeddings to zero: native affine biases, normalization, token networks and attention can make their representations nonzero. No sparse propagation, features, graph, labels, factor sites or decoder are added.

The native source has LayerNorm and no registered mutable buffers. `PolyAttn.bias` is a plain tensor attribute; `FactorLinear.member` is a mutable integer. The adapter guards both around every forward and restores/rejects a mutation. It only copies a bias back if its value changed, so an unchanged saved tensor does not receive an autograd version increment. The existing member context restores the route index.

## Implementation delta

`adapter.py` accepts a fresh exact factual-only V3 Session. For M4 it uses the existing shared4_own body, all23 existing factor sites, unchanged native Adam groups and complete TRAIN CE. It creates three persistent auxiliary banks and adds first-order private cotangents only. All cotangents are accumulated at unchanged parameters before one native Adam transition. There is no higher-order graph or shared auxiliary `.grad` accumulation.

For routes1–3, the private derivative is `(own + .5 assigned)/(4*1.5)`; shared credit is the mean factual CE derivative. Route0's full auxiliary loss is the **same** factual stochastic loss, so its derivative is reused and its private coefficient stays ordinary own1/4. This is equality of the intended derivative, without a claim of floating-point identity to a differently ordered backward accumulation.

Factual and auxiliary passes use the existing separate persistent DropoutStreams keys, paired by seed across M4 conditions. Auxiliary passes are checked not to advance factual states. The native forked stream mechanism restores outer RNG. The adapter records exact forward/gradient/optimizer counts, rejects an incomplete or repeated commit, and retains finite-gradient/native-state checks. The shared/private block field is generally not the gradient of one scalar objective; later shared trajectories may change through the private factors.

## Paired controls and reference source paths

| Source condition | Factual paths | Additional paths | Total / update | Native Adam transitions |
|---|---:|---:|---:|---:|
| Existing shared4_own |4|0|4|1|
| Private missing-hop M4 |4|3|7|1|
| Full-input auxiliary M4 |4|3|7|1|
| Common nonfull-view M4 |4|3|7|1|
| All-block missing-hop M4 |4|3|7|1|
| Factorized M1 ordinary |1|0|1|1|
| Factorized M1 all-view |1|3|4|1|
| Genuine independent factorized I4 ordinary |4|0|4|4|
| Genuine independent factorized I4 all-view |4|12|16|4|

The corrected common-view control retains route0's full factual reuse. Routes1–3 all receive `[X,0,P²X]`, then `[X,PX,0]`, then `[X,0,0]`, cycling every3 updates. Every update pays exactly7 paths. Aggregate nonfull mask exposure matches the candidate over complete3-update cycles; each route's exposure distribution intentionally differs. Finite endpoints and stopping horizons need their complete exposure counts. The old proposal's all-four common cycle is not implemented and its claimed7-path average is superseded here.

`prepare_factor1` installs one unit R/S row at all23 existing affine sites on a fresh native single, retaining all original parameter objects and rebuilding the same empty native Adam groups. `factor1_allview` gives shared coordinates factual L and private coordinates `[L+.5 mean(A0,A1,A2,A3)]/1.5`, with A0 exactly reused from L. Its own derivative is unscaled: no division by4 for a one-body fit. This supplies the capable all-view factorized M1 source path.

`independent_factor1_step` requires four separately constructed/seeded factorized M1 bodies, four RNG stream objects and four independent native Adams. It rejects overlap in **all** native/factor parameter identities, gathers all4 bodies before commits, and gives each body the complete unscaled M1 update. It supports both ordinary and all-view I4 reference paths; it never renames one shared body as independent. Four individual own VALID selectors/checkpoints and final probability pooling must be supplied by the existing external owner, not a pooled I4 selector.

The native predictor has2,069,875 parameters. M1 has2,083,818; shared M4 has2,125,647; genuine factorized I4 has8,335,272. Native serving remains full-bank probability pooling. At2000 updates each7-path M4 arm costs14,000 route paths versus8,000 for existing shared4_own; M1 all-view costs8,000. These are work counts, not measured timing/peak-memory ratios. The persistent PubMed FP32 zero block costs39,434,000 bytes and is reused by all masks; the native stack still materializes each forward.

## Limits and next admission requirements

Static checks establish source syntax, selected bindings and the declared structural update/counter rules. They do not establish numerical gradient equality, provider compatibility, model competence, resource throughput or benefit. Native Session/factory binding, runtime qualification, complete paired seeds/splits, the full native horizon and selector, resource measurements and frozen competence/gain criteria remain external owner requirements. This packet contains no runner, release, selector, checkpoint loader or data loader. The reference **gradient/body** paths are implemented; their complete scientific fit/selection campaigns are not admitted.

Root supplied the closed PubMed means: native89.9374%, factorized M1single90.8168%, shared M4 90.9014%. Most old gain is already reproduced by M1. Those results were not recalculated or reopened here. Original paper scores remain unchanged; TEST remains closed; current CMCL/Wiki/IMDBquality is unavailable to this role. Missing-hop training can weaken factual member decisions and may be explained by generic multiview factorized learning or dropout averaging. A factual served gain needs the capable M1/control comparisons, acquired correct alternatives, retained competence and unused confirmation.

GRAND, FAGEL, AdaGCN, MIMO and selective learning credit are known ancestry. This source delta claims neither novelty nor benefit. It does not rescue the failed CORE screen, whose node-quarter masks, recomputed polynomial banks, decoder and all-block reconstruction credit differ.

Targeted filename searches found the earlier adaptive_sharing novelty literature folders and the saved proposal, but no partial missing-hop adapter/protocol to resume. Those folders were preserved. Only this new bounded folder was written.
