# Prospective protocol amendment

## Immutable parent

Parent correction driver: `continuous_method_gap_search_v1/round16_derived_roles_amendment_v2/prototype/correction_screen_driver.py`, SHA256 `376e8b779d8d71cf21785a62b8543ecadc9c09c9db8174dc230b55b8d0dbd253`. Parent aligned correction and CF-GNN adapter are copied byte-for-byte. The new copied driver binds every sibling Python file. Parent files, role protocol and existing scientific records are not edited.

Both parent-prepared roles with that exact driver hash and new preparation with the identical derived_roles_v2 transformation are admitted. Twenty percent teacher training and the train/validation/A/B/D/final-pool partition stay fixed. Squirrel uses the committed label-blind official-mask thinning rule; Photo retains full-node uniform allocation. This is a new role protocol relative to native papers (Squirrel roughly50% train; Photo60% train), not an author-score reproduction.

## Native pair

PolyFormer-Mono, `air029/PolyFormer@d390f39e88d0eaac80318fdc7704bd3bf3cf8b13`: filtered Squirrel N2223/F2089/C5; hidden256, order12/13 tokens, two native blocks, four heads, FFN128, q1.4, multiplier1, dropout.3 and dprate.8. Adam ordinary lr1e-4/wd0; every parameter whose native name contains `attnmodule` gets lr1e-3/wd1e-7. Cap2000, family patience250. Raw verified features are not normalized; author monomial preprocessing retains gcn_norm/self loops, SciPy COO orientation and FP32 sparse conversion. No author filename-only pickle cache is reused. Tokens are recomputed and identity-bound to raw input manifest, source/implementation hashes, preprocessing, order and runtime.

Polynormer-r, `cornell-zhang/Polynormer@fc8c276c9c5dfbd616d83f65338a3392188a5e08`: Photo N7650/F745/C8; hidden64 per head times eight=512; seven GAT local layers and two native global layers; Adam lr1e-3/wd5e-5; input dropout.2, local/global dropout.7, learned beta−1, shared q/k and no preLN. Verified provider features receive actual PyG NormalizeFeatures once. Edges are undirected, self loops removed then added once; native GAT has add_self_loops=False.

Photo follows the released **200 local plus1000 global=1200 actual optimizer updates**, with no early stopping. Paper Table5 says warmup200/total1000; the released run/main sums1200. The local selected model and Adam state are both restored at the transition. The global selector starts fresh and final selection admits global checkpoints only. Native `_global` is saved explicitly because it is absent from state_dict. The NLL selector/reset/global-only rule is an explicit prospective selection adaptation.

## Families and selection

The native single and each independent model use native full widths. GNNM shares all native interior parameters and boundary weights; private R/S/B are added only at input and output projectors. Stem R starts independent Rademacher, S starts one, and output R/S start one. Each private B copies the native bias (zero if absent). Copying native bias differs from the older GNNM zero-B initialization and is declared. Dormant native boundary owners become Identity and are not optimized. Native reset occurs before wrapping, never afterward. All member-dependent GAT scores and global node reductions run separately through four complete trajectories; no hidden-state averaging or shared attention-score shortcut is permitted.

Four configurations per family/graph: native settings; all learning rates times.5; all learning rates times2; native rates with ordinary/local/global dropout lowered.2, floored0. PolyFormer dprate and Photo input dropout never change. Seeds17/29/43 and splits0/1/2 are paired. Single and shared common native initialization use the same seed; independent member m uses seed+100003m. Training RNG is reset to seed+70000 after construction. Loss is arithmetic mean member cross entropy. Independent models have one synchronized family checkpoint and one family-level selector/patience, with no per-member favorable choice.

Every checkpoint uses mean raw-logit predictor-validation NLL, earliest exact tie. One configuration per family/graph minimizes the arithmetic mean of its three selected seed NLLs, listed order breaks exact ties. All12 cells must complete before family selection; all72 paired cells must close before correction. No extra/adaptive configuration, shortened final schedule, task replacement, favorable retry or final-label selection is allowed. Train/predictor-validation labels alone enter teacher fit/tuning. A/B/D are opened only by the copied correction fit. Final pool is opened only by report after immutable score freezes.

## Pooling contract and fixed secondary

All primary prediction, selection, correction inputs, APS scores, serving and CPU64 report references use softmax(mean raw logits), first-index argmax. This preserves actual GNNM/v2 semantics and deliberately differs from the literature draft's proposed mean-probability primary. No erroneous mean-probability canonical claim from the older vectorized source packet is inherited. Mean probabilities is only a fixed secondary from the same saved logits; it never changes checkpoints, configs or correction fits and has no claim of improving original scores.

## Attributed native extraction

`SOURCE_SLICES.json` binds exact retained class/function bodies. Only imports are reduced for PolyFormer; import-time seed mutation, unused dataset/utils/DGL imports and unused utilities are removed. PolyFormer class bodies, block bodies, monomial and sparse-conversion bodies are unchanged. Polynormer model.py is copied in full. `AUTHOR_SOURCE_BINDINGS.json` retains author commit/Git blob/SHA identities and local raw source copies. This is source attribution, not runtime qualification.
