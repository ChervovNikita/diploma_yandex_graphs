# Retained supervised graph routes: two recent primary scopes

2026-10-11. **Bounded literature comparison, not a fit or novelty verdict.** Saved locators/conclusions were read first. Four relevance-prefix metadata searches were followed by **two new contiguous method passages**, one each from STEM-GNN and GMoPE. No other new method passage, full-paper read, code inspection, numerical import, remote science, dataset/checkpoint/prediction payload or current partial quality was used. Current GRU/LM4 contracts and the P/T architecture memo remain unchanged.

## Exact identities and reading boundary

**STEM-GNN:** Xiaoguang Guo et al., *Generalizing GNNs with Tokenized Mixture of Experts*, [arXiv:2602.09258v2](https://arxiv.org/abs/2602.09258v2), first posted 2026-02-09, version updated 2026-06-12. Read §4 only: MoE encoder, VQ interface and head regularization, Eqs 12–23. The arXiv record says accepted to KDD 2026; publisher-deposited Crossref confirms the exact title/DOI in *Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2*, published 2026-08-08, [DOI:10.1145/3770855.3817952](https://doi.org/10.1145/3770855.3817952). The inspected method is the pinned arXiv version; equality to the final publisher version was not audited.

**GMoPE:** Zhibin Wang, Zhixing Zhang, Shuqi Wang, Xuanting Xie and Zhao Kang, *GMoPE: A Prompt-Expert Mixture Framework for Graph Foundation Models*, [arXiv:2511.03251v1](https://arxiv.org/abs/2511.03251v1), posted 2025-11-05. Read the contiguous Methodology passage from Expert Prompt through Loss Function, Eqs 7–17. That record has no journal reference/DOI; a bounded Crossref title query returned different works. The pinned preprint is verified; later publication remains unverified, not disproved. Neither paper's experiments, appendices/proofs or author code were inspected. Theorem statements inside the selected passages were visible and are not endorsed.

## The relevant operations

STEM-GNN has node/layer-dependent `πᶫ(v)=softmax(gᶫ(hᶫ_v))`. A common neighborhood summary `zᶫ_v` enters globally shared expert operators, then Eq 14 sets

`hᶫ⁺¹_v=σ(sum_k πᶫ_k(v) fᶫ_k(zᶫ_v))`.

Thus expert alternatives are repeatedly mixed into **one continuing node state**. “Shared experts” means global operators reused across inputs; it does not establish one role's dense map tied across four separately continuing routes. Eqs 18/21–22 give one tokenized representation/task head and task loss plus a head Frobenius penalty. The selected scope does not show own expert class predictions/losses. Training uses Gumbel-Softmax relaxation; deployment uses deterministic softmax. VQ and head norm control are useful attributed interface mechanisms, not demonstrated remedies for our kernel failure or new fit ingredients.

GMoPE is the closer new **separate expert prediction-loss** overlap. Each prompted input enters its corresponding `E_m`; Eq 17 minimizes a gated sum of per-expert task losses, `sum_(i,m) gᶦ_m L_task(f_φ(Φ_(θ_m)))`, plus prompt orthogonality. Expert cores are indexed `θ_m`; the inspected scope does not impose common graph-transform parameters. Separate full expert embeddings survive until terminal aggregation, `h_final=sum_m ω_m h_m`, with entropy-derived confidence weights. This is stronger prior than a single fused CE, although its gated branch objective differs from four unweighted complete own CEs and fixed probability serving.

GMoPE's router scores **batch-average expert loss** and selects the largest printed Rawscore, then uses soft or uniform top-K weights. It is not an inspected sequence of node controllers recomputed from each private state before successive P/T blocks. Positive-loss/highest-score orientation is retained as written; its intended sign and runtime interface are unqualified. Prompt-only transfer freezes the expert cores and adapts prompts/task components. It does not establish tied dense-core ensemble acquisition.

## Collision boundary and useful references

| Work | Continuing state / supervision | Missing from an exact collision in the inspected scope |
| --- | --- | --- |
| STEM-GNN, new 2026 scope | One mixed state per node at every layer; one task head | Several separately supervised continuing states and route-tied dense transformations |
| GMoPE, new 2025 scope | Separate expert graph predictions with gated own task losses; terminal embedding fusion | Tied graph cores and repeated current-private-state operation/filter routing |
| GNNMoE, reused published 2025 source conclusion | Repeated current-state PP/PT/TP/TT mixture; one continuing state/classifier | Several own-supervised continuations; official T maps are separate across experts |

GNNMoE remains the closest **P/T operation** collision. STEM-GNN adds a verified 2026 published ordinary conditional-encoder reference; GMoPE establishes recent graph-expert own-loss/prompt/fusion ancestry. Own branch prediction losses, multiple expert graph trajectories and conditional routing should not individually be presented as new. Saved MoSE/SAGMM/HGEN/LHGEL/shallow-ensemble conclusions remain relevant, with their original scope and unresolved interfaces preserved; none was reread as a new primary method.

**Conclusion:** these two passages do not establish the complete conjunction of tied graph transformations, several own-supervised continuing states, and repeated node-conditioned operation choices. This is a scoped observation, not absence proof or novelty clearance. Capable mixed-continuation single/equipped I4 and the matched middle-state mean remain the discriminating controls already sealed. No source repair, baseline implementation, fit nomination or acceptance verdict follows.

## Accounting

Three arXiv metadata prefixes returned 15/168, 2/2 and 15/18 entries; one OpenAlex prefix returned 15/2,300. These are 47 returned metadata entries before deduplication, not paper reads or exhaustive coverage. Two versioned HTML bodies were fetched for custody; only the two declared method passages were semantically inspected (45 paragraphs, 11 method headings, 23 distinct display equations/26 MathML fragments). Table-hosted equations were explicitly recovered within those same scopes. Incidental published numerical claims in discovery abstracts were visible and unused. Raw responses, identities, selected passages and exact scope hashes are saved. Zero old primary rereads, scientific execution or canonical/index edits occurred.
