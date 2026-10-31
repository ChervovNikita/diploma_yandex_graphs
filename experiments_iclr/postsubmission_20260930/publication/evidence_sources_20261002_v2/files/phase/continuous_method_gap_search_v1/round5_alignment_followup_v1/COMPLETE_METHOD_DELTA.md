# Complete method delta and graph-specific utility target

## Input/output contract

Input: one fixed undirected complete graph X/A, source-fitted K4 teacher logits z[K,N,C], fixed label-independent APS uniforms, and source A/B/D labels outside final pool. Output: one frozen real-valued class score table, then independently calibrated class sets. Original point probabilities softmax(mean raw logits) are retained.

Member probabilities p_k are centered by their probability mean pbar. The edge statistic a_uv=mean_k[d_k(u)^T d_k(v)] and normalized q_uv=a_uv/sqrt((b_u+1e-8)(b_v+1e-8)) are unchanged by a global member permutation. Independent nodewise permutations preserve all nodewise member multisets and raw-logit pools while changing correspondence. This establishes input distinction only. Finite K4 private factors provide an empirical function statistic, not posterior consistency or label-compatibility guarantees.

The gate receives pstar/pbar, full lexicographically sorted member probability vectors and log(1+degree) at both endpoints. The aligned arm adds a/q; the marginal arm adds zeros; the shuffled arm adds statistics computed after fixed independent nodewise member permutations. A pooled-only arm zeros all multiset and a/q channels. Every arm uses the same symmetric two-hidden-layer width32 MLP and signed score-difference operator. Neither gate gets raw teacher hidden states or extra labels.

## Existing parts

- Coherent global graph/model samples: Bayesian GCNN and Bayesian graph CP; empirical member joint information: BatchBALD/k-BALD.
- Pairwise score aggregation: DAPS/SNAPS and localized graph CP.
- Signed heterophily/edge-compatible score diffusion: HeAD-CP.
- Learned conformal efficiency correction: CF-GNN and entropy correction.
- Learned graph filtering plus conformal training objective: SparGCP.
- Graph uncertainty influence: GPN and earlier Bayesian graph structure methods.
- Independent final calibration and finite-population marginal coverage: existing graph conformal theory.

**Remaining complete-method specification:** centered same-member cross-node probability covariance/correlation as auxiliary inputs to a source-trained signed correction, with all nodewise marginal information supplied to a capable control and separate final random calibration. No inspected complete method specifies that exact pipeline. Non-exhaustive search and familiar ingredients prevent an originality conclusion.

## What would make the graph-specific gap defensible

A practical graph correction must decide which neighboring class scores to attract or repel. Same-teacher pooled/marginal controls can already learn pooled compatibility, graph degree and private spread. The aligned feature matters only if its source-learned edge relation predicts a useful *held-out correction* beyond those inputs. A gain surviving an equally trained nodewise alignment-shuffled control is the relevant correspondence interaction. Comparing only with uniform diffusion, entropy or a single variance would not establish that interaction.

The initial outcome is set efficiency at independently calibrated marginal coverage. Changes in point accuracy are not its target. Improvement concentrated in poor marginal coverage, empty sets, one selected seed, or disappearing under a capable correction rejects promotion. A pass still needs model-family/cost attribution: single, shared-trunk/private-head, private LoRA and untied ensembles at complete matched deployment cost. It would not identify weight sharing as the cause or prove GNNM-specific superiority.

## Scalar mechanism limitation and falsifier

The exact scalar identity is a_uv=mean_k p_k(u)^T p_k(v)−pbar(u)^T pbar(v): matched-member soft compatibility in excess of pooled compatibility. It is a trace contraction of the cross-node class covariance, retaining same-class sums but losing individual-class and off-diagonal relationships. Different class covariance structures may have identical a/q and call for different score corrections. The current scalar signed gate remains fixed; no matrix/variant grid is added. GPN's class-specific evidence and HeAD's existing scalar pooled compatibility are close conceptual limits, not current utility evidence.

Independent nodewise member shuffling preserves canonical pools and every endpoint member multiset, but changes correspondence. A competent separately source-fitted shuffled arm matching or beating aligned correction at valid independent final calibration is a concrete falsifier. The same holds for a capable marginal control. Complete-method novelty is unestablished; the claimed delta is only this exact conventional-statistic input in the fixed pipeline.
