# Graph-specific shared-factor learning: bounded scout

**No new distinct method is recommended.** One candidate was assessed: train aligned members on the joint labels of adjacent source-training nodes. It supplies a precise operation to examine, but neither natural pooling version establishes the required increment. Do not open another training grid for it on this evidence.

This is a methodological source/design assessment, not an empirical impossibility claim. A graph-pair auxiliary objective might have utility. No exact published duplicate of its complete factor/backbone recipe was verified, and no novelty follows from that retrieval limit.

## Memory and preserved dispositions

The saved literature index v10 and existing proposal reports were consulted before retrieval. Their descriptors and read scopes are in `REUSED_SOURCES.json`. The coordinate permutation/factor screen remains **STAGE1_NO_GO**, with all 54 competent cells retained. Ordinary embedding/contrastive repulsion, early topology routers and type-indexed factors retain their saved no-go conclusions. The exact equivalence between eligible factor maps and equally placed diagonal activation paths remains an attribution constraint.

The active Round15/R17 operation remains one-time graph-band residual/VJP initialization followed by ordinary member-CE training. Repeating that operation, adding a type index or changing a spectral gate is not an independently substantiated learning principle. The separate exact-sampling residual-count proposal addresses a different task and makes no factor-specific claim. The audited Photo precision diagnostic and its failed original gate are untouched.

One new accessible primary was read: **GMNN: Graph Markov Neural Networks**, [arXiv:1905.06214v3](https://arxiv.org/html/1905.06214v3), dated 23 July 2020. Scope was the abstract and §§3.1–4.4, Eqs.1–14 and Algorithm1. The versioned primary bytes, selected passages, retrieval record and structured conclusion are saved. No result/proof/pixel/code review or full-paper review is claimed. The already retained GNCL Eq.5 was revisited only for the new pair-objective question; it was not fetched again. New primary count: **1 of at most 2**.

## The one assessed learning operation

Let four aligned members of the existing shared-factor modern backbone output logits \(z_{m,v}\), with \(p_{m,v}=\operatorname{softmax}(z_{m,v})\). Let \(T\) be source-training nodes and \(E_T\) the original undirected edges whose **both** endpoints lie in \(T\). No validation/final label is a target.

During ordinary supervised updates, add

\[
L_{\mathrm{pair}}=-\frac{1}{|E_T|}\sum_{(u,v)\in E_T}
\log\left[\frac14\sum_{m=1}^4
p_{m,u}(y_u)p_{m,v}(y_v)\right]
\]

to the existing uniform mean member node CE, with one fixed weight and no new inference router. An empty \(E_T\) makes the operation unavailable; it cannot be repaired by reading other-role labels. This changes the supervised learning operation and targets cross-node member alignment rather than hidden distance.

For an edge, its ordinary mixture responsibility is

\[
\rho_{m,uv}=
\frac{p_{m,u}(y_u)p_{m,v}(y_v)}{
\sum_j p_{j,u}(y_u)p_{j,v}(y_v)}.
\]

The edge-loss logit gradient at \(u\) is
\(\rho_{m,uv}(p_{m,u}-\operatorname{onehot}(y_u))\), and similarly at \(v\). Thus it pays a member when it supports both observed endpoint labels. These are standard finite-mixture responsibility gradients, not a new gradient-allocation rule.

The corresponding pair distribution is an outer-product mixture,
\(Q_{uv}=\tfrac14\sum_m p_{m,u}p_{m,v}^{\mathsf T}\). Its difference from independent probability-mean marginals is exactly aligned member covariance. It is also a pair marginal of the already familiar coherent global-component model
\(Q(y_V)=\tfrac14\sum_m\prod_v p_{m,v}(y_v)\).

## Decisive pooling boundary

The current point predictor is
\(P_v=\operatorname{softmax}(\tfrac14\sum_m z_{m,v})\).

If additive member pair logits are combined with that same raw-logit rule, then

\[
\operatorname{softmax}_{a,b}\left[\frac14\sum_m
(z_{m,u}(a)+z_{m,v}(b))\right]
=P_u(a)P_v(b).
\]

Therefore its edge loss is exactly

\[
\frac1{|E_T|}\sum_{v\in T}d_{E_T}(v)[-\log P_v(y_v)].
\]

The proposed label dependency disappears: this is degree-weighted existing pooled node CE. The identity holds for **every** output of PolyFormer, Polynormer or another backbone. No graph/model run is needed to establish it.

Using the probability outer-product mixture avoids that factorization, but its marginal is \(\bar p_v=\tfrac14\sum_m p_{m,v}\), generally different from \(P_v\). It can be declared as an auxiliary objective and separate secondary joint law; it cannot silently inherit the current raw-logit predictor's marginal semantics or replace its frozen primary pool.

## Exact prior overlap and limits

| Evidence | Supported overlap | Remaining boundary |
|---|---|---|
| GNCL, retained arXiv:2011.02952v2, §4.1 Eq.5 | Set each normalized ensemble output to the \(C^2\)-vector \(h_m(u,v)=p_{m,u}\otimes p_{m,v}\) and its label to \((y_u,y_v)\). The mixture pair term is exactly established ensemble CE on these examples. | The complete uniform-node-CE plus pair-term graph/factor recipe is not claimed published by GNCL. Its mean member **pair** CE degree-weights endpoint member losses, whereas the current node auxiliary CE is uniform. |
| GMNN v3, §§3–4 | Explicit edge label dependencies, neural local label conditionals and alternating relational inference/learning are established graph operations. | Native CRF/pseudolikelihood and mean-field variational EM are different from this uniform global product mixture. GMNN is not an exact duplicate. |
| Saved Round09 graphical-model inference | Aligned global component identity and normalized product mixtures already are an existing candidate ingredient. | Known-target importance optimization is a different task/objective. |
| Saved Round5 covariance correction and Round7 field distillation | Aligned cross-node covariance information and the insufficiency of pair moments already have explicit analysis. | Post-fit correction or teacher-field fitting is not identical to this training term. |

This is not a new mixture loss, graph label-dependency principle or factor-specific function class. The remaining delta is using one established structured supervision term in a particular restricted ensemble. No concrete failure of the capable alternatives is documented that makes this composition the next warranted method lane.

The objective also does not require predictive diversity: if unrestricted members fit every training label with probability one, identical members attain zero member CE and zero pair loss. Constraints and generalization might make a noncollapsed solution useful, but the proposed objective supplies no diversity or uncertainty-quality certificate. A low edge-pair NLL likewise cannot identify the complete label law.

## Representative falsifier, conditional on later utility reopening

**No run is recommended or admitted now.** The raw-logit pair variant is already rejected by the exact identity above.

If this composition is separately reopened strictly for utility, use two complete paired modern mechanism cases—PolyFormer-Mono/Squirrel17 and Polynormer-r/Photo17—at native cfg0, with identical qualified starting factors, optimizer states, schedule and primary raw-logit selection. The Photo original failure remains a failure; any future implementation requires its own valid qualification.

Use four fixed objectives, without a sweep:

1. Existing member node CE.
2. The same CE plus true-edge probability-mixture pair CE.
3. The same CE plus independent probability-mean marginal pair CE on those exact edges. This matches endpoint degree weighting and exposes whether member covariance supplies the increment.
4. The same CE plus mixture pair CE on a prospectively fixed, label-blind, degree-preserving shuffled source-edge set. The backbone still receives the unchanged real graph; only the auxiliary supervision pairing changes.

Normalize losses consistently and freeze one auxiliary weight before outcomes. Do not redraw shuffles or select a weight, seed or graph by results. Root would freeze a meaningful minimum benefit and whole resource cap before any separate run.

The true-edge objective must improve the fixed primary mean-raw-logit **node** NLL over every matched control in both complete cases at charged total cost. A gain confined to auxiliary pair NLL, member spread or one favorable case does not support the current node-prediction claim. Retain all failed fits, qualified full member graph/dense work, pair construction and scoring, selection, peak memory and preparation costs. One seed per case is a mechanism falsifier, not confirmation or evidence across seeds/graph populations.

This conditional design separates edge-pair supervision, degree weighting, probability pooling and graph pairing. It is recorded to prevent a future weak comparison from being mistaken for a distinct method result; it does not justify executing an unpromoted hypothesis.

## Disposition and scope

**Zero distinct extensions promoted; one precisely defined candidate rejected for new-method promotion.** Useful pair-learning adaptation is not ruled out, and exhaustive novelty absence is not claimed. Stop this scout rather than manufacture novelty or another benchmark/grid.

Only public source retrieval, saved text/JSON inspection, symbolic derivation and artifact writing occurred. No model/data/GPU/SSH/scientific execution, supplied-code import, sealed/current-source modification, manuscript edit or original-score change occurred. No paper acceptance verdict is given.
