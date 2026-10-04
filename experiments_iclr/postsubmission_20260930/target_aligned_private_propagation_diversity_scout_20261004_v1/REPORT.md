# Target-aligned private propagation allocation: bounded follow-on proposal

## Conclusion

A direct graph-specific prior now covers learning diverse neighbour-dependence patterns: RGMoE separates estimated neighbour-to-representation MI vectors and uses a robustness-aware inference router. Node-MoE covers different learned graph filters, filter initialization/smoothing and node-specific selection under mixed structural patterns. Existing DICE, FoRDE, sMCL, boosting, graph-filter, embedding/rotation and initialization notes already cover the broader ingredients. **No new diversity principle or novelty clearance is established.**

One attributable adaptation remains testable: continue learning **only explicit private propagation filters**, using bounded target-loss improvement over frozen common-filter references to allocate TRAIN supervision, while the shared task backbone and all reference functions stay frozen. The ordinary supervised and generic task-allocation controls use the same architecture, warm stage, frozen blocks and unchanged uniform serving. This is a source-specific hypothesis, not an admitted implementation, launch or manuscript verdict.

The most important negative result is algebraic: **subtracting a member-identical common reference from each member loss does not change softmax, winner or balanced assignments.** The unqualified private-minus-common idea is rejected on that basis. A live or weak reference introduces further failure modes.

## What was reused

Index v47 was the initial memory. DICE's conditional-on-label redundancy, FoRDE's true-label log-probability input gradients, sMCL's minimum-member-loss assignment, AdaGCN's sequential error/hop specialization, BernNet/C&S/TabM initialization notes, and existing contrastive/rotation assessments were reused without reopening their primaries. Predictor-preserving embedding separation and auxiliary-specialization-with-unchanged-query-score counterexamples were retained from the sealed endpoint packet.

The corrected architecture note matters: original tied-weight GNNM retains private member trajectories; the later shared-mean-message extension is the model with recurrent member-state compression. Weight sharing alone does not imply state mixing or universal graph blindness. A feature-diagonal factor also commutes with a fixed linear graph operator, \(P(XD_m)=(PX)D_m\); it must not be relabeled a distinct spectral filter without an actual propagation mechanism. No present DDI/PENCIL source was inspected here.

## Primary scope evidence and a corrected prior identity

| Source and exact scope | Operation verified in scope | Relation and limits |
|---|---|---|
| **Han et al., Node-wise Filtering in Graph Neural Networks: A Mixture of Experts Approach**, arXiv:2406.03464v1. Preliminary notation on PDF p.3; §2.2 Definition/Theorem 1 statement on p.4; §3.1–3.4 on pp.4–7, before §4. | Independent GNN experts with learned filters; distinct low-pass/constant/high-pass initialization; spectral smoothing Eq.(2); context gate using \([X,|AX-X|,|A^2X-X|]\); weighted representation fusion Eq.(1), optionally top-k. The stated mixed-CSBM analysis connects filter sign with class separability under linear/Gaussian/two-class/equal-degree/asymptotic assumptions. | Graph-filter specialization and structurally conditioned selection are prior. Its inference gate and expert models differ from a uniform served bank on a tied backbone. Appendix proof/code/results were not audited; no theorem transferred. |
| **Feng, Ma and Dai, Backdoor or Manipulation? Graph Mixture of Experts Can Defend Against Various Graph Adversarial Attacks**, arXiv:2510.15333v1. §3.1–3.5 on PDF pp.3–5, Eqs.(2)–(15), before §4. | RGMoE forms a per-expert neighbour vector from estimated \(I(h_u^k;\tilde h_v^k)\), then penalizes pairwise cosine similarity above a margin, Eq.(8). Training includes task/load-balance loss, MI estimation and this penalty. A separate GNN router is refined using expert disagreement and soft labels. | Direct graph-specific diversity prior. The proxy concerns representations; it is not conditional-on-label target information, a target-margin derivative, or causal evidence of a neighbour's contribution. Its learned router changes serving. Per-edge estimator mechanics, appendix algorithm/source and efficacy were not certified. |

Primary URLs: <https://arxiv.org/pdf/2406.03464v1> and <https://arxiv.org/pdf/2510.15333v1>. RGMoE is one genuinely new paper. A later phase-report identity check found that Node-MoE §§3.1–3.4 had already been read, despite its absence from raw index v47 identity text. Those method passages were reexposed here and are disclosed as repeated exposure, not a new paper. Its §2.2 model/theorem statement is a qualified scope extension; the earlier reports record only §§3.1–3.4. No replacement paper was added. None is a full-paper certification. Public experiment prose adjacent to boundaries was incidentally displayed and not adopted. RGMoE's running conference/author headers contain template placeholders; they are not publication metadata.

Both methods reinforce the existing nullspace limit. Different neighbour-dependent representations or different graph filters do not ensure different useful target predictions. Even \(PX\) and \(-PX\) cancel under a shared linear readout and uniform mean. Node-specific selection can avoid that particular cancellation; the present uniform serving contract cannot silently inherit its gate.

## Loopholes in the initially suggested loss advantage

Let \(\ell_{im}\) be target loss for query/node \(i\), and let \(c_{im}\) be loss of a common-filter reference. Define \(\Delta_{im}=c_{im}-\ell_{im}\).

1. **Identical reference cancels.** If \(c_{im}=c_i\), then \(\operatorname{softmax}_m(\Delta_{im}/T)=\operatorname{softmax}_m(-\ell_{im}/T)\), and the winner is the lowest-loss member. More generally, for any assignment matrix with row sums one,
   \(\sum_{im}A_{im}\Delta_{im}=\sum_i c_i-\sum_{im}A_{im}\ell_{im}\).
   Column-balance constraints or an entropy term do not change the constant. Thus even balanced advantage allocation is generic loss allocation in this case. Freezing the reference does not remove this equivalence.
2. **A live reference can be gamed.** Differentiating an advantage objective can worsen the reference to increase apparent gain. Detaching the assignment/reference at one update is insufficient if their parameters continue changing between refreshes. A live member mean is therefore not the proposed reference.
3. **Weak fixed references can misallocate.** If one reference loss is 10 and its member's current loss is 2, its gain is 8; another reference/current pair 1/0.1 has gain 0.9. Raw advantage prefers the worse current predictor. Freezing fixes manipulation, not this comparison.
4. **Loss allocation is not a diversity or pool guarantee.** It may concentrate on one member, or favor TRAIN progress that does not generalize. Different loss references may simply encode existing member difficulty, rather than a uniquely graph-specific mechanism. The generic-allocation contrast is mandatory.

## One concrete adaptation, with fixed boundaries

**Frozen-reference, competence-bounded private-filter allocation.** This is the single proposal; the following choices are part of its definition.

**Architecture and warm stage.** Use the same member bank and mean raw-logit serving throughout. Identify an explicit private propagation block \(\beta_m\), with matched shape across members, and shared/root/head parameters \((\theta,\psi_m)\). If the current model lacks such a block, the minimal candidate is a degree-two polynomial propagation \(F_{\beta_m}(P)=\sum_{k=0}^2\beta_{mk}P^k\) at one fixed graph-message site. Every arm receives exactly that same block; this is established learned-filter machinery, with its full sparse cost charged. Do not claim an unchanged current-source architecture or infer a spectral filter from diagonal feature scales.

First train ordinary member target losses with one common propagation filter \(F_0\), retaining the allowed private root/head functions. At a prospectively fixed warm-stage boundary, freeze \(\theta^*,\psi_m^*,F_0\). Copy this identical state into all arms. Only the private \(\beta_m\) blocks learn during the continuation, with identical paired perturbations/optimizer states. Freezing the shared backbone is a deliberate constraint and may limit utility; it supplies no generalization theorem.

For each TRAIN example under the exact same graph/mask/support, compute frozen reference logits
\(z_{im}^0=f_m(G;\theta^*,\psi_m^*,F_0)\), their uniform mean \(\bar z_i^0\), and losses \(c_{im}=\ell(z_{im}^0,y_i)\), \(c_i^{\rm pool}=\ell(\bar z_i^0,y_i)\). The reference includes the frozen non-filter parameters; it is never replaced by a live member mean or refreshed model. Stochastic views/masks require the frozen function to be reevaluated on the identical context, not a stale loss from another mask.

At each epoch refresh, evaluate current member target losses without dropout, detach all scores, and define

\[
g_{im}^{P}=\big[\min(c_{im},c_i^{\rm pool})-\ell_{im}\big]_+.
\]

The pool cap ensures that worsening a member reference beyond the competent common pool cannot raise its score. A member earns allocation gain only by beating both its own frozen common-filter reference and the frozen uniform pool on that target. No target label is assigned a counterfactual causal interpretation.

Set \(A_{im}=g_{im}/\sum_jg_{ij}\) when the gain sum exceeds \(10^{-12}\); otherwise use \(A_{im}=1/M\). Keep zero-gain/fallback rows. During the epoch optimize only \(\beta_m\), with ordinary true derivatives of

\[
L_{\rm cont}=\frac1N\sum_{im}\left[\frac{1-\lambda}{M}+\lambda A_{im}\right]\ell_{im},\qquad \lambda=\tfrac12.
\]

Assignments are detached and fixed until the next refresh. All shared/root/head/reference blocks stay frozen. This is an alternating frozen-assignment task curriculum, not gradient descent on a permanently fixed joint assignment objective. Every member retains the ordinary supervision floor \(1/(2M)\). No member repulsion, custom graph adjoint, inference router or auxiliary serving head is added. The allocation can still collapse; balance/diversity is not guaranteed.

The gain clipping/normalization differs from merely subtracting a reference inside softmax. It remains an attributable progress-allocation adaptation. If the member reference functions all equal the frozen pool function, the repaired rule also equals the generic control below. That is an exact stop condition for a claimed filter-specific allocation difference.

**Remaining counterexample.** A qualifying member gain need not improve the uniform pool. For a binary target \(y=1\), frozen logits \((0,10)\) give pool logit 5. Current logits \((6,0)\) let member 1 beat both its own reference and the frozen pool, but the current uniform logit is 3 and its loss is worse. The ordinary competence floor does not logically rule out such a state under limited capacity/optimization. Neither the allocation score nor a TRAIN loss gain can replace the served prediction comparison.

## Closest-prior delta

- **sMCL/AdaGCN and related task allocation:** assignments, hard-example specialization and classifier competence are prior. Here the allocated score is bounded progress attributable to a private propagation block while the other predictor blocks and common-filter references are frozen. This is an adaptation, not a new assignment principle.
- **Node-MoE/BernNet and graph-filter initialization:** private learned filters, low/high-pass choices and filter smoothing are prior. The proposed continuation trains those filters under target progress allocation and retains uniform serving rather than node-specific gates.
- **RGMoE/DICE/FoRDE:** graph-neighbour representation separation, conditional redundancy and target sensitivity diversity are prior. This proposal uses observed target loss progress rather than representation MI, hidden distances or gradient-angle repulsion.
- **Earlier initialization proposal:** this is repeated supervised continuation under fixed predictor references, not a one-time graph-band cotangent pulse. It does not inherit earlier initialization claims or outcomes.

No exact duplicate of this complete amended schedule is established by these two scopes. No absence or novelty clearance is inferred. A positive result would establish a useful, specifically controlled adaptation in one recipe.

## One falsifiable paired experiment, proposed only

Use **ogbn-arxiv's official node-classification split** for one representative development comparison, avoiding the ongoing DDI/PENCIL task. Keep TEST unopened. Freeze the backbone, warm/continuation update counts, bank size, propagation site/order, optimizer/LR, dropout, graph visibility, validation checkpoint rule and all paid costs before fitting. Use five paired seeds, one \(\lambda=1/2\), the fixed gain floor, and no method-specific HPO. No fits are authorized by this packet.

Three arms differ only in continuation assignment:

1. **U — ordinary supervised private filters:** \(A_{im}=1/M\), giving ordinary mean member target loss exactly.
2. **G — matched generic target-progress allocation:** \(g_{im}^{G}=[c_i^{\rm pool}-\ell_{im}]_+\), with the same gain normalization, fallback and competence floor.
3. **P — private/common-filter allocation:** \(g_{im}^{P}=[\min(c_{im},c_i^{\rm pool})-\ell_{im}]_+\), as defined above.

G deliberately matches the bounded-gain and frozen-pool machinery; comparing only to minimum-loss softmax would confound the filter reference with clipping/difficulty weighting. All arms pay the same warm stage and assignment-refresh evaluation budget; report their actual complete sparse forward/backward work, wall time and peak memory. Cached frozen reference values are valid only for identical deterministic contexts. Uniform serving and source competent baselines remain required; this narrow study establishes no broad efficiency or bank-necessity claim.

**Primary falsifier:** P must improve paired VALID served accuracy over both G and U; report all five paired differences, with paired NLL, member competence and cost as fixed secondary results. The filter-reference interpretation fails if P merely differs from U while G matches it, if gains appear only in allocation/embedding statistics, or if member/reference functions make P and G identical. If references are member-identical by construction, stop before a redundant fit. No favourable subset, added filter order, balance rule or changed gain threshold after outcome inspection. Any subsequent TEST confirmation needs a separately frozen independent protocol; this report does not open it.

## Provenance and action boundary

The corrected primary accounting is one genuinely new paper (RGMoE), one qualified Node-MoE §2.2 scope extension and one disclosed repeat of Node-MoE §§3.1–3.4. The initial index-only unread classification was insufficient; `PHASE_IDENTITY_CHECK.json` preserves the correction and old report hashes. The public discovery query and Crossref fuzzy query and both pinned PDF downloads succeeded; zero remote failures occurred. Truncated local locators were recovered and recorded as search limitations, not absence evidence. Exact scopes, incidental exposure, source/extraction/render hashes and reused note boundaries are saved alongside this report. No earlier cached primary file was opened; the repeated Node-MoE semantic method exposure came from the new retrieval of the same version and is not counted as a new method.

No scientific source changed; no tensor/model/dataset/heldout payload, scientific server or full HPO was accessed or executed. No extra agent, manuscript edit, current DDI/PENCIL review duplication, index edit or execution authorization occurred.
