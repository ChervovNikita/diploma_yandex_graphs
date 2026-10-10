# Graph-supported CMCL responsibility: prior collision and one inactive question

**Decision:** ordinary graph coherence supplies a standard structured assignment prior. It does not establish a new training principle. The exact shared/private recipe is not proved duplicated by one complete published method, but combining its known ingredients does not clear methodological novelty. No implementation, solver, experiment grid or fit is admitted.

## Exact reduction for the current ownership cardinality

Let `c_i,m = CE_i,m − beta KL(U || P_i,m)` be the stopped CMCL ownership cost. CMCL chooses the K members with the lowest costs. For M=4 and K=3, write `v_i = 1 − e_i`, where e_i is the one-hot identity of the omitted member. Then

`sum_m v_i,m c_i,m = sum_m c_i,m − c_i,omitted`.

The first term is constant during ownership selection. A graph penalty `sum_(i,j) w_i,j ||v_i − v_j||²` becomes `2 sum_(i,j) w_i,j 1[omitted_i != omitted_j]`. Therefore this extension is exactly Potts MAP assignment with CMCL unaries. For general fixed K, the states are member subsets and the penalty is their Hamming-distance metric: standard metric labeling. These are algebraic identifications, not a solver or predictive theorem.

Keeping shared weights on complete own supervision while private factors receive the additional stopped assignment loss changes gradient ownership. With a live common body, the combined block update generally is not the gradient of one scalar structured loss. It must not be called joint descent on the Potts objective. Every member can still execute at inference with the existing probability mean; this distinction does not make the assignment operation new.

## Closest sources, consulted in the requested order

Saved sMCL/CMCL already supplies current-loss winner/top-K ownership and confidence suppression. The October 3 local structure-specialization scout already holds topology-conditioned private risk with unchanged shared supervision. The October 5 local learnability operator already combines complete own supervision, balanced/entropic graph responsibilities and dense probability serving. Its finite-response cost, live outer credit and recommit differ from current stopped CMCL; this prospective work must not rebrand that local composition.

New bounded primary checks resolve four related families:

- **Structured MCL, NeurIPS 2012:** binary lowest-hinge flags, alternating assignment/model fitting and top-K overlap for complete structured outputs. Internal output structure is different from graph-coupled ownership between node examples; its objective is hindsight/oracle quality.
- **Flitti and Collet, 2007, DOI10.1007/s11760-007-0010-y:** a Markov quadtree couples mixture responsibilities; posterior marginals then weight component estimation. Spatially coherent responsibility learning is prior. It is an unsupervised PPCA segmentation model, and exact tree inference does not apply to arbitrary graph loops.
- **GraphRevoker, 2403.07353v1 / WWW 2024 DOI10.1145/3589335.3651265:** learned graph/label-aware node partitioning, isolated submodel fitting and dense attentive fusion. It changes graph support, unlike credit-only training. Printed loss-sign/order ambiguities remain unresolved; no native code or baseline quality is adopted.
- **Xue et al., JMLR 2007:** latent task grouping under a Dirichlet-process prior, variational categorical responsibilities and shared logistic atoms. Compatible-task grouping and joint responsibility fitting are prior; the tasks are datasets rather than graph-node/private-member credits.

Saved Node-MoE, MoE-NP and MoSE already provide structural/filter context routing. Their serving gates differ, but using graph context to select specialists cannot be claimed here as a new general idea. All reading is bounded; zero full papers/repositories or proofs are certified.

## Homophily, heterophily and error reinforcement

Adjacent nodes do not necessarily need the same expert. On a homophilic graph, smooth ownership can follow label communities rather than a useful correction mechanism. On a heterophilic graph, adjacent different labels may either need different corrections or share the same structural rule. Heterophily alone does not justify switching ownership signs. Requiring equal or unequal member identities across every edge imposes an unverified task prior.

A simple counterexample shows the tradeoff. For two linked training nodes, let the ownership costs for two routes be `(0,delta)` and `(delta,0)`. Pointwise optimal ownership differs. If a Potts disagreement penalty exceeds delta, constant ownership wins the assignment objective while sacrificing one node's best current route. Lower structured energy is therefore not retained competence or served improvement. With member-independent costs, a positive-entropy soft assignment plus nonnegative graph energy instead has uniform ownership as its minimizer; topology alone does not create useful specialists.

Graph smoothing introduces no independent evidence when every member shares an error. It can propagate the least-bad route's mistakes, and complete own CE is only a supervision floor, not a held-performance guarantee. Balancing member usage can prevent a one-owner allocation but can also force arbitrary assignments across weak/small contexts. Owner stability, graph smoothness and oracle coverage are insufficient evidence.

All ownership costs use TRAIN labels. Public graph/features may be transductive if the native protocol permits them. Building compatibility from VALID/TEST labels, held error profiles, or label-informed pseudo-targets leaks the outcome. Same-TRAIN-label affinity is permissible supervised weighting but is not label-free topology; class-composition and exposure controls would be necessary. Giving unlabeled nodes invented unary targets is outside this question. Sparse labeled subgraphs, public interior-path conventions and solver approximation costs must be explicit before admission.

## One decisive question, inactive

Test whether a **prespecified task-compatible context graph**, used only to couple stopped private ownership, improves the uniformly served VALID prediction over both pointwise private CMCL and a context-alignment permutation control, while meeting a prospectively fixed member-competence criterion. The unaries, K/beta/lambda, complete shared/own losses, initialization, native optimizer/reductions, full training budget and inference remain fixed. No compatibility map or solver has been selected here.

The alignment control must retain the relevant context sizes, label-class composition if used, degree/weight budgets and assignment-solver work while breaking node-to-context alignment. A gamma-zero method must truly reduce to pointwise CMCL: added entropy/balance, solver iterations or denominator changes cannot silently remain. If context alignment does not improve served quality, the graph contribution fails. If gains are explained by class reweighting, extra fitting, or member degradation, a competence-specific contribution is unestablished. Extra graph construction/assignment work must be charged; dense inference supplies no conditional-compute claim.

The current fixed studies continue unchanged. Any positive development result would still be evidence for this attributed composition, with unused confirmation and a further novelty argument required. A null result is recorded without a rescue hyperparameter grid. The prospect is held; no GPU request is justified by this source-only assessment.
