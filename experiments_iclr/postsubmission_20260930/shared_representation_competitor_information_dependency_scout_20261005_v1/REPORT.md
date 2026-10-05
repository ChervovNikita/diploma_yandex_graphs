# One dependency: useful corrections must remain learnable after sharing

5 October 2026. Distinct scoped literature branch after the preserved Amazon training note. **No successor learning rule or study is admitted.** Index72, supplements V3, the saved note and retained learning/private-adaptation conclusions were consulted first. Three new primary method/theory scopes were read; zero full-paper certifications, proof certifications or author-code audits. No scientific compute/payload, SSH, source implementation, canonical/index edit or agent occurred.

## Decision

Retain one unresolved dependency: **on a prospectively defined graph regime and class pair, a competent correct-versus-competitor decision must be learnable from the information crossing the shared/private boundary, by the actual private learner under its finite budget.** Neither private parameter diversity, an optimally expressive theoretical head, nor a lower average meta-training objective establishes this property.

This dependency has strong ancestry in representation regret, shared/private adapters and meta-learning. It is not an unoccupied principle. The bounded unresolved question is whether the *particular graph-sharing boundary and finite private response* retain/use the correction information needed for complete served quality. A finding could distinguish representation loss from limited private learning; it would not by itself demonstrate ensemble necessity, novelty or graph superiority.

The adopted Amazon recurrence interpretation is consistent with asking this question: shared-member unanimity has a positive matched excess, while pooled and marginal wrong-member controls are mixed/comparable. That description does not identify which information, if any, is missing or unlearnable. The negative fusion screen remains negative. The existing two training interventions and the current private-transfer/D2 work remain unchanged; this note adds no arm, gate or implementation to them.

## Exact recent antecedents

### 1. DSA: shared/private channels plus structurally different adapters

Jiaqi Wu, Junbiao Pang and Qingming Huang, **Decorrelating Structure via Adapters Makes Ensemble Learning Practical for Semi-supervised Learning**, exact [arXiv:2408.04150v1](https://arxiv.org/html/2408.04150v1). Method SectionsIII-A-D, Eqs1-13, were inspected. This is a direct 2024 shared-representation ensemble antecedent, rather than an output-calibration method.

The common feature map is expanded and partitioned into shared channels H and member-private channels G_m. Each member receives

    F_m = concat(H + Delta H_m, G_m).

Delta H_m is produced by a private two-layer 1x1 convolutional adapter. The first layer expands channels, the second restores them; widths vary across members (the example adds10*m channels), as do activations such as PReLU and ELU. The method removes the predecessor CBE's private-channel decorrelation loss. Labeled examples receive mean own-member CE. For unlabeled examples, Eq12 constructs a confidence-filtered average prediction; Eq13 uses it as the common pseudo-target for all heads.

This occupies **architecture-level private processing of shared features, retained private channel information, and heterogeneous adapters without an explicit repulsion loss**. A graph port would replace the image operations and expose a real graph-information boundary; it would not invent the split or adapter idea. The selected semi-supervised prediction is a confidence-filtered mean, not the existing fixed Amazon arithmetic pool. Exact test-time aggregation, pseudo-target detachment and native optimization are not source-qualified. Eq12 divides by M without renormalizing retained heads; it need not sum to1. A faithful port requires a declared interpretation/code check.

The theoretical discussion does not certify competitor-specific information preservation. Eq5 names the operation `cov` but defines a Pearson correlation of flattened features. Eqs9-10 then approximate concatenated-feature correlation using expressions for sums, omit cross terms, and invoke variance/covariance inequalities not implied solely by different widths/activations. Statistical independence of learned adapter/private outputs is asserted; it does not follow from architectural difference under a common input and joint training. These are limits of the displayed derivation, not a rejection of the implemented architectural antecedent.

For example, all residual adapters can implement zero and all private channels can implement the same function. Distinct widths/activations then coexist with identical usable features and predictions. Conversely, branches can differ only on task-irrelevant coordinates. Thus different structures or lower raw feature correlation do not establish useful class-pair correction capacity. No DSA score or runtime is adopted.

### 2. Representation game: task-specific regret beyond dominant covariance

Neria Uzan and Nir Weinberger, **A representation-learning game for classes of prediction tasks**, exact [arXiv:2403.06971v1](https://arxiv.org/html/2403.06971v1). Section2, the linear-MSE formulation and pure minimax result in Section3, and Section4's method/Algorithm1 were inspected. Mixed-strategy formulas were exposed; their full proof is not certified.

The representation player's objective is the worst task's **regret relative to the original features**, where each represented task is given an optimally fitted downstream predictor:

    regret(R,f) = inf_q E loss(Y, q(R(X)))
                  - inf_Q E loss(Y, Q(X)).

In the stated linear-MSE setting, X is centered with positive-definite covariance Sigma; representations have rank at most r<d; response vectors satisfy f^T S^-1 f<=1 with positive-definite task-prior S; noise has conditional mean zero. The reported pure minimax regret is the (r+1)th eigenvalue of Sigma^(1/2) S Sigma^(1/2), with a whitened projection onto its top-r eigenspace. **Task direction importance enters explicitly through S; raw feature variance alone is insufficient.** This supplies direct theory ancestry for preserving information useful to a class of corrections rather than merely preserving high-variance features.

The standard fixed-R calculation explains the limitation precisely. Optimal linear fitting leaves residual risk

    f^T [Sigma - Sigma R (R^T Sigma R)^-1 R^T Sigma] f

for full-column-rank R. Every downstream linear member uses the same retained subspace; multiplying heads does not recover its orthogonal complement. This is a restricted linear calculation, not a theorem that the actual nonlinear graph model loses information. The saved information-boundary report already establishes the more general identical-input obstruction and warns that raw-feature/skips/additional graph context can defeat it.

The paper also optimizes randomized representations. Algorithm1 alternates finding a poorly represented response function and adding a representation atom, fitting a predictor for each representation/task pair and updating game weights. **This is already task-directed representation learning**, rather than arbitrary feature repulsion. Its mixture is a randomized representation strategy over repeated games, not a trained shared-core committee's fixed mean prediction. Known feature distribution and enough labeled data for each downstream task are assumed. For the general nonconvex-nonconcave algorithm, the paper explicitly leaves convergence guarantees elusive. Neither optimal-head regret nor mixed-game benefit proves useful information is reachable by a few private Adam updates or preserved by a fixed graph pool.

### 3. DERTS: preserve an adapted task-pool learning gradient

Donglin Zhan and James Anderson, **Data-Efficient and Robust Task Selection for Meta-Learning**, exact [arXiv:2405.07083v1](https://arxiv.org/html/2405.07083v1). Sections3-4.4, Algorithm1 and AppendixA's MAML/ANIL definitions were inspected. AppendixB gradient-estimator derivation and AppendixC theorem proof were not certified.

A support update produces task-adapted parameters phi_i; query gradients with respect to the meta-initialization define task descriptors. Tasks are represented by nearby selected tasks in gradient space; selected-task weights count assigned tasks. A facility-location objective covers the pool's estimated gradients, and standard meta-learning runs on the weighted subset. The optional noise heuristic drops selected tasks with high estimated gradient norms. This supplies **adaptation-dependent, task-sensitive sampling/weighting ancestry**, with both gradient-based and metric-based applications. It is not member repulsion, graph-conditioned private ensembles or adaptation-free serving.

The main theoretical target is training dynamics/full-pool loss under smoothness, local PL and Hessian assumptions, plus bounded gradient approximation error. It is not a bound on every class-pair task or on information retained by a shared graph encoder. Even accurately following the full-pool training rule does not make that rule an information-preservation certificate. The paper's gradient cover is stronger than merely equating an average, so it should not be caricatured as ignoring all task gradients; the limitation is the endpoint it certifies.

The implementation-oriented text calls the softmax preactivation gradient `logits - label`; the usual CE cotangent is probability minus label. The intended last-layer estimator needs its actual derivation/author source before a native port. The noisy-task rule is an explicit heuristic: a difficult but informative rare competitor task can also have a large gradient. No literal estimator, theorem, threshold or recipe is inherited here. The scoped theorem statements concern ordinary episodic tasks, not dependent same-graph regions, private Adam histories or graph ranking.

## The one dependency, made concrete

Define a finite collection of graph regimes prospectively from permitted public topology/features, and include **all class pairs**, using training labels only. A class-pair task must contain both classes; a positive-only task admits a constant answer and cannot measure competitor information. Keep empty/low-support cells explicit. Use the same allowed graph context and all-target competence requirement for every predictor. Do not construct tasks from the saved VALID recurrence pairs or select favorable class cells after outcomes.

Let B_theta(X) mean **all information the private learner can actually receive** after the declared sharing boundary, including available graph/context, raw-feature skips and retained states. It must not be replaced by one node embedding if the native private path consumes more. Let L_a be binary class-pair query CE on fixed cell a, with the rest of the multiclass task protected by the same competence constraint.

For an admissible private predictor f_B produced by the actual B-budget update map from declared private weights/optimizer state/support, use the conceptual decomposition

    L_a(f_B) - inf_(full allowed predictors) L_a
      = [inf_(capable predictors using B_theta) L_a
         - inf_(full allowed predictors) L_a]
        + [L_a(f_B)
           - inf_(capable predictors using B_theta) L_a].

The first bracket is information/representation regret; the second is a private capacity/finite-learning gap. They are nonnegative when the comparator classes are nested, use the same information/competence constraints, and contain f_B. If the learned member fails competence, it is inadmissible rather than evidence of preserved information. Infima are theoretical benchmarks; finite probes do not certify them. No regret or response was computed here.

The unresolved dependency is that these two sources of failure must be small on the correction tasks the ensemble is meant to resolve. **Good pooled risk, differing private factors or more decorrelated features do not separate them.** An optimal theoretical head may exploit information that a restricted private update cannot reach. A powerful private learner cannot recover information erased at the common boundary. Public graph context/private message paths can invalidate an apparent compression obstruction; their exact placement must be identified before interpreting it.

This is different from the saved optimizer-history alignment question: it concerns what correction can be learned at the sharing boundary under the stated response budget, not whether a row's old Adam moments are aligned with its weights. It is also different from testing graph-smoothed responsibility or edge masks. Those interventions could affect either bracket; their quality alone does not identify which bracket changed.

## A discriminating interpretation, not a new run

A future, separately authorized source/experiment could compare the actual finite private learner with a capable nonlinear predictor on the **same frozen boundary information**, and a competent full-context graph single. All receive the same permitted labels, selection opportunities, information outside the disputed boundary, complete schedules and global competence constraints. Every probe's fitting and tuning is paid. Prospectively split training support/query roles; shared graph dependence and earlier label exposure remain disclosed. Existing validation-selected endpoints are not an untouched measurement.

| Possible complete comparison | Supported interpretation and limit |
|---|---|
| Boundary probe succeeds; finite private learner fails | Consistent with limited private capacity/optimization/response budget. It does not identify representation information loss. |
| Boundary probe fails; competent full-context single succeeds | Consistent with a harmful sharing boundary, but finite probe failure does not prove positive optimal representation regret. Rule out weak optimization/information mismatch. |
| Both fail | Does not distinguish representation, noise, data support or inadequate training. |
| Both succeed; fixed pool still fails | Useful information/private competence are insufficient to establish the served ensemble gain. Do not substitute an oracle selector. |

Keep ordinary independent4 as a full trained reference; use an untied version of the same private-learning rule if attributing benefit to sharing. A capable single given the full boundary/task information tests whether one model already learns the corrections. If it matches/exceeds the ensemble, the dependency may be satisfied while ensemble necessity remains unsupported. Mean class-pair improvements cannot rescue a failed complete pooled accuracy/NLL/Brier criterion or weaker members. This comparison is an interpretation specification only: no graph/seed/budget/selector/arm is prepared or admitted.

## Search, failures and accounting

Six bounded OpenAlex queries returned54 metadata rows across 2024-2026 searches on shared ensembles, adapters and representation/meta-learning theory. They are relevance-ranked discovery, not a field census. Three selected IDs were absent from index72 and the consulted supplements; all three version-specific HTML requests succeeded. No GENNN route or inaccessible prior locator was retried. There are **no HTTP failures in this branch**.

One local record-generation assertion failed because its identifier parser split the letter `v` in `arXiv`; no record had yet been written. The corrected terminal-version-suffix check verified all three IDs. LOCAL_RECORD_FAILURE.json preserves that failure and correction; it changed no primary read or scientific state.

Other results remain metadata/abstract-only, including2025 conditional representation learning,2026 adapter-interference geometry and a recent LoRA overview; no method claim from them is adopted. Searches did not justify a later-version or published/preprint equivalence claim. DSA's method section contains a cost/result table, and representation-game method text contains empirical examples; those were incidentally exposed but no value supports this conclusion. Figure captions/algorithms/equations were read as stated; figure pixels, full results, complete references and proofs were not audited. All extraction output is distinct from semantic/full-paper reading.

Exact primary bytes/scopes, identifier checks, query receipts and retained conclusions are in SEARCH_LOG.json, RETRIEVAL.json, READ_SCOPES.json, PAPER_CONCLUSIONS.json and SOURCES.json. Saved DICE/FoRDE, GNCL, ANIL/BMAML, TabM/LoRA-Ensemble/CAMERO, information-boundary and persistent-private-learning conclusions remain attributed. The strong predecessor literature and these new methods prevent calling task-conditioned feature retention, diverse adapters or adaptation-sensitive task weighting new.

**Disposition:** preserve the one learnability dependency and its causal alternatives. No new graph-learning operator survives as a novelty-cleared successor from this bounded pass. The task-specific, finite-private-learning question remains open for the actual shared graph architecture, with no predictive result or execution authority supplied here.
