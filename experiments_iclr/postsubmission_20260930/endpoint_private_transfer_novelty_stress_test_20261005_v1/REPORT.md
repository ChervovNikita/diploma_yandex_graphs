# Endpoint-conditioned persistent learning: narrow novelty stress test

The frozen rule has a precise configuration difference in the inspected sources, but its strongest conceptual objection remains: it is a form of graph meta-regularization whose episode sampler conditions current private label eligibility on outer-query endpoints. A new meta-learning operator, persistence principle, or adaptation-free deployment principle is not established. No inspected paper specifies the complete frozen mechanism; that bounded observation supplies no global novelty clearance and is not an acceptance verdict.

This assessment binds the current hypothesis and `shared_private_transfer_paired_pilot_preparation_20261005_v2/STUDY_SPEC.json`. It does not alter the rule, schedule, selection, decision gates, or outcomes. The earlier graph-meta report and its eight reused conclusions were read as retained synthesis; their primary papers were not reread. Three additional primary method scopes were inspected, with zero full-paper reads. Exact identities, retrieval hashes, scopes and decisions are in `SOURCE_LEDGER.json`.

## The objection that matters most

For fixed episode state and support, write the anchored outer objective as `F_O(theta, phi)` and the private displacement as `d_m(theta) = U_m(phi_m, s_m, I_m; theta) - phi_m`. The candidate optimizes

\[
F_O(\theta,\phi+d(\theta)),\qquad
g_{\rm live}=\partial_\theta F_O+
\sum_m(D_\theta d_m)^T\nabla_{\phi'_m}F_O.
\]

That is the standard differentiation-through-learning construction, applied to persistent private blocks. For a hypothetical small SGD private step, `d_m = -alpha * grad_phi L_own^I_m`, the adapted outer loss has the familiar first-order expansion

\[
F_O(\theta,\phi')=
F_O(\theta,\phi)-\alpha\sum_m
\langle\nabla_{\phi_m}F_O,\nabla_{\phi_m}L^{I_m}_{\rm own}\rangle
+O(\alpha^2).
\]

This explains why generic transfer or gradient compatibility can account for a benefit. The frozen optimizer is stateful Adam; its displacement and sensitivity must be used exactly, so this SGD illustration is not an equivalence claim about native Adam. Persistence and recomputation make the served state follow a defined training trajectory, but SELAR already supplies virtual update → slow meta update → recomputed persistent update in graph training. ANIL/OML/BMAML and the retained MLDG/MetaReg conclusions supply the broader representation, ensemble and deployment ancestry.

What endpoint exclusion changes is the conditional distribution of the current private labels. It does not make an outer endpoint new to the persistent heads or shared representation. Previous TRAIN episodes may have supervised it, and other incident TRAIN edges and correlated neighborhoods remain context. The rule therefore defines correlated training episodes, not independent tasks or an unbiased estimate of unseen-node generalization. The final predictor performs no adaptation. Its benefit, if any, must come from the training constraint and the resulting ordinary predictor. Calling the episodes “endpoint transfer” is a useful hypothesis description, not evidence that a new learning principle has been isolated.

## The remaining nearest-prior class

| Exact primary identity and inspected scope | What the source establishes | What the scope does not establish |
|---|---|---|
| **H-GRAM:** Choudhary, Rao and Reddy, *Hyperbolic Graph Neural Networks at Scale: A Meta Learning Approach*, [arXiv:2310.18918v1](https://arxiv.org/html/2310.18918v1). §3.1, §3.3, §3.4, Appendix C Algorithm 1, opening LP protocol paragraph in §5. | §3.1 explicitly permits tasks formed from one graph, including shared labels. Local-subgraph support/query adaptation followed by a query meta update is prior. This one-graph precedent belongs in future citation discussion. | The LP protocol separates fixed edge sets. Neither that protocol nor Algorithm 1 specifies exclusion of both endpoints of every positive and negative outer query. §3.4 explicitly performs parameter updates on meta-test support until convergence. It does not specify a persistent private learner ensemble served without adaptation. Broad “disjoint nodes” wording must not be substituted for the concrete LP eligibility rule. |
| **Meta-iKG:** Zheng et al., *Subgraph-aware Few-Shot Inductive Link Prediction via Meta-Learning*, [arXiv:2108.00954v1](https://arxiv.org/html/2108.00954v1). §III-A, §III-C Eqs. (2)–(4)/Algorithm 1, §IV-A/B and prediction-complexity paragraph §IV-F. | Graph triplet support/query tasks, differentiated support learning, persistent iteration of the final GNN parameters, and a subsequent support update already occur. Training tasks distinguish high-frequency and few-shot relations. | Relation separation is not endpoint-union exclusion; train/test entity separation is also not current-episode inner/outer endpoint separation. Eq. (4) updates the full GNN from the meta-updated parameters, rather than recomputing only private learners from their original state at a new shared core. No persistent same-task shared/private ensemble is specified. Adaptation wording and the evaluation description do not alone certify the deployment implementation; this unresolved detail is unnecessary to reject an exact mechanism match. |
| **Episodic DG:** Li et al., *Episodic Training for Domain Generalization*, [arXiv:1902.00113v1](https://arxiv.org/html/1902.00113v1). §1 opening problem statement; §3.1–§3.5, Eqs. (1)–(6), Algorithm 1. | Private domain branches are initialized once and repeatedly trained. Shared features are evaluated with a classifier trained on another domain. A final fixed agnostic feature/classifier pair is returned. This is a close persistence/episodic-training precedent. | Eq. (3) treats the private classifier as constant during shared-feature training; it does not differentiate through its current learning step. Actual domains define episodes. The private branches are training partners and are not the returned served ensemble. No graph endpoint condition or candidate recomputation partition is specified. |

H-GRAM was already a metadata lead in the retained graph-meta search, so this packet claims new bounded method inspection, not new discovery of its identity. Meta-iKG and Episodic DG are complementary checks of graph update persistence and persistent private episode partners. Taken together with retained SELAR and BMAML, they leave little room for broad ingredient novelty. They still do not supply the exact conjunction of all-positive-and-negative current endpoint exclusion, shared-core-only live meta update, private inner-only recomputed commitment, and the competence-anchored raw-logit ensemble objective.

That conjunction can be stated as the configuration studied. The exact half/half objective, optimizer state partition and replay conventions are not, by themselves, evidence of historical or substantive novelty. Same-graph episodic learning already exists; this assessment does not use Meta-Graph’s multiple-graph setting to suggest otherwise.

## One discriminating analysis already available in the frozen plan

After the complete frozen family is available, use the four existing shared-F4 cells to report the paired endpoint × live-credit interaction for each block:

\[
\Delta_b=
\big(R_{b,E_{\rm end,live}}-R_{b,E_{\rm end,detached}}\big)
-\big(R_{b,E_{\rm random,live}}-R_{b,E_{\rm random,detached}}\big).
\]

Here `R` is the served VALID ranking measure at the already frozen checkpoint selector. This is one descriptive attribution contrast, not a new selection rule or adoption gate. The paired support masking and matched random arm help separate endpoint eligibility from the target masking and exposure shared by the arms.

A consistent positive interaction would support a specific within-pilot statement: the live private-learning derivative helps more under endpoint eligibility than under the matched random eligibility scheme. A live benefit that is similar in both geometries is consistent with ordinary meta-regularization; an endpoint benefit that is similar with live and detached credit is consistent with sampler regularization. Either pattern can be useful, but would not isolate the proposed endpoint-conditioned learning-credit explanation.

Even a positive interaction would not establish historical novelty, independent generalization, or an ensemble-specific benefit. Those four cells all use F4; the existing capable-single and untied controls remain relevant, and there is no single-model random-geometry factorial from which to infer that the interaction is unique to ensembles. Checkpoint selection and three development blocks limit the inferential claim. No score or outcome was inspected for this report.

One conditional trace diagnostic could explain the interaction locally: for an outer query fixed by block/cycle/batch position before inspecting its value, use already retained per-query gradient components to form `c_q = g_live,q - g_direct,q`, with both components evaluated at the same episode state and virtual private parameters, and project it onto that episode’s actual shared displacement, `-delta_theta^T c_q`. This measures the first-order contribution of private-learning credit to that query’s outer-loss change. It does not measure ranking or generalization. It can be read without a new fit or score-based example selection **only if those components and the displacement are already present in the frozen trace**. The current input documents do not establish that coverage; aggregate loss logs are insufficient. No trace replay, new logging requirement or new fit is requested here.

## Scope boundary

The search used seven bounded primary arXiv queries. No match in these inspected scopes proves no absence elsewhere. Navigation exposed incidental published numerical language and adjacent tables; no published outcome was analyzed or adopted. This packet contains compact selected excerpts, receipts and decisions rather than a raw-paper collection. No author code, server action, Desktop action, sudo, PDF compilation, GENLINK, canonical edit, model/metric execution, extra agent, plan change or manuscript verdict was involved.
