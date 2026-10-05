# Dense graph ensembles: one scoped accuracy hypothesis

## Conclusion

Preserve **graph-local member-error cross-moment pooling followed by ordinary Correct-and-Smooth (C&S)** as an attributed composition worth a supported screen. The useful question is whether graph context estimates *which members fail together* better than global weights, local competence alone, or a small graph stacker. No prediction gain, novelty clearance, experiment, configuration freeze or confirmation admission follows from this scout.

Recent learned graph prediction/hidden-state fusion is already covered by indexed HGEN (IJCAI2025), LHGEL (2025 preprint), Link-MoE (2024), diversified graph experts (2025) and FAGEL (first online2026; full chapter unavailable). Newly scoped E2GNN (2024 preprint) selects reliable teachers for distillation using a VALID-trained policy. CETNet (2024 preprint) combines member/fused supervised losses, symmetric KL and detached entropy-weighted hidden-state concatenation. These are useful ingredients and comparators; neither establishes this accuracy hypothesis. Their peer-reviewed publication status was not resolved.

## Exact small operator

Let each frozen member supply a class probability vector p_m(v), m=1,…,M. For an honest calibration label Y_v, define e_m(v)=p_m(v)−onehot(Y_v), and stack rows into E_v∈R^(M×C). The relevant population quantity is

Σ(v)=E[E_v E_v^T | context(v)]
     = Σ_c Cov(e_:c | context(v)) + Σ_c μ_:c μ_:c^T.

This is an **uncentered error second moment**: it includes conditional bias. It is not prediction variance across members, centered parameter covariance, or covariance between different graph nodes.

At calibration anchors l∈B, store G_l=E_l E_l^T. Take one prospectively fixed nonnegative graph restart diffusion H. For s_v=Σ_(l∈B) H_vl>0,

R(v)=Σ_(l∈B) H_vl G_l / s_v.

Use the global anchor average R_0 when s_v=0. Fix global shrinkage η and ridge λ>0, and form

R̃(v)=(1−η)R(v)+ηR_0+λI.

Exact reduction: e_m·e_n=(||e_m||²+||e_n||²−||p_m−p_n||²)/2. With the same normalized anchor weights, R_mn=(R_mm+R_nn−D_mn)/2, where D_mn is the graph-local average pairwise prediction squared distance. Thus full cross-moments are exactly reconstructible from local diagonal competence plus pairwise disagreement. They are not an additional independent labeled statistic. Reconstruction is the same operator and needs no duplicate experiment. For four members, one scalar disagreement generally loses which pairs differ.

Nonnegative normalized graph averaging preserves positive semidefiniteness; global shrinkage and ridge retain it. Solve the small convex problem

w(v)=argmin_w w^T R̃(v) w,
subject to Σ_m w_m=1 and w_m≥ε/M, with fixed 0<ε<1.

The lower bound retains every member in dense serving. Pool q(v)=Σ_m w_m(v)p_m(v), then apply the same ordinary C&S correction and smoothing used for every relevant control, with the same authorized labels and graph. H,η,λ,ε and the corrector must be fixed prospectively in any later supported screen; this packet adds no parameter grid.

Equivalently, before C&S, q=p̄+Σ_m(w_m−1/M)(p_m−p̄): consensus plus a weighted disagreement component. Calling that a new correction architecture would overstate it. It is local supervised ensemble weighting, with explicit graph transport of labeled error cross-moments.

## What differs from close operators

| Operator | Estimated/learned object | Relationship |
|---|---|---|
| C&S | C-dimensional labeled residual vector, then label/prediction smoothing | Existing graph error correction. Our extra object is an M×M member-error cross-moment, used before the same correction. |
| Local competence/diagonal weighting | Per-member error magnitude | Drops off-diagonal shared-error information. This is the key cheap control. |
| Global moment/convex/ridge stacking | One global member weighting or predictor-score map | Direct ancestry. If graph moments are constant, the candidate collapses to global pooling plus C&S. |
| Graph-conditioned stacking | Supervised map from all member predictions and graph context | Can represent or approximate this operation given the same moment/context inputs. The candidate offers a small constrained estimator, not an exclusive hypothesis class. |
| NCL / self-error adjustment | Training-time member losses or cross-member error penalties | Established complementarity ancestry. This candidate leaves the bank fixed and estimates context-dependent error moments for serving. |
| HGEN/LHGEL/CETNet | Hidden-state fusion plus supervised/correlation/consistency losses | Dense fusion ancestry, with different fitted objects and cost; no automatic transfer of reported quality. |
| E2GNN | Policy-guided teacher/null distillation and a served student | Reliable-teacher graph precedent. Native policy uses VALID labels; student/deployment and supervision differ. |

With identical member predictions, weighting cannot change the output. With exchangeable error moments, uniform weights are optimal. Identical consensus/disagreement patterns can correspond to very different errors because the true label is missing from prediction-only disagreement. Under common fixed linear correction/scaling/smoothing, averaging per-member corrections commutes with correction of the average; native autoscaling need not commute. A renamed disagreement GNN or gate supplies no extra principle.

## One falsifiable quality hypothesis and strong cheap controls

**Hypothesis:** on a task where member competence and shared-error patterns vary smoothly over graph neighborhoods, and calibration anchors represent those neighborhoods, local full error moments improve held-out development Brier loss and may improve accuracy over global and diagonal pooling, even after the same C&S. Improvement over both diagonal and global moments is necessary evidence that the off-diagonal graph context helps. Accuracy/MRR is not guaranteed; MRR requires a separately justified link-context construction and ranking objective, absent here.

The supported screen should compare the same frozen prediction bank, supervision and correction against:

1. Uniform probability pooling and native uniform logit pooling, each with and without the same C&S. Different pooling rules must stay explicit. All comparisons use fixed disjoint fusion-fit and development-evaluation nodes, with the entire evaluation-label mask excluded from every moment, residual diffusion and C&S anchor.
2. Global convex/ridge score stacking and global full error-moment pooling, with the same C&S.
3. Graph-local diagonal-only moments, global full moments, and graph-local full off-diagonal moments, with matched shrinkage, ridge, density bound and corrector.
4. A small predeclared ridge/logistic or MLP stacker using all member probabilities, consensus, scalar disagreement and the same graph context/authorized anchor summaries. Giving that stacker the full moment fields (equivalently, matched local diagonal competence plus all pairwise disagreement fields) also tests whether the analytic weighting rule adds value beyond its statistics. It must receive the same labels and comparable tuning allowance; contextual competence is not unique to the candidate. This is a future baseline specification, not an admitted model/tuning grid.
5. A matched context-permuted full-moment control: retain the frozen predictor bank and corrector, but permute the estimated moment-context assignment. This tests whether locality rather than added statistics explains a gain; it is not a new training-graph perturbation.
6. Apply the same pooling/corrector to a competent independent bank. Otherwise an apparent advantage could be compensation for a weaker shared bank.

Reject the quality claim if corrected uniform/global/diagonal/contextual stacks match it, if full moments help only the fitted calibration nodes, or if a gain comes from extra labels/tuning. Do not promote diversity statistics alone. A constant-context ablation is the global-moment control and needs no duplicate run.

## Supervision, cost and failure limits

For a prospective design, use fixed disjoint base-fit A, fusion-fit B and development-evaluation D nodes within the permitted supervision contract. Base predictions at B should come from a model not trained on B labels. All moment matrices, R_0, propagated residuals, C&S label resets and stacker fits use B labels only, with equivalent opportunity for controls. Score D only. A calibration node must never be scored using its own labeled residual diffusion or correction: that can memorize its target. Removing only its own Gram entry is insufficient if its label remains in C&S. For fold-based development evaluation, mask the entire scored fold throughout all moment/diffusion/correction stages, recompute those fields from the remaining permitted anchors, and report the extra fits and bank changes. Graph-correlated nodes do not become iid through cross-fitting. Final labels remain closed under the existing protocol. VALID-trained native E2GNN/Link-MoE rules cannot silently be called TRAIN-only.

Existing base checkpoints may already have used full VALID for selection. An aggregator-only VALID split or cross-fit with that bank remains retrospective development, even when the fusion-fit/evaluation masks avoid direct target leakage. It supplies no independent confirmation. Fix split, folds, subset definitions and thresholds before observing development outcomes; do not choose a retrospective threshold or favorable subset. This report prepares a protocol only: no input/output access, training, screening grid or new confirmation admission is authorized here.

The exact surrogate identity is E||Σ_m w_m e_m||²=w^TΣw for fixed context-conditioned weights. Minimizing the *estimated* regularized matrix only improves its surrogate; it does not guarantee target Brier, CE, accuracy or ranking. Few labels, rare classes, boundary/heterophilous edges, localized common bias, highly dependent anchors and distribution shift can defeat the estimate. Graph label homophily does not imply smooth member-error moments. Shrinkage may erase real specialists; weak shrinkage may overfit noise. Applying C&S changes the error field after moment fitting, so corrected evaluation is essential.

Every member still executes. Added work transports M(M+1)/2 scalar Gram fields plus an anchor-mass field and solves an M-dimensional convex problem per node. M=4 needs ten unique moment fields. There is no checkpoint/model reduction or measured runtime claim. Hidden-state fusion needs compatible heads/features and additional fitting; it is not a free substitution.

## Custody and scope

Consulted index69 first, SHA256 e7d3049ff23d8f870e2e25ea04876c6073bc2f8c56f0e33c67f40e7e630a884c. REUSED_CONCLUSIONS.json preserves 23 indexed entries, their source bindings and limits. READ_SCOPES.json records three new scoped primary identities, zero full-paper reads, exact block ranges and source hashes. The 2026 MHA/Nadaraya-Watson preprint was read only for scalar convex aggregation setup and its standard bias/variance/covariance algebra; geometry, monotonicity and architecture claims are unqualified.

An initial title lookup missed C&S under “Combining Label Propagation…”, causing one redundant v2 retrieval/reinspection. It is disclosed as a revisit with zero new identity/method-scope count. Indexed author code already resolves its printed residual-sign mismatch as Y−p followed by addition; no repeated source read was needed.

QUERY_RETRIEVAL_LEDGER.json saves exact discovery URLs, queries, UTC times, response hashes and failures. Broad discovery had many irrelevant hits; narrowed arXiv title/category/phrase searches were used. Google returned challenge text. Crossref verified “Graph ensemble neural network” DOI10.1016/j.inffus.2024.102461 (October2024), but publisher retrieval was403: metadata only, no abstract/method conclusion. GETS and Adapt/Agree/Aggregate remain abstract/metadata leads, not method reads. No numerical results, scientific datasets, score histories, logits, models/checkpoints, training or compute endpoints were accessed. This is a bounded scout, not a whole-literature absence certificate.
