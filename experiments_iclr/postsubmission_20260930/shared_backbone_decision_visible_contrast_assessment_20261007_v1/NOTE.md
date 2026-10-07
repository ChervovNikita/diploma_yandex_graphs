# Decision-visible contrast under a shared graph backbone

7 October 2026. **Class-centered hidden contrast admits a rigorous output-invisible counterexample compatible with its alignment term. A finite graph-response loss removes that specific blindness, but is an attributed functional-diversity ablation—not a new learning principle or a demonstrated remedy for common errors.**

The supplied closed initialization21 summary reinforces the reason to test the distinction: average memberMRR increased from0.259492 to0.271067, while poolMRR fell from0.282533 to0.271700 and common negative errors increased. Every graph-versus-independent seed lost, with wide intervals. These are supplied bounded results; no raw scores were read, significance inferred or general claim adopted. Better average members did not supply a better pool in that study.

## 1. The two-loss counterexample is valid, with explicit conditions

At a representation/linear-head interface, construct

`h_m(v) = [a(v), C c_y, A b_y(v) u_m]`, and head `[W_a,0,0]`.

Here c_y are unit class codes; u₁,…,u₄ are a unit regular simplex, so `u_m·u_n=−1/3` for m≠n. Let b_y be view-invariant, nonzero and sum to zero within each constructed TRAIN class. Balanced±1 values work for even class counts; general nonzero zero-sum values avoid requiring even counts. Singleton classes are skipped by the frozen residual objective. Keep a bounded across the two views, take `A≫||a−classmean(a)||`, and take C large enough that `max||a||/C→0` and `A/C→0`. Merely comparing C with the centered a residual would not suffice for raw alignment.

The logits are exactly `W_a a(v)` for every member. Class-centering removes `C c_y` and preserves the zero-mean nuisance block. As A dominates the remaining residual, normalized residuals approach `sign(b_y(v))u_m` in both views. The same-member positive cosine tends1, and every other-member cosine tends−1/3. The frozen symmetric residual-member CE therefore tends

`L_R = log[1+3 exp(−4/(3T))]`,

approximately0.00381 atT=.2, despite identical predictions and errors across members.

Meanwhile raw normalized features approach c_y. Same-class opposite-view positives have cosine1, and different-class pairs have cosine `c_y·c_y′`. For an anchor in class y, the alignment loss tends

`L_A(y)=log[n_y+Σ_{y′≠y} n_y′ exp((c_y·c_y′−1)/T)]`.

This matches a class-cluster representation and satisfies the intended within-class alignment. **The alignment CE does not tend zero:** averaging log-probabilities over all n_y positives has the unavoidable lower bound `log n_y`. Orthogonal or class-simplex codes separate different classes while the nuisance/member block becomes negligible to this term. Thus both auxiliaries can be compatible with task-invisible member separation; claiming both losses become zero would be incorrect.

This is an interface/loss counterexample. It is not proof that the frozen nonlinear/normalized BE GNN realizes these coordinates, a full-objective stationary point, or a generalization outcome. In particular, own CE can give a nonzero gradient to the currently unused class-code head columns. Head learning may subsequently use them. The construction proves that a small residual loss and good class alignment do not certify decision diversity at the evaluated state.

An even more general interface statement is `H_m→H_m+U_m` with `U_m W_head=0` and within-class row means zero: logits remain unchanged while residual contrast can change. Compensated orthogonal feature/head transformations also preserve logits and within-member cosine alignment while changing cross-member residual angles. Neither statement is claimed as an arbitrary whole-network symmetry under tied weights and private diagonal factors.

## 2. What changing to graph responses does—and does not—solve

Use **actual prediction functions** on fixed inputs. For WikiCS's mean-probability pool, take competitor probability margins

`f_m(v,c;G)=p_m(y_v|G)−p_m(c|G)`, c≠y_v,

and fixed graph/feature interventions T_k. Define the response vector

`d_m(v)=concat_{k,c≠y_v}[f_m(v,c;T_kG)−f_m(v,c;G)]`.

Hidden-coordinate changes that preserve predictions on G and all T_kG leave d exactly unchanged. A loss on d therefore cannot reward the counterexample's classifier-nullspace code. For a ranking task, the corresponding functional is the **positive-versus-specific-negative score difference**, retaining competitor identities; node probability-margin conclusions do not automatically transfer to link MRR or its pooling rule.

This does not force useful evidence. Members may have identical factual errors yet different harmful responses on perturbed inputs. A shared wrong competing class can still defeat the mean-probability pool. Confidence changes are prediction-visible without being complementary correct decisions. Finite interventions expose only selected inputs, and thinning/zeroing is not certified label-preserving. Saturated probabilities, tiny response norms and normalization can generate misleading angles. Capacity restrictions remain: a diversity loss cannot recover information absent from every retained path or remove a single linear BE map's separable cross-ratio coupling.

The prior mapping is decisive:

- **FoRDE** already diversifies normalized true-label function sensitivities; finite graph responses are a domain/projection adaptation, not a new functional-diversity principle. Its particle update is not identical to an ordinary CE-plus-response loss.
- **Function-space repulsive ensembles** already use finite predictive evaluations. d is a fixed differencing map of that function vector.
- **CDLG** precedes the same-object cross-member/channel negative pairing; **DICE** precedes conditional task-information preservation motivation. Class-centering is still not conditional mutual information.
- **SuGAr** precedes learned graph-evidence diversification with supervised risk/contrast; **attention disagreement** precedes operator/head diversity. Their mask/attention penalties are not equal to final prediction responses, but the distinction does not establish novelty.
- **NCL/GNCL** precede prediction-error diversity and member/pool loss tradeoffs. A quadratic variance reward on d would be NCL-like on transformed predictive errors. The proposed pairing below is not that quadratic identity, but it does not become a new principle by changing its proxy.

No genuinely unread source section is needed to resolve these distinctions: the decisive scopes are already saved. No absence-of-prior certificate is inferred.

## 3. Earlier project proposals close the apparent method gap

The three additional retained reports resolve the apparent next-method opportunity:

- `heterogeneous_label_relevant_diversity_theory_20261003_v1/REPORT.md` already gives classifier-nullspace and compensated-output falsifiers, plus exact loss/pooling bookkeeping. The new simplex construction specializes that concern to **both auxiliaries in the present loss**, including the alignment entropy floor; it does not supply a new learning rule.
- `graph_contrastive_private_paths_quality_gap_20261003_v1/REPORT.md` already specifies class/neighborhood-conditioned finite-removal probability responses, centered/normalized redundancy, private intermediate adaptation with common maps/heads frozen, native/probe CE and energy guards, augmentation-only/class-only/permuted-mask controls, and qualified DICE/FoRDE comparators. Its warm acquisition, graph caches, guard roles, backtracking and all-member paths have substantial unqualified cost. Those safeguards protect TRAIN control sets only and can overfit or reject legitimate invariance. This proposal is not reissued here.
- `conditional_graph_response_closest_priors_20261003_v1/REPORT.md` proves that squared response cosine is exactly a degree-two function-kernel energy. With identical responses, groups, centering, normalization, coefficients, update permissions and guards, the purported kernel comparator is the same method and needs **no duplicate fit**. The report also covers function-space RBF/KDE ancestry, CF-GNNExplainer's finite graph-removal responses, and AD-GCL's learned graph-view role.

A response InfoNCE loss would differ algebraically from that degree-two energy, and allowing live shared-W learning differs from its frozen-common/private-only update. These are explicit loss/optimization choices within an already attributed family. CDLG-style pairing on function features, or changing which parameters are permitted to learn, does not by itself create a new mechanism. A bare ADP determinant is also unavailable as an unmodified control when M>C−1: binary molecule/M4 has rank at most1. Epsilon or pseudodeterminant variants change its method.

The unresolved AAAI2025 counterfactual-augmentation lead (DOI10.1609/aaai.v39i18.34101) remains method-unread here. Resolving it would refine complete-operation overlap, but would not erase the already established function-kernel, graph-removal and diversity ancestry. It is not decisive to the present counterexample or a basis for a novelty claim, so no new retrieval was made.

## 4. What remains useful now

**No distinct next mechanism survives this assessment.** The new useful distinction is a rigorous **two-loss compatibility counterexample**, rather than the earlier generic nullspace objection. It makes the interpretation of the running pilot sharper: lower hidden residual CE plus class alignment cannot certify useful decision diversity. The initialization21 closure likewise shows why mean-member improvement is insufficient.

Finish the unchanged full family and its already frozen response/error panel. The meaningful observations are whether the contrast package changes actual class/competitor decisions and fixed graph responses, reduces common wrong-class/negative identities, and improves the declared served pool against capable controls. If hidden loss/separation improves while those functions do not, output-invisible satisfaction remains a supported explanation. If responses change but pooled utility does not improve, prediction visibility alone did not repair useful complementarity. Neither observation proves the specific appended-code construction occurred.

For ranking, retain exact wrong-negative identities and the actual score-pooling contract; same-query error correlations or memberMRR cannot establish rescue. For node classification, the mean-probability same-competing-class condition remains the relevant non-rescue witness. The existing panel uses fitted TRAIN targets and finite, potentially harmful interventions, so it does not establish causal sufficiency or heldout generalization. Subsequent confirmation and stochastic-path-matched single/untied controls remain necessary if the current package warrants them.

Disposition: record the counterexample and the existing prior mapping; do not rename the old conditional-response proposal, add another response objective, reopen its expensive guard study, or change running families on this evidence. No implementation or compute is admitted.

Scope: supplied initialization closure and counterexample assessed; saved DICE/CDLG/SuGAr/attention/FoRDE/function-space/NCL-family conclusions and the three3October reports reused. New primary source reads/retrievals, full-paper credits, author-code scopes, raw outcome reads and experiments: zero. Exact reused bindings and limits are retained alongside this note.
