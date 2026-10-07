# Update-wise target assignment shuffling: inactive assessment

8 October 2026. Saved-prior and mathematical assessment only.

## Judgment

**Retain as a useful conditional control for persistent route/context association.** Compared with COMMON alone, it matches persistent ROUTE's instantaneous target concentration/support/entropy multiset while removing a fixed route's long-term context identity. It is more informative for that particular attribution question. COMMON remains necessary as the deterministic average-target reference; shuffling adds temporal assignment noise and is not a cleaner replacement for every purpose.

The conditional expected objective and every parameter-block raw gradient equal COMMON at arbitrary current states and views. No route symmetry is required. The mathematical assignment-averaged objective therefore has the same minimizers and stationary points as COMMON wherever defined. Finite stochastic training, Adam states and selected endpoints can differ; those are not different optima of the averaged objective.

Keep this proposal inactive. Only after whole9 closure and a worthwhile result would the specified three-seed comparison warrant consideration. No source, target, active condition, gate or outcome was changed or opened here; no novelty or execution claim follows.

## The operation and exact conditional identity

At each update, draw a uniform permutation pi of the four factual target indices from a dedicated assignment RNG. Give route m target Q_pi(m). Every factual target appears once; sum_m Q_pi(m)=4 Qbar on that update. Keep the assignment through both original view directions, gradient collection and any replay/recomputation. All scored-panel restriction and row normalization happen before this selection, using the original frozen definitions.

Condition on the complete pre-update history/state H and current forward/view realization xi. For direction d, write the original full-denominator alignment as

```
A_m(Q;H,xi) = -(1/(2n)) sum_d,ij Q^d_ij log P^d_m,ij(H,xi),
A_shuffle(pi) = (1/4) sum_m A_m(Q_pi(m)),
A_common = (1/4) sum_m A_m(Qbar).
```

For each m, the marginal distribution of pi(m) is uniform on all four targets. Therefore

```
E_pi[Q_pi(m)^d | H,xi] = Qbar^d,
E_pi[A_m(Q_pi(m)) | H,xi] = A_m(Qbar),
E_pi[grad_b(F + alpha A_shuffle) | H,xi]
    = grad_b(F + alpha A_common)
```

for any block b, including shared parameters, each private factor block, and a complete stacked parameter vector. The four assignments are dependent, but expectation is additive; independence across routes is unnecessary. This is a finite average over24 assignments, so differentiating that average introduces no expectation-interchange problem beyond well-defined pathwise derivatives. The same identity holds for scalar losses. It does **not** assert equality on any one sampled assignment.

At a fixed state/views, if J_b,m,d is the score Jacobian, the residual raw gradient is

```
delta g_b = -(alpha/(8n)) sum_m,d
            J_b,m,d^T vec(Q_pi(m)^d-Qbar^d),
E_pi[delta g_b | H,xi] = 0.
```

Equal Jacobians can additionally make a shared residual zero for every permutation, but they are not required for the mean identity. Private residuals remain in separate block coordinates even when their route laws match. Unlike the earlier fixed-assignment cancellation, this expectation holds after private states diverge because the current assignment is randomized independently of those states.

### Required assumptions

- The permutation is conditionally uniform given history and current views. Its dedicated RNG does not advance member dropout/sampling streams or select targets according to current losses, route strengths or representations. Past assignments may influence current parameters; conditioning on that history is valid because the new assignment is independent.
- Forward scores, candidate denominators, view laws, object weights and Jacobians do not depend on which target is assigned. Targets are fixed/stopped and enter only the linear CE weighting. Keep the original complete denominator; the previous target-support-dependent masking proposal would invalidate this identity.
- COMMON is the exact directional mean after the identical panel restriction and normalization. If reverse targets have a separate convention, shuffle their paired forward/reverse target together and average each direction correctly.
- Use the original mean-member reduction, coefficient, target bank and own CE. The identity applies before nonlinear gradient clipping, projection or optimizer transitions.

Drawing four targets independently with replacement would retain marginal unbiasedness but lose the per-step target multiset/mean match and change cross-route noise. Cycling one target across all routes is also different. Avoiding repeat permutations or identity assignments can break the stipulated conditional law; they must not be introduced as undocumented improvements.

## Dependence and variance: what can be said exactly

Concatenate the two directional target matrices into q_k and set Delta_k=q_k-qbar. For a uniform permutation, with C_Q=(1/4)sum_k Delta_k Delta_k^T,

```
Cov(Delta_pi(m)) = C_Q,
Cov(Delta_pi(m),Delta_pi(l)) = -C_Q/3,  m != l.
```

Assignments have negative cross-route covariance in these target coordinates. Different route Jacobians transform this into the supplied gradient covariance; it is not equivalent to four independent target draws.

There is a useful stronger variance statement in this particular comparison. At fixed H, COMMON's raw gradient g_c(xi) is exactly E_pi[g_shuffle | H,xi]. For finite second moments and the same target-independent view law,

```
Cov(g_shuffle | H)
 = Cov(g_c | H) + E_xi[Cov_pi(g_shuffle | H,xi)].
```

The added term is positive semidefinite. Thus COMMON weakly reduces raw-gradient variance relative to this stochastic estimator at a matched state; the extra term can vanish for some blocks. This is different from comparing persistent ROUTE against COMMON, where the necessary conditional-mean property generally fails and a covariance-sign conclusion is unavailable.

Greater gradient variability need not harm quality and is not evidence of useful specialization. Adam's second moments, nonlinear normalization, clipping and subsequent state dependence can change immediate and later displacements. Even raw SGD having the same conditional mean step at a matched state does not equate whole trajectories or their expected final parameters.

## What the control identifies, and remaining failures

Persistent ROUTE and update-wise shuffling receive the same four factual target distributions once each per update. The auxiliary panel, denominator, contexts, positives, capacity and number of model forward/backward/update operations remain the same. Each shuffled route encounters multiple contexts over time, rather than a fixed context. This removes the COMMON arm's instantaneous smoothing/concentration difference from the persistent-versus-shuffled contrast.

It still changes temporal target continuity, target/state association and stochastic optimizer inputs. A weak shuffling result can reflect disruptive task switching or noisy moment estimation. It does not by itself prove that persistent routes learned useful competent roles. Compare the same selected whole-population quality and prediction-level readouts with COMMON as well as persistent ROUTE; never substitute a favorable error cohort or auxiliary-loss improvement.

| Later complete pattern | Bounded interpretation |
| --- | --- |
| Persistent ROUTE beats both shuffling and COMMON while retaining competence | Supports stable context association as useful under this recipe; switching/noise remains part of the tested difference. |
| Shuffling matches ROUTE and both beat COMMON | Persistent association is unnecessary in this comparison; stochastic concentrated targets may account for the benefit. It is not proof that entropy alone caused it. |
| ROUTE matches COMMON and both beat shuffling | Shuffling may be harmful; its loss does not rescue a persistent-specialization claim. |
| Shuffling beats ROUTE | Persistence is unsupported as the proposed advantage. |

All rows are prospective interpretations, not read results or replacement gates. Ordinary own CE may dominate so none differ materially. Jacobian nullspaces, parallel same-class cosine derivatives or unchanged decision margins may erase useful target steering. Full denominators retain the existing unselected-same-class distractor risk. Matching the multiset does not certify useful graph context, prevent competence damage or isolate topology causality; the existing factual-versus-node-permuted comparison keeps its separate role.

## Saved-prior collisions and distinctions

The original method already saves an inert same-context cycling option: all routes use Q_(update modulo4), averaging to Qbar only across four updates at an unchanged state. This is local temporal-target ancestry, not an exact duplicate of a bijection using all four targets on every update. It provides no conditional identity at changing states.

The original `shared_route_permuted` arm permutes node incidence within its constraints once. It preserves route context identity and changes factual relations. The proposed shuffling preserves factual targets and changes their route association repeatedly. Name it separately; do not relabel or alter the frozen PERMUTED arm.

Retained SupCon/BotSCL, weighted class-conditional contrast, PMGCL positive mining, AMCL shared-backbone heads and CGCL multiple unchanged-graph views establish the ingredients. PMGCL remains abstract-only and MA-GCL method access remains unresolved. The two saved AMCL/CGCL method scopes do not certify this exact assignment operation or its absence elsewhere.

A targeted saved-index search also found an existing prospective balanced-random private-allocation control in `graph_structure_conditioned_specialization_scout_20261003_v1/PROSPECTIVE_TEST.json`: it matches cell sizes/work and replaces structural ordering with a fixed hash assignment. That is local matched-assignment-control ancestry; it is not this update-wise contrast-target permutation. No new paper body was read. Unbiased sampling of a linear target average is ordinary stochastic-objective calculus, not a new optimization principle. Exact complete published duplication remains unresolved; no novelty clearance is attempted.

## Conditional comparison and custody

If justified after complete whole9 interpretation, retain one new condition: update-wise uniform factual-target assignment shuffling. Use all three original paired seeds, the original shared unit predictor, native two-own-view CE, temperature0.2, auxiliary coefficient0.05, full denominator, trajectory, selection rule and mean-probability serving. If source/target/init/view/selector contracts and terminal custody match, reuse every eligible original ROUTE and COMMON anchor and add exactly3 complete paired-seed fits. If compatibility fails, do not claim that budget or comparability; resolve it before any fit. No grid or favorable-seed replacement.

Bind a dedicated assignment RNG specification and its state without borrowing member streams. Preserve the exact permutation through both views, all cotangent collection and checkpointed/recomputed forwards. Save enough step/RNG identity for deterministic replay and continuation; do not redraw inside backward/replay or after an outcome-dependent retry. Validation/inference needs no target assignment or extra predictor. Match model operation counts; charge assignment draw/indexing, metadata and replay overhead rather than claiming identical wall time without measurement.

Report the existing full-population accuracy/NLL, mean/worst member competence, paired repairs/harms, COMMON-frozen all-rival rank acquisition and pooling rescue. Keep actual repairs distinct from candidate-correct counts and name all reference populations. No new acceptance threshold, source implementation, active protocol permission or fit is supplied by this assessment.

## Scope

Used saved method/prior conclusions and local prospective controls only. A grep against the compact one-line saved index initially emitted a truncated broad line; it was not treated as a whole-index semantic read or absence certificate. A bounded JSON string-leaf search then returned three matches, with only the relevant local prospective control's assignment descriptions inspected. No raw scientific result, model, target tensor, current/partial outcome, paper body or server was opened. Hand algebra only; output and seals are confined to this inactive assessment folder.
