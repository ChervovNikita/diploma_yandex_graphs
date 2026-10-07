# Expected shared gradients and a split context-alignment policy

8 October 2026. Theory and method assessment using retained notes only.

## Verdict

**The expected shared raw-gradient difference between fixed route targets and their common mean is zero at a fully symmetric state under matching, target-independent joint two-view stochastic laws. Independent dropout realizations do not invalidate this expectation identity.** They invalidate equality for an individual realization. Matching the marginal law of each view separately is insufficient.

The proposed rule gives shared parameters own CE plus common-target alignment, and private r/s factors own CE plus route-target alignment. At the **same current parameter state and realized views**, its shared raw gradient equals common-all and its private raw gradient equals route-all by construction. Those equalities do not compare different training trajectories, ensure equal optimizer displacements, protect competence, or establish improvement.

**Disposition: retain the exact specification as an inactive, attributed ablation; defer new source preparation.** The policy is a genuine but incremental gradient-allocation change. Its initial motivation cannot be removal of a nonzero expected shared drift from heterogeneous targets under the stated symmetry assumptions. It could change shared stochastic variability and subsequent coupled dynamics, in either direction. The already prepared scalar route/common context comparison addresses target utility; this assessment supplies no distinct diagnosed failure that warrants extending it now. No implementation or experiment is proposed as an automatic next step.

## 1. Define the comparison before taking expectations

Let theta contain all shared trainable parameters and phi=(phi_1,...,phi_M) the route-indexed r/s factors. The formulas for individual private blocks below assume phi_m influences only route m. Any additional trainable parameter must have a declared recipient; unit r/s values alone do not certify a symmetric or exhaustive partition.

Fix a common auxiliary panel of n objects, a parameter state, and one realization xi of all stochastic views. Nonzero representations h_m^1(i), h_m^2(i) produce scaled cross-view cosine scores

```
S_m^12(i,j) = cos(h_m^1(i), h_m^2(j)) / tau,
S_m^21 = (S_m^12)^T,
P_m^d(i,:) = softmax(S_m^d(i,:)),  d in {12,21}.
```

Each directed target Q_m^d is fixed and row-stochastic after the identical panel restriction and normalization. Define its common counterpart **after that operation**:

```
Qbar^d = (1/M) sum_m Q_m^d,
A_m(Q_m) = -(1/(2n)) sum_d sum_ij Q_m^d(i,j) log P_m^d(i,j),
A_r = (1/M) sum_m A_m(Q_m),
A_c = (1/M) sum_m A_m(Qbar),
Delta_m^d = Q_m^d - Qbar^d,   B = A_r - A_c.
```

The common-target and route-target losses use exactly the same representations, scores, denominator, views, panel and reduction. Let F be mean-member own CE, with the original two-view and object reductions, and let alpha be the common auxiliary coefficient (0.05 in the retained method). Private gradients retain 1/M; substituting unscaled per-member CE or alignment changes the learner.

For a fixed direction, ordinary soft-label cross-entropy gives

```
partial A_m / partial S_m^d = (P_m^d - Q_m^d)/(2n).
```

Since each Delta row sums to zero, the log-normalizer cancels in the loss difference. For any parameter block b, writing J_b,m,d=partial vec(S_m^d)/partial b,

```
B = -(1/(2Mn)) sum_m,d <Delta_m^d, S_m^d>,
partial_b B = -(1/(2Mn)) sum_m,d J_b,m,d^T vec(Delta_m^d).
```

These are same-state identities at arbitrary route states. No equal-score assumption and no expected-softmax argument is needed for the difference: the target-independent P term cancels within each route. The total supplied shared gradient difference is alpha partial_theta B because the own CE term is identical.

### The reverse-target convention matters

If the implementation uses the same directed Q_m in both anchor directions, set Q_m^12=Q_m^21=Q_m and Qbar=mean Q_m. Then

```
B = -(1/(2Mn)) sum_m <Delta_m + Delta_m^T, S_m^12>.
```

Consequently, this symmetric loss's target-dependent difference sees only the symmetric part of the directed target contrast. An antisymmetric Delta with zero row sums is invisible, even when the targets differ. This is an additional possible null signal, not an assertion about the saved masks.

If reverse targets are constructed differently, the matching requirement is sum_m(Q_m^21-Qbar^21)=0 as well as its forward counterpart. Transposing and then row-normalizing is nonlinear: rownorm(Qbar^T) need not equal mean_m rownorm(Q_m^T). A mean formed before panel restriction/renormalization can likewise be wrong. Such conventions need an explicit direction-specific common target. If unnormalized transposed targets are used, the score gradient is row_mass(Q)P-Q rather than P-Q; target linearity still permits cancellation when the common target is the exact directional mean, using the Jacobian of log P. This report does not audit native source or claim which reverse convention it implements.

## 2. Expected shared cancellation at the symmetric state

Condition first on a common panel/input state C. It can include fixed TRAIN-derived targets, graph inputs and a target-independent panel schedule; no such data are accessed here. Require

```
E[J_theta,m,d | C] = Jbar_theta,d(C)  for every m,d,
sum_m Delta_m^d = 0.
```

For integrable pathwise gradients, fixed targets under this conditioning give

```
E[partial_theta B | C]
 = -(1/(2Mn)) sum_d Jbar_theta,d(C)^T sum_m vec(Delta_m^d)
 = 0.
```

Thus E[partial_theta(F+alpha A_r)|C] equals E[partial_theta(F+alpha A_c)|C]. Independence **between members** is unnecessary. Identical member laws, correlated equal-law masks, and shared equal-law masks can all satisfy the condition. The sufficient stochastic symmetry is a matching joint law for each member's complete two-view forward calculation and shared score Jacobians, at a fully symmetric state. Direct equality of the expected Jacobians is weaker and is sufficient on its own.

At unit factors, this law follows only if all other route-local parameters, maps, buffers, normalization states, stochastic mechanisms, sample weights and view-pairing rules also match and target assignment does not alter the forward/noise law. For a coupled-member operator, permutation equivariance and exchangeable stochastic state can still supply symmetry; coupling by itself is neither a proof nor a refutation. A route-specific forward transformation, RNG law or buffer can break it. Different sampled masks with the same law do not.

The proof concerns expected raw derivatives, so it does not need to interchange differentiation and expectation. Identifying them with gradients of expected losses requires the usual differentiability/domination assumptions. Equal full score laws also make E[A_r]=E[A_c] by linearity. Equality of expected Jacobians alone establishes the gradient identity, not necessarily equality of loss values. Cosine normalization must be defined and the derivatives integrable; zero vectors or nonsmooth boundaries need the implementation's stated convention.

At identical realized scores and shared Jacobians, cancellation holds exactly before expectation. Equal realized scores alone do not imply equal parameter gradients. Under independent dropout streams it is usually the expectation identity, not realized equality, that is available at a symmetric state. After private factors diverge, their shared Jacobian laws can diverge; no induction preserving this identity has been established.

### Private derivatives can differ, but need not steer anything useful

For a route-local private block,

```
E[partial_phi_m (alpha B)|C]
 = -(alpha/(2Mn)) sum_d E[J_phi_m,m,d|C]^T vec(Delta_m^d).
```

There is no cancellation across m in the stacked vector of separate private parameters. If the block coordinate systems are identified, their summed expected differences can cancel under symmetry, but that does not make every block zero. A nonzero target contrast may also lie in a private Jacobian nullspace, or in the antisymmetric target nullspace above. Different target masks do not guarantee nonzero private gradients.

The retained exact collapse example gives another failure: nonzero embeddings within each class are all parallel in both views. Same-class positive cosine is 1 with zero vector derivative. Every route puts total target mass 1 in that class, so changing positive incidence produces no target-dependent embedding or parameter gradient. The target-independent denominator contribution remains. The actual model is not diagnosed as occupying this state. Hidden changes that do occur can also lie in a classifier nullspace or harm a truth-versus-competitor margin; own CE supplies supervision without a competence guarantee for the composed update.

## 3. What can break the expected identity

1. **Unequal Jacobians.** For one target-row contrast, let q_1=qbar+a and q_2=qbar-a, a nonzero, with corresponding score-contrast Jacobians u and v. The shared difference is proportional to -a(u-v), which is nonzero when u differs from v. Equal target means cannot cancel unequal pullbacks. This is a symbolic coordinate witness, not a native architecture diagnosis.

2. **Equal view marginals but unequal joint view laws.** There is an exact cosine-coordinate illustration. Let xi,eta each have symmetric signs, choose 0<a<pi, and use unit vectors

   ```
   h^1(theta,xi)=(cos(theta xi), sin(theta xi)),
   h^2(eta)=(cos(a eta), sin(a eta)).
   partial_theta S|theta=0 = xi eta sin(a)/tau.
   ```

   In one member set eta=xi, in another eta=-xi. Both individual view laws match, but the expected score Jacobians are +sin(a)/tau and -sin(a)/tau. This defeats the required Jacobian symmetry. It is an analytical score example, not an executed fixture or a claim about native dropout.

3. **Target-dependent noise, sampling or moving targets.** If Delta_m is random and correlated with J_m, the relevant expectation is E[J_m^T Delta_m]; E[J_m] equality does not justify factoring it. Conditioning on fixed common inputs is valid only if the matching conditional law survives. A fixed signature target can be correlated with public inputs without breaking the proof, because those inputs and all targets can be conditioned on together. A target-dependent augmentation law generally cannot be ignored. Differentiating an unstopped learned target introduces additional derivatives.

4. **Different reductions or common-target definitions.** Unequal member/row weights require the corresponding weighted target mean and matched directional weights. A single uniform Qbar does not cancel arbitrary recipient weights. Re-normalizing the common target after averaging can change it. Dropping rows or candidates differently changes the compared loss.

5. **Broken state symmetry or undefined gradients.** Route-local parameters/buffers, target-conditioned operators, asymmetric coupled-member feedback, nonintegrable derivatives, or unspecified zero-vector cosine conventions can invalidate an assumption. The report establishes the conditional algebra, not native satisfaction of every condition.

## 4. Zero mean difference is not a variance or Adam guarantee

Couple the two policies through the same current state and realized stochastic views, and let g_r=g_c+delta g be their shared raw gradients. At the symmetric state E[delta g]=0. For finite second moments,

```
Cov(g_r) = Cov(g_c) + Cov(delta g)
           + Cov(g_c,delta g) + Cov(delta g,g_c).
```

The cross covariance can have either sign. Common targets do not necessarily lower shared variance, and route targets do not necessarily raise it. No conditioning, signal-to-noise, stability or generalization advantage follows from zero mean alone. Neither the coupling nor a variance measurement has been executed here.

Adam applies nonlinear, history-dependent normalization. Even a first step from zero moments can distinguish distributions with the same mean. A scalar illustrative distribution is g=1 with probability 2/3 and g=-2 with probability 1/3, while the other gradient is identically 0. Both means are 0. With bias correction and epsilon>0,

```
E[g/(|g|+epsilon)] = 2/[3(1+epsilon)(2+epsilon)] > 0,
```

whereas the identically zero case updates by zero. The parameter displacement has the corresponding negative step sign. This generic optimizer counterexample is not an embedding of the native contrast model. Later moments, gradient clipping, decay and finite curvature add further distinctions.

Even exact equality of a shared raw gradient at a matched state does not force equal shared optimizer input after **clipping based on the norm of all parameter gradients**: the private gradient can change the common scaling factor. If shared optimizer history is equal and the transition acts separately on coordinates/blocks with no such coupling, identical shared gradients do give an identical immediate shared transition. Across independently trained policies, private states, shared moments and forward states generally differ.

## 5. The candidate and its three necessary recipient controls

Define G_c=F+alpha A_c and G_r=F+alpha A_r. All four policies collect derivatives from the same pre-update parameter/buffer state and the same realized views:

| Policy | Shared theta receives | Private phi receives |
| --- | --- | --- |
| Common-all | partial_theta G_c | partial_phi G_c |
| Route-all | partial_theta G_r | partial_phi G_r |
| **Common-shared / route-private candidate** | partial_theta G_c | partial_phi G_r |
| Mirror recipient control | partial_theta G_r | partial_phi G_c |

Common-all tests ordinary aggregate context alignment. Route-all tests the already prepared persistent route assignment through every reachable parameter. The candidate tests the proposed recipient allocation. The mirror tests whether that specific allocation matters. At the same state/views, the candidate has the shared gradient of common-all and private gradient of route-all; the mirror has the converse. All four are different from a context-cycling schedule. No extra trained arm is authorized here.

A valid future source implementation would collect the required partials from current forward graphs before one optimizer transition. Updating theta first, drawing new views for phi, or advancing stochastic buffers twice would implement another policy. Detaching the hidden representation destroys upstream r/s paths and generally does not implement this table. The partition, auxiliary endpoint, reductions and clipping location need to be explicit; recipient protection is a statement about supplied partials only.

Direct block-gradient collection is conceptually sufficient. A stop-gradient functional construction can also reproduce the first-order field when parameters are substituted correctly. With fixed reference tensors (theta_0,phi_0) equal to the current state, the ephemeral surrogate

```
S(theta,phi;theta_0,phi_0)
 = F(theta,phi) + alpha A_c(theta,phi_0) + alpha A_r(theta_0,phi)
```

has the candidate's partials at that state. Both auxiliary terms must use the same stochastic realization and frozen buffers. Refreshing the reference tensors at each step changes the surrogate. Its forward scalar value is not G_c or G_r, and treating its frozen-reference Hessian as the derivative of the refreshed field omits cross-state dependence. Ordinary stop-gradient syntax therefore supplies neither a single true potential nor a global-descent argument.

### Scalar-objective and local-descent limits

Common-all and route-all are raw gradients of G_c and G_r. The candidate field is

```
V = grad G_c + alpha P_phi grad B,
partial_phi V_theta - (partial_theta V_phi)^T = -alpha H_theta,phi B.
```

In a smooth simply connected neighborhood, an exact Euclidean potential requires that mixed block to vanish throughout the neighborhood (alpha is fixed). If B is locally separable, a potential can exist; a nonzero mixed entry rules it out locally. Zero at one point is insufficient. The mirror has the opposite cross-partial mismatch. Nonpotentiality alone establishes neither instability nor harm, and does not exclude a Lyapunov function or special well-behaved cases.

For simultaneous small Euclidean SGD steps write a=partial_theta G_c, b=partial_phi G_c, c=alpha partial_theta B, d=alpha partial_phi B. The first-order changes under the candidate are

```
Delta G_c = -eta_theta ||a||^2 - eta_phi b^T(b+d),
Delta G_r = -eta_theta (a+c)^T a - eta_phi ||b+d||^2.
```

Either can be positive. Even decrease of an alignment-inclusive loss does not certify CE decrease, margin improvement or competence. These equations are local raw-gradient calculus, not native Adam descent or a new theorem. No projection, lookahead optimizer, acceptance guard or new guarantee is added.

## 6. Retained priors and the strength of the claim

| Retained scope | Relation to this candidate |
| --- | --- |
| GNCL v2, saved section 4.1/Eq.5 and prior author-code conclusions | Scalar mean-own/pool-risk interpolation is established. This candidate uses own CE on all blocks and allocates two established contrast targets differently. It introduces no new ensemble-risk identity. A scalar mixture G_c+beta alpha B gives the same beta to both blocks, whereas the candidate gives 0 to theta and 1 to phi; they differ generically. B need not be a nonnegative Jensen gap. |
| Differentiable Game Mechanics v1 and retained mixed-block notes | Simultaneous partial gradients, cross-partial potential criteria and block games are established. The displayed field is an application of that framework, not discovery of block optimization. |
| PCGrad v1, retained method scopes | Altering supplied task gradients is established in multitask learning. Its conflict projections differ from this fixed recipient assignment; its guarantees do not transfer. The common-shared/private-specialist motivation is a conventional shared-representation/private-task hypothesis, not a new multitask principle. |
| Song and Chai 1805.11761v1, saved sections 3.1–3.3 | Shared intermediate layers, private hard/peer supervision and shared-backward rescaling are direct routing ancestry. Exact targets, reductions and deployment differ. |
| ONE v2 / PCL v2, retained method conclusions | Shared/private supervision and architectural partial loss exposure are close ancestry. PCL's peer classifiers bypass a separate ensemble hard CE. Extra distillation, fusion/gates, EMA, reductions and serving rules differ; native detach semantics remain unqualified. |
| Retained detached-weight CE note and generic stopped-reference construction above | Stop-gradient losses can reproduce specific first-order pullbacks; scalar values and higher-order/trajectory semantics can differ. Expressibility with detach is not a novelty argument or a descent guarantee. |
| Saved SupCon/BotSCL, graph high/low-pass context and BE/TabM conclusions | Weighted same-class cross-view alignment, graph context views and shared/private factors are attributed ingredients. This assessment does not reread their primary bodies or establish exact complete-rule duplication. |

The earlier shared-pool/private-own CE split exists locally as a saved short comparator. It is not identical to this common/route alignment split, but establishes local gradient-allocation ancestry. The bounded modern rank-one report found adjacent efficient-ensemble methods and did not certify an exhaustive published absence. No exact complete published duplication of this candidate is established here; neither is novelty. Generic multitask or stop-gradient comparisons are conceptual operation comparisons, not new primary literature credits.

## 7. Why source preparation is deferred

The candidate preserves ordinary label gradients in every block and confines heterogeneous alignment targets to private capacity. That is a plausible inductive-bias hypothesis. Heterogeneous private alignment can also emphasize nuisances, erase through Jacobians, damage members, or drive later shared changes through altered representations. Common shared alignment can remove useful route-derived shared credit. Neither direction is favored by the derivation.

At the symmetric unit start, route-all and the candidate differ in a zero-mean shared raw contribution; route target assignment can already supply private symmetry breaking under route-all. Therefore “keep a common shared mean while differentiating private factors” does not itself identify an unavailable function in the current scalar method. It selects stochastic and later state-dependent credit differently. That difference is real, but its value remains an attribution question.

A later narrow preparation could be justified if useful private route steering is established and a specific shared-credit problem remains: realized shared noise harmful under the native transition, or later shared pullbacks that erase useful specialization or hurt CE/serving. Those are possible diagnoses, not assumptions to smuggle into the rationale, and no such measurement is made in this task. A positive route/common difference alone would not identify them; failure of both would not establish a rescue by splitting. Mirror/common-all/route-all remain necessary controls for any allocation claim, with cost and normalization disclosed.

Retain this mathematical specification without adding it to running source, a study, protocol permissions or ledgers. Do not use it as a new fallback, claim competence protection, or expand compute solely because the shared/private field has a formal interpretation.

## Scope and verification

Only retained scientific/method reports were read. No new network search, primary paper reread, model/data/checkpoint/target tensor/current outcome access, import of numerical/model libraries, executed numerical fixture, fit, GPU/server/heldout access, agent spawn, or source/protocol/index/ledger modification occurred. Some historical scalar summaries were exposed incidentally by a keyword scan of a retained report; none informs this derivation or disposition, and the underlying results remain unopened. The saved CPU helper is mentioned only as an existing limited artifact, not executed or adopted.

The output is limited to this new assessment folder. JSON parsing, file hashing and manifest consistency checks are bookkeeping only. The two-view proof, failure constructions, potential criterion and optimizer counterexample are hand algebra, with explicit assumptions and no native outcome claim. Input bindings and read scopes distinguish resumed-note knowledge from targeted rereads.
