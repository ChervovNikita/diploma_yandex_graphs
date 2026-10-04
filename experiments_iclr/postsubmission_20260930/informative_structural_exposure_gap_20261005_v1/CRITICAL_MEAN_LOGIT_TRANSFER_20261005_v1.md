# What the three-edge matching loss can change at native serving

5 October 2026. Bounded critical synthesis using the saved assessment and index_v57 conclusions. No new retrieval, model execution, numerical experiment or edits to existing records.

**The proposed mixture can improve its only extra fixed-marginal assignment dependence while leaving every served mean logit unchanged.** Direct use of native target scorers removes a separate auxiliary-decoder escape route, but does not remove this ensemble-contrast escape route. The mixture can also change ordinary link interactions through training; that effect is a responsibility-weighting hypothesis requiring predictive evidence against identical-bundle single and independent controls. The current design is not an established method beating those controls or an established novelty.

## The extra dependence is absent from mean-logit serving

Write `S_m` for member `m`'s nine native link logits on the common masked graph, `P_pi` for a permutation-incidence matrix and

```
E_m(pi) = <S_m,P_pi>,
q_m(pi) = exp(E_m(pi))/Z_m,
q_J = mean_m q_m,
S_bar = mean_m S_m.
```

Let `epsilon_pi` be permutation parity, with three signs of each kind. Every edge cell occurs in exactly one even and one odd permutation, so

```
sum_pi epsilon_pi * E_m(pi) = 0,
sum_pi epsilon_pi * log q_m(pi) = 0.
```

The hypothetical matching law obtained from `S_bar` is

```
q_G(pi) = softmax_pi(<S_bar,P_pi>)
        = normalized geometric mean_m q_m(pi).
```

It satisfies the same zero log-parity relation. It is an additive assignment law and is maximum entropy for its own row marginals. Its row marginals need not equal those of the arithmetic mixture `q_J`. This derived law is a way to describe what the served mean score matrix can encode; no matching readout is being added at serving.

The arithmetic mixture can violate the log-parity relation. Its one extra degree of freedom at fixed row marginals is exactly the parity direction removed by `q_ME`. That dependence is absent from `q_G`. A richer arithmetic assignment law therefore does not itself give a richer mean-logit ranker. This is established structured exponential-family and mixture algebra, not a new mathematical principle.

## An exact failure path

Consider one fixed bundle with observed identity matching. Let `P_0,P_1,P_2` be the three even permutation matrices and `J` the all-ones matrix. For any `t>0`, choose four member score matrices

```
S_0=t P_0, S_1=t P_1, S_2=t P_2, S_3=-t J.
```

Because `P_0+P_1+P_2=J`, `S_bar=0` throughout this path. All nine served bundle logits remain tied. Cyclic symmetry gives every `q_J` row marginal exactly `1/3`; hence `q_ME` is the uniform six-matching law throughout the path.

Put `x=exp(t)` and `Z=x^3+3x+2`. A member favoring an even permutation gives that permutation weight `x^3`, the other two even permutations weight `1` each, and the three odd permutations weight `x` each. The fourth member is uniform. For any even permutation, including the observed one,

```
q_J(even) = ((x^3+2)/Z + 1/6)/4
          = 1/6 + (x-1)^2(x+2)/(8Z) > 1/6.
```

Thus `L_J` improves strictly over the common-zero-score state while `L_ME` and all served bundle rankings remain unchanged. The improvement is pure fixed-marginal parity. By continuity, a sufficiently small additional common score perturbation can worsen a chosen served positive-versus-alternative margin while leaving this strict auxiliary gain intact.

This is an output-level counterexample. It does not assert that a particular native factorized network can realize this entire path across all bundles. A native competence loss can oppose it. Neither fact permits an implication from improved joint fit to improved serving. The example has no fitted outcomes or chosen training amplitude.

## How improvement remains possible, and what must be shown

Let `Y` be the observed matching-incidence matrix, `mu_m=E_{q_m}[P_pi]`, and

```
gamma_m = q_m(observed) / sum_l q_l(observed).
```

The joint loss gradient with respect to member scores is

```
dL_J/dS_m = gamma_m * (mu_m-Y).
```

These gradients do reach the native link scorer. They can change common score interactions as well as private contrasts; it would be incorrect to claim that every joint-loss gradient is invisible. The extra dependence can alter which member's ordinary interaction correction receives weight. That is the remaining possible learning benefit. There is no guaranteed useful specialization or extra served parity channel.

For an actual served positive-versus-negative query, define its native mean-logit margin

```
D(theta) = mean_m f_m(G_serve,positive)
         - mean_m f_m(G_serve,negative).
```

At a differentiable model state, a small parameter displacement `delta_theta` improves this margin to first order exactly when `grad D dot delta_theta > 0`. For a fixed illustrative SGD step on the auxiliary, this condition is `grad D dot grad L_J < 0`. Relative to a matched `L_ME` step, the required sign is `grad D dot (grad L_J-grad L_ME) < 0`. An actual Adam comparison must use its actual state-dependent parameter displacement; the SGD formula is not an optimizer equivalence claim.

If the joint-minus-marginal displacement lies in the nullspace of all relevant served-margin derivatives, its extra fit has no first-order serving effect. If its dot product has the opposite sign, it hurts that margin. A benefit requires the joint reweighting to change target-relevant mean interactions in the useful direction **under the native serving graph**, and to transfer from selected hidden TRAIN matchings to genuine future/unobserved links. Using the same scorer on the different masked graph does not establish that transfer. Correct sums over three edges also do not ensure that the anchored single link is ranked better.

These conditions specify the unresolved mechanism. They are not demonstrated alignments, measured gradients or quality thresholds.

## Independent training receives the same structural information

Four ordinary models given the identical bundles and masks, each trained on its own matching NLL, have

```
L_own = mean_m [-log q_m(observed)],
dL_own/dS_m = (mu_m-Y)/M.
```

The signed link constraints are identical: increase the observed diagonal logits and decrease the competing off-diagonal logits. The mixture supplies no extra TRAIN label, structural witness or alternative assignment. It changes the member's whole-bundle coefficient from `1/M` to `gamma_m`. Members with low responsibility receive less of this correction than under ordinary independent training; the ordinary objective trains every member on the complete matching.

For unconstrained score coordinates on this same masked bundle, both losses move every diagonal-versus-off-diagonal mean-score margin in the correct direction in an isolated gradient step. There is no universal ordering of their total gradient magnitude or predictive benefit. With network parameter coupling and changed serving context, even that direct coordinate argument is insufficient. What can be said precisely is that ordinary training has the same observations and an unsuppressed correction for each model. A weak ordinary control would manufacture an apparent ensemble-specific advantage.

Any proposed advantage is consequently an optimization/finite-sample effect of responsibility reweighting and possibly parameter tying. An untied joint-loss bank can apply the same reweighting. A capable structured single can also learn the complete-assignment dependency, although its auxiliary dependency likewise need not improve its native link ranking. No function-class superiority of a tied bank follows.

## Literature synthesis and a decision that avoids wasted fits

The saved scopes already establish the relevant operations: Maslov–Sneppen supplies degree-preserving switches; HeaRT supplies endpoint-related hard alternatives and the temporal negative-filtering caveat; MaskGAE supplies structured masking whose supervision reaches a served encoder; GRAN supplies complete-block mixture responsibilities. The saved *Joint Training of Deep Ensembles Fails Due to Learner Collusion* scope supplies a close generic warning about optimizing pooled competence. None establishes that the proposed arithmetic matching law improves the current raw-logit serving rule. CAM/PIFM further rule out a claim that coherent graph configurations themselves are new. These conclusions are reused at their saved scopes; no primary bytes were reread for this note.

**Do not allocate a new predictive cohort whose premise is that the extra parity is a served representational advantage.** That premise is false for the declared mean-logit pool. Keep the loss as an untested attributed training option, rather than promoting a source-qualified six-assignment module to a novel quality method.

If the bundle observation is pursued later, ordinary independent own-matching training must receive the identical bundles from the first predictive comparison; it already receives every new structural constraint. A claimed mixture benefit must be framed narrowly as improved served ranking from responsibility reweighting, with the capable single and untied joint control able to defeat it. Auxiliary `L_J` improvement over `L_ME`, informative-mask prevalence or a nonzero score gradient cannot justify additional fits on their own. Current frozen cohorts and the existing capable controls remain the priority.

The useful progress is a concrete corruption and a clean dependence control, plus identification of a serving-null failure. It is presently an implementable hypothesis with an exposed weakness, not the requested evidence of a novel method beating competent singles and ordinary ensembles.
