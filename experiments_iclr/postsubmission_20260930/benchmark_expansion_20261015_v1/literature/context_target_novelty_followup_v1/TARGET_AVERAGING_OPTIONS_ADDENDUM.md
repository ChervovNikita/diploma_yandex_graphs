# Analytic addendum: alternative ways to combine factual targets

Requested follow-up only. **No method, arm, source recipe or run is changed.** The original report/derivation/manifest are preserved. These options clarify what a later different control would mean; neither is an automatic replacement for current COMMON.

## 1. Exact source binding and notation

The frozen `benchmark_expansion_20261015_v1/context_affinity_v1/core.py` has SHA256 `c510c0dbf02236c6345df99d2906534866cab9a65c8562557994e907303baa3b`.

| Lines | Definition implemented |
| --- | --- |
| 12–14 | tau=.2, auxiliary coefficient=.05, normalization epsilon=1e-12 |
| 55–58 | Same observed class/relation; distinct object; eligible if at least one donor |
| 73–74 | z_mi=normalize(h_mi), shared compatible mask |
| 76–82 | Detached u^r_i=normalize(c^r_i), then q^r_ij=softmax_j((u^r_i dot u^r_j)/tau) on that mask |
| 85–89 | log p_mij=logsoftmax_j((z_mi dot z_mj)/tau), with excluded entries set to zero only after softmax for safe CE multiplication |
| 90–91 | COMMON qbar=(qA+qB)/2; SPLIT qA for m<2 else qB; row CE=-sum_j q_ij log p_mij |
| 92–108 | Source-specific row reduction, then uniform four-route mean and .05 coefficient |

`context_affinity_node_port_v1/extension.py` lines149–160 binds existing four live routes, unchanged native F, the selected TRAIN preheads, F+extra, one backward and one native optimizer step. The molecular bridge lines149–164 does the corresponding existing-route/observed-label construction and update. Exact selected-line hashes and source receipts are in `SOURCE_LINE_BINDINGS.json`.

In a compatible block, let G^r_ij=u^r_i dot u^r_j and K^r_ij=exp(G^r_ij/tau) for admitted i!=j, zero outside support. The support is the same symmetric mask for A, B and p. Set Z^r_i=sum_j K^r_ij, so q^r_ij=K^r_ij/Z^r_i. All following equations refer to this same bounded cohort and exclude diagonals.

## 2. Arithmetic affinity first, then row normalization

The proposed alternative is

    Ktilde = (KA+KB)/2,
    qtilde_ij = Ktilde_ij / sum_k Ktilde_ik
               = alpha_i qA_ij + (1-alpha_i) qB_ij,
    alpha_i = ZA_i / (ZA_i+ZB_i).

Ktilde is symmetric, so qtilde is reversible. This removes the precise three-cycle obstruction demonstrated for the current average of conditionals. It retains the same factual channels and donor support and introduces no need for new inference inputs.

It is **not the same equally weighted context target**. Row partition sums determine an anchor-dependent channel weight. A channel with more/larger within-class affinity mass has more influence. Multiplying a complete channel's unnormalized kernel by a positive constant would leave that channel's q unchanged but change qtilde; the fixed cosine/temperature convention therefore becomes part of the between-channel scale contract. There is no permission here to select that scale from outcomes.

Reversibility is insufficient for cosine realizability. In the ordinary unit-vector regime, exact realization on a connected block would require, for some common additive constant b,

    G_ij = tau*log(Ktilde_ij) + b  for i!=j,
    G_ii = 1,
    G positive semidefinite, rank(G)<=native H.

The global b reflects a common kernel scale that cancels in row normalization. A log of a sum of exponential Gram kernels is not generally guaranteed to satisfy the required unit-Gram/rank conditions. Epsilon-clamped short vectors add boundary cases; they do not remove the symmetry, Gram or native-mapping constraints. Thus this option repairs reversibility only, not established native representability.

Because mean split target remains (qA+qB)/2, replacing only COMMON by qtilde breaks the original aggregate-target and mean-score-cotangent match. The difference is qtilde-qbar=(alpha_i-.5)(qA-qB). Equal shared Jacobians would no longer cancel it in general.

One can algebraically restore aggregate credit for this alternative by multiplying each A-route auxiliary row by 2alpha_i and each B-route row by 2(1-alpha_i), keeping COMMON's row credit unchanged. Across four routes the weighted total row mass stays one and the effective split target is qtilde. But that changes SPLIT's route/anchor credit and per-route exposure law; it is another method definition, not a free repair or an authorized new arm. Ordinary per-route renormalization would generally destroy that exact equality again.

## 3. Normalized geometric mean, equivalently average cosine scores

For positive probabilities on admitted donors,

    qstar_ij = sqrt(qA_ij*qB_ij) / sum_k sqrt(qA_ik*qB_ik)
              = softmax_j((G^A_ij+G^B_ij)/(2tau)).

The row partition functions cancel. The corresponding symmetric kernel is sqrt(KA*KB), so it is reversible. If both normalized context vectors are unit length for every object, concatenate

    v_i = [u^A_i ; u^B_i] / sqrt(2).

Then ||v_i||=1 and v_i dot v_j=(G^A_ij+G^B_ij)/2. This provides an exact unrestricted cosine-Gram realization in dimension D_A+D_B (or the rank of the concatenated feature matrix). It is a stronger representability guarantee than the arithmetic affinity mixture.

### Zero and epsilon-short context vectors

The actual normalization yields ||u^r_i||<=1, with norm zero for a zero context and possibly less than one below epsilon. Then the naive concatenation v_i can be short. Normalizing it to length one generally changes the intended average score. A unit-vector existence construction is

    vhat_i = [v_i ; sqrt(1-||v_i||^2) e_i],

where e_i are mutually orthogonal object-specific directions for the objects needing padding. Every vhat_i has norm one; off-diagonal dot products, and hence all admitted target probabilities, remain unchanged. The extra directions may raise the required dimension by up to the number of deficient objects. This is a finite-cohort mathematical construction, not a proposed ID feature, extra projector, model input or permissible implementation change. It does not show a native shared mapping will generalize to new objects.

For an anchor whose A context is zero, qA is uniform. Then qstar is proportional to sqrt(qB), equivalently B scores at temperature 2tau. That differs from both the current half-uniform mixture and the affinity-first mass weighting. If both anchor contexts are zero, the target stays uniform.

### What information matching changes

qstar is a consensus of score evidence. It is the minimizer of .5KL(p||qA)+.5KL(p||qB), whereas qbar is the minimizer of .5KL(qA||p)+.5KL(qB||p), over unconstrained row distributions p. These are different summary objectives. Both retain positive mass on the same support, but they combine conflicting affinities differently. There is no universal entropy ordering that makes their task difficulty equal.

The original split targets still average to qbar, not qstar. Equal average log-target scores do not imply equal CE score cotangents, because CE is linear in target probabilities. Replacing only COMMON by qstar therefore loses the original target-mass/shared-gradient cancellation identity. Using .5CE(qA,p)+.5CE(qB,p) instead merely returns the existing arithmetic-target objective; it does not implement the geometric alternative.

## 4. Can these options isolate the mismatch?

| Property | Current conditional arithmetic mean | Arithmetic affinity first | Geometric/score mean |
| --- | --- | --- | --- |
| Same factual input channels and donor support | Yes | Yes | Yes |
| Matches uniform 2A/2B split target mass | Yes | Generally no | Generally no |
| Reversible target guaranteed | No | Yes | Yes |
| Unrestricted unit-cosine realization guaranteed | No | Not established by the mixture construction | Yes: concatenation, with orthogonal norm completion if needed |
| Native dimension/shared mapping realization guaranteed | No | No | No |
| Current frozen method | Yes | No | No |

The geometric option is the cleaner analytic way to remove the unrestricted Gram obstruction; the affinity-first option removes only its reversibility component. Both could define meaningful later questions about fitting one common context summary versus specialized channels. They do not isolate only feasibility while preserving the present aggregate-target match.

There is a simple incompatibility: if every COMMON route uses one q_common and the unchanged two/two SPLIT bank has mean target qbar, exact aggregate-target matching requires q_common=qbar. If that qbar is nonreversible, no alternative reversible common distribution can satisfy both requirements. One must change the question/credit/targets or accept the original attribution limit.

No current pilot or roster should be edited from this analysis. First interpret existing outcomes with target entropy, support and realizability diagnostics. Any later option needs its own explicit scientific question, source freeze and cost/selection contract under the standing bounded-pilot authorization. No predictive evidence or novelty verdict is supplied by the algebra.
