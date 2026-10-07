# The current internal-risk allocation as a block game

7 October 2026. Independent theory/source note only. No model import, numerical fixture, training, graph data, checkpoints, current outcomes, private-server access, extragradient implementation, active Mol18 change, or root-ledger/status change. The equations below were derived by hand. They describe raw supplied gradients, not the full native Adam dynamics.

**Finding:** I is generally a nonpotential differentiable block game in the exact Euclidean sense. The obstruction is the mixed external/internal curvature of the risk gap, with a further contribution from the restricted alignment loss. This interpretation is established game/gradient-allocation mathematics applied to the current policy. It is neither a new optimization principle nor evidence that I is unstable or more accurate in the actual model.

## 1. Exact field and the potential condition

Combine shared/body parameters theta and prediction boundaries psi into x; denote the designated internal factors by y=phi. Let F be exact mean-member risk, L the declared served-pool risk, D=F-L, and A the source alignment loss. The source uses alpha=.05 and a fixed lambda, with

```text
J = (1-lambda)F + lambda L = F-lambda D
V_I = (F_x, J_y + alpha A_y)
    = (F_x, F_y-lambda D_y+alpha A_y).
```

V is the positive supplied-gradient field; a raw SGD transition would subtract it. x can be treated as one player minimizing F and y as another minimizing J+alpha A. This is a general-sum differentiable game with explicit block permissions.

[method.py](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/public_internal_be_allocation_controls_20261007_v1/method.py:66>) forms the two view-specific risks before averaging views, then collects all block gradients at the same old parameter state. Lines 109–112 give x the own gradient and y the mixed-plus-alignment gradient. There is no upstream detach or theta-first transition. These details are necessary for the displayed field.

Fix data, the two realized stochastic views, parameter roles and a smooth activation region. On a simply connected smooth neighborhood, a scalar potential Phi with V_I=grad Phi exists exactly when its cross-partials agree:

```text
d_y V_x - (d_x V_y)^T = lambda D_xy-alpha A_xy = 0.
```

The within-block Hessians are already symmetric. With the auxiliary omitted, lambda>0 requires D_xy=0 throughout that neighborhood. Locally separable D admits a potential, for example Phi=F-lambda D_y_only(y). Therefore “I differs from grad J” alone is insufficient to show that no other potential exists. A single nonzero cross entry rules out an exact potential on a neighborhood of that point; zero curl at one point does not certify a neighborhood.

This is the unweighted Euclidean criterion. A fixed positive block-weighted potential would instead require weighted cross blocks to agree throughout the neighborhood. Special proportional cases can satisfy it, and a Lyapunov function can exist without any exact potential. ReLU boundaries require piecewise/weak analysis rather than an unqualified C2 argument. The expected stochastic field obeys the corresponding condition only when differentiation and expectation may be interchanged; individual-view curl need not equal expected curl.

**The auxiliary matters to the control interpretation.** G's supervised field is grad J, but the actual full G field is `grad J+alpha P_phi grad A`. It too is generally nonpotential if A_xy is nonzero. Full O has the same restricted-auxiliary issue. The incremental I-versus-O field is `-lambda P_phi grad D`; its extra cross-curvature is lambda D_xy, regardless of the matched auxiliary. Thus a finding of curl in I alone would not identify the internal pool-risk contribution.

## 2. A direct nonpotential witness under mean-logit BCE

Use four affine member logits `z_m=x+y_m`, where x is shared and y_m are private coordinates. For binary loss, put `a(z)=sigmoid(z)*(1-sigmoid(z))`. With no auxiliary, the source mean-member/mean-logit risk contract gives

```text
d_{y1} V_x = a(z1)/4
d_x V_{y1} = [(1-lambda)a(z1)+lambda a(mean z)]/4.
```

At x=0, y1=0 and y2=y3=y4=t with finite t nonzero, their difference is `lambda*[1/4-a(3t/4)]/4`, which is strictly positive for lambda>0. This is an exact legal loss/role witness, independent of the binary label. It proves that the allocation can fail the potential condition without a new loss or adversarial objective. It is an affine toy family, not a curvature estimate or exact architectural embedding of the native model.

The saved 3 October mixed-objective scout already gave a related cross-partial witness for the reverse shared-pool/private-own policy. The present conclusion specializes that established reasoning to internal-only mixture allocation and its actual auxiliary permissions.

## 3. Exact local instability is possible even with convex risks

Nonpotentiality alone gives no stability verdict. A stronger existence counterexample uses the current four-member binary mean-logit pooling contract and the already supplied illustrative lambda=.5. Omit A to isolate supervised allocation. Give each of two otherwise identical examples a different hard label, 0 and 1; their averaged binary loss is

```text
b(z) = softplus(z)-z/2
z1 = (13/4)x+4y;    z2=z3=z4=x/4
F = b(z1)/4 + 3b(x/4)/4
L = b(x+y);        J=(F+L)/2.
```

Here y influences only member 1. b is convex, b'(0)=0 and b''(0)=1/4. F, L and J are convex; F and J have the same unique minimum at (0,0), with positive-definite Hessians there. D is the exact nonnegative Jensen gap. Nevertheless the Jacobian K of the supervised I field `(F_x,J_y)` at that minimum is

```text
K = (1/64) [[43,52],[34,40]]
det K = -3/256
eigenvalues K = (83 +/- sqrt(7081))/128.
```

One eigenvalue is negative. The linearized gradient flow `w_dot=-V(w)` has an expanding direction. Simultaneous raw SGD has multiplier `1-eta*kappa>1` on that direction for every eta>0. The exactly initialized minimum remains a fixed point; nearby perturbations can grow. Along its unstable direction, both convex risks initially increase. Convex own/pool ingredients, a nonnegative gap, and each block's individual descent direction do not ensure stability of their simultaneous composition.

This is a hand-derived existence example, not an observed pathology of Mol18, WikiCS or the adapter. It contains one internal coordinate and no auxiliary, other native parameters, dropout, optimizer history or weight decay. It directly matches binary mean-logit BCE; the WikiCS probability-pool contract needs its own local dynamics analysis. Scaling both risks by the same positive reduction factor, as a positive-plus-negative group convention can do in this balanced toy, preserves the instability sign.

## 4. Stability facts that can actually be used

For a smooth raw field at a stationary point, positive real parts of all eigenvalues of K imply local asymptotic stability of gradient flow. A negative real part gives an unstable direction. Simultaneous raw SGD additionally requires `abs(1-eta*kappa)<1` for every mode; purely imaginary modes are not asymptotically damped by this update. Zero-real-part modes require more than linearization. The symmetric/antisymmetric decomposition of K is useful, but antisymmetric magnitude alone is not a stability test. A uniformly positive-definite symmetric part is a sufficient local contraction condition; its failure is not itself proof of instability.

Nonpotential fields can also be well behaved. If L is independent of y and alpha=0, I is a positive block-scaled gradient of F for lambda<1. Then F decreases along the continuous flow even when the raw Euclidean Jacobian is asymmetric. This special case prevents treating nonpotentiality as inherently harmful.

For the general supervised I flow, the own-risk derivative is

```text
F_dot = -norm(F_x)^2-(1-lambda)norm(F_y)^2-lambda F_y^T L_y.
```

The cross term can reverse its sign. There is likewise no automatic decrease of J or L under simultaneous block updates; the restricted auxiliary adds further cross terms. A block can decrease its assigned risk with the other blocks fixed while the complete transition increases both risks.

An extragradient lookahead is not a generic remedy for a nonpotential field. For a constant linear mode kappa its multiplier is `1-eta*kappa+eta^2*kappa^2`. A negative real kappa, as in the counterexample, still has multiplier greater than one for every positive eta. Pure rotational modes present a different case where a suitably bounded lookahead step can damp the discrete oscillation. This distinction matters before choosing an optimizer study.

The adapter commits the collected gradients through native Adam, whose displacement depends on moments, second-moment scaling, epsilon and optimizer groups. Its state includes that history. Raw-field symmetry and SGD eigenvalues cannot be relabeled as stability guarantees or exact local multipliers for native Adam. State-dependent preconditioning can also change whether a displacement is a Euclidean gradient. None of the displayed results establishes global descent or global convergence.

## 5. Closest prior support and limits

| Prior and inspected scope | What it establishes here |
| --- | --- |
| [GNCL v2](https://arxiv.org/html/2011.02952v2#S4.E5), saved exact §4.1/Eq.5 passages | F/L interpolation is established. Its scalar supervised objective gives all blocks the derivatives of J. Allocating that derivative to only phi changes the field, rather than introducing a new ensemble risk. Saved Taylor-remainder cautions remain applicable; the exact Jensen gap here is not its remainder-dropped Hessian quadratic. |
| [Differentiable Game Mechanics v1](https://arxiv.org/html/1905.04926v1#S2), Definition 1; §§2.3–2.5 definitions and potential characterization | Simultaneous block derivatives, asymmetric Jacobians, symmetric/antisymmetric decomposition, and distinctions between local Nash points and stable dynamics are established. The cross-partial criterion is an application of this framework and elementary calculus. The inspected statements are not a certification of all author convergence proofs. |
| [A Variational Inequality Perspective on GANs v1](https://arxiv.org/html/1802.10551v1#S3.SS1), §§3/3.1 and §4.1 | Convex player costs need not produce a monotone game operator. The paper motivates lookahead methods for regular/monotone operators. Those assumptions are not supplied by F/L convexity or by the current neural architecture. No GAN performance evidence is transferred. |
| [PCGrad v1](https://arxiv.org/html/2001.06782v1#S2.SS3), §§2.1/2.3 | Deliberately altering supplied gradients to address conflicts is established. Its conflict projection differs from fixed block permissions; it gives no exact equality or guarantee for I. |
| [ONE v2](https://arxiv.org/abs/1806.04606v2), [PCL v2](https://arxiv.org/abs/2006.04147v2), reused saved complete-method/deployment conclusions | Shared/private member and ensemble supervision, differential loss exposure through architecture, and online-KD update paths are close ancestry. Their extra distillation/gates/EMA/fusion heads and loss reductions differ. PCL's peer classifier bypass of ensemble hard CE is a concrete routing precedent. Exact backward/code detachment remains unqualified. |
| Saved Song/Chai, Jeffares and mixed-objective scout scopes | Shared/private backward rescaling, learner/pool interpolation, score/probability distinctions and generic nonconservativity were already recorded. The new contribution of this note is precise interpretation of the current source, the auxiliary qualification and a concrete local BCE counterexample—not discovery of split-gradient learning. |

Fresh primary method/definition scopes: three; fresh complete-paper/proof/performance credits: zero. ONE/PCL/other ensemble conclusions were reused from saved notes, not newly certified. No exhaustive closest-prior absence or method-novelty claim follows.

## 6. Evidence that would warrant a further optimizer question

The useful research question is whether **the particular internal-risk allocation creates a material optimization problem that limits an otherwise useful supervision policy**, and whether addressing that problem improves the served bank at a disclosed work cost. Merely naming the learner a game, finding nonzero curl, or reducing TRAIN J is insufficient.

Specific signatures should precede any added optimizer mechanism:

1. **An incremental coupling signature.** On exactly the same frozen batch/views/parameter state, check normalized external/internal directional projections of D_xy, separately from A_xy. Compare I with the matched O/G supervised fields and their identical auxiliary permissions. A few declared probes can establish nonzero coupling; they cannot establish the full spectrum or be chosen from favorable outcomes. At equal member predictions, grad D can vanish while D_xy remains nonzero when member parameter Jacobians differ; one-point gradient equality is not a dynamics diagnosis.

2. **A repeatable trajectory problem attributable to that coupling.** Persistent lag/oscillation, wasted opposing updates or near-stationary cycling should survive checks for stochastic-view noise, ordinary learning-rate effects and Adam history. Inspect actual displacements and deterministic losses/readouts separately from raw gradients. Estimated antisymmetric curvature matters only if it explains the observed behavior; a large curl with steady progress is not a reason to intervene.

3. **The right local stability regime.** Evidence of predominantly rotational, weakly damped modes would make a lookahead/correction question plausible. A genuine negative-real-part expanding mode or poor supervision tradeoff calls for a different explanation: extragradient does not repair the displayed counterexample. No local diagnostic alone establishes global monotonicity.

4. **A statistical target that remains useful.** Full-population served accuracy/NLL, mean/worst-member competence and repairs-versus-new-errors must support the underlying allocation. Better field residuals or TRAIN fit do not imply better WikiCS accuracy, MolHIV ROC AUC or Collab Hits@50. If the original policy has no serving benefit, optimizer polish does not by itself validate its graph-evidence rationale.

Any later optimizer comparison would change and charge field evaluations, native moment transitions and possibly stochastic-view coupling. Equal epochs would not establish equal work or the same learner. None is implemented, selected or launched here, and no such signature has been established from current outcomes in this task.

**Disposition:** retain a precise, attributed dynamics question. Reject novelty of the generic nonpotential/block-game interpretation and any claim of automatic accuracy gain, global descent, or extragradient necessity. Further optimizer work needs an observed, allocation-specific dynamics problem and a useful clean serving target.
