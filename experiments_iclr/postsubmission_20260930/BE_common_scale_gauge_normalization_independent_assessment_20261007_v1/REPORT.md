# BE common-scale gauge normalization: independent assessment

Date: 2026-10-07. Scope: saved literature first; selected retained and public primary method passages; exact algebra and prospective controls. No model or numerical imports, fixtures, training, dataset/checkpoint/raw internal-outcome payload reads, private-server access, active-job edits, root-ledger changes, or prior-artifact changes. Incidental inherited outcome interpretations in saved literature notes are not used as evidence for this assessment. This folder records an assessment, not an adopted method or execution protocol.

## 1. Conclusion

**Common-coordinate RMS normalization is a valid partial gauge choice, not a new function class or an established competence remedy.** Away from zero coordinate norms, it preserves every BE member function if the common scales are absorbed into the shared affine parameters with the correct bias convention. Unit and Rademacher factors already meet the proposed RMS constraint.

Forward differentiation through normalized factors, sphere updates, and post-update canonicalization are different optimizers. Magnitude/direction separation, tangent projections, rescaling-sensitive optimization, and invariant adapter updates have substantial prior art. A bounded search did not establish whether this exact across-member BE convention has previously been published; it does not support an absence or novelty claim.

The related **relative-credit** proposal has a precise diagnostic interpretation: keep ordinary own-loss Adam on the shared/body/boundary parameters and on the initial private update; restrict the additional ensemble-minus-own private correction to the tangent of each cross-member factor column, then retract to its post-own norm. This excludes that correction's first-order common positive scaling component. It neither keeps the pooled function fixed nor ensures stronger members, descent of the actual training risk, or improved quality. Common-only, relative-only, unrestricted, and no-correction controls could distinguish common feature scaling from relative steering. Until that distinction and a harmful optimizer mechanism are demonstrated, reject normalization alone as a scientific contribution and retain this as a prospective diagnostic question.

## 2. Exact common-scale algebra

Use column-vector activations and

\[
A_m=\operatorname{diag}(s_m)W\operatorname{diag}(r_m),
\qquad W\in\mathbb R^{d_o\times d_i},\quad m=1,\ldots,M.
\]

For each input coordinate j and output coordinate i define

\[
a_j=\sqrt{M^{-1}\sum_m r_{mj}^2},\qquad
b_i=\sqrt{M^{-1}\sum_m s_{mi}^2}.
\]

When all a_j,b_i are positive, set

\[
r'_{mj}=r_{mj}/a_j,\quad s'_{mi}=s_{mi}/b_i,\quad
W'=\operatorname{diag}(b)W\operatorname{diag}(a).
\]

Then

\[
\operatorname{diag}(s'_m)W'\operatorname{diag}(r'_m)
=\operatorname{diag}(s_m)W\operatorname{diag}(r_m)=A_m,
\]

and every r/s column has member RMS1. Applying this identity to each eligible affine map preserves the complete network function, including nonlinear downstream stages, because each affine output is unchanged. The absorption occurs within the affine map; no assumption that feature scaling commutes through a nonlinearity or graph attention is required.

The shared parameter must be available for the stated absorption. If the same W or LN affine parameters are tied across distinct sites or used by an unmodulated branch, independent sitewise absorptions can conflict or alter that other use. Those ties require a compatible joint transformation. If a protocol freezes W, treating absorbed W as trainable changes that protocol even though initial functions can match.

### Bias convention is part of the identity

The inspected public FactorLinear source implements

\[
y_m=\operatorname{diag}(s_m)W\operatorname{diag}(r_m)x+b.
\]

Its shared bias is outside s, so **b is unchanged**. TabM's retained method uses outside private biases b_m, which are also unchanged by this transformation. If instead the implementation is

\[
y_m=\operatorname{diag}(s_m)\bigl(W\operatorname{diag}(r_m)x+q\bigr),
\]

then q'=diag(b)q must accompany W'. The same absorption applies to q_m if the inside bias is private. Mixing these conventions changes functions.

### Post-affine LayerNorm factor

For

\[
h_m=c_m\odot\bigl(\gamma\odot\mathrm{LN}(u_m)+\beta\bigr),
\quad d_k=\sqrt{M^{-1}\sum_m c_{mk}^2}>0,
\]

the exact transformation is

\[
c'_m=c_m/d,\qquad \gamma'=d\odot\gamma,\qquad
\beta'=d\odot\beta.
\]

It scales the whole affine LN output. If the factor multiplies only the gamma term, beta is unchanged. A factor inside LN is a different operation: general featurewise scaling changes the normalized direction and does not commute with LN. Even uniform positive input scaling is not generally exactly invariant at nonzero epsilon, since its denominator is \(\sqrt{\alpha^2\operatorname{Var}(u)+\epsilon}\). Missing shared affine gamma/beta parameters can prevent the proposed absorption.

### Singular coordinates and incomplete gauge fixing

An all-zero factor column has zero RMS and no direction. One can sometimes preserve its zero effective contribution by choosing a nonzero direction and zeroing the corresponding W column/row, but that replacement is noninvertible and changes derivatives and subsequent learning access. A clamped/epsilon denominator gives a defined map but does not yield exact RMS1 for every column; its derivative and compensation must be declared. A zero factor entry in an otherwise nonzero column is allowed and does not create this column singularity.

This removes common **positive** input/output coordinate scales only. Common signs, per-member balancing \(r_m\mapsto k_m r_m,\ s_m\mapsto s_m/k_m\), and possible network symmetries remain. Per-member balancing can be followed by common RMS normalization and W absorption, giving different normalized representatives of the same functions. Zero entries in W can introduce further degeneracies. This is not global identifiability or a complete quotient construction.

Every all-one or elementwise ±1 initialization has coordinate RMS1, so its initial functions can be retained exactly. For arbitrary initial factors, canonicalize with W absorption before comparing optimizers; normalizing factors while leaving W untouched generally changes the starting functions.

## 3. Private radial credit is not automatically a null update

For a fixed input coordinate, the infinitesimal transformation

\[
\delta r_{:j}=\varepsilon r_{:j},\qquad
\delta W_{:j}=-\varepsilon W_{:j}
\]

is a joint first-order gauge direction. The analogous output direction uses \(\delta s_{:i}=\varepsilon s_{:i}\) and \(\delta W_{i:}=-\varepsilon W_{i:}\), with inside-bias compensation when required. For a differentiable function-only loss L and the outside-bias convention, invariance implies

\[
\sum_m r_{mj}\partial_{r_{mj}}L
=\sum_i W_{ij}\partial_{W_{ij}}L,
\qquad
\sum_m s_{mi}\partial_{s_{mi}}L
=\sum_j W_{ij}\partial_{W_{ij}}L.
\]

Thus the full Euclidean gradient is orthogonal to each joint gauge direction. A private radial gradient component need not vanish: it can express a useful common column/row scaling of all A_m. Removing it routes that degree of freedom to a different parameter block and changes its effective learning dynamics. It is incorrect to infer wasted/null learning merely from large private radial gradients or drifting factor RMS values. Adaptive preconditioning, finite steps, regularizers, and poorly balanced representatives can create gauge-dependent behavior, but that behavior must be shown in effective maps/functions and its consequences.

An elementary algebraic conditioning example is \(p=wr,\ L=(p-y)^2/2\). One simultaneous scalar SGD step gives

\[
p^+=p-\eta(p-y)(r^2+w^2)+\eta^2(p-y)^2p.
\]

The gauge \(w\mapsto w/a,\ r\mapsto ar\) keeps p fixed but changes the first-order coefficient \(r^2+w^2\). This establishes possible gauge sensitivity of an optimizer. It is not evidence that the current BE training has that failure or that RMS fixing improves its task performance.

## 4. Three normalization operations must be distinguished

### Forward normalization of unconstrained raw factors

For each nonzero raw cross-member column u, use

\[
n(u)=u/\mathrm{RMS}(u)=\sqrt M\,u/\|u\|,
\quad P_u=I-uu^\top/\|u\|^2,
\quad Dn(u)=\frac1{\mathrm{RMS}(u)}P_u.
\]

This is the magnitude/direction derivative appearing in weight normalization, applied across members rather than across a neuron's input weights. Every member's gradient is coupled by the common norm. The raw radius remains redundant in the forward map and influences the effective step size. Forward normalization therefore does not literally remove the radial coordinate from storage or optimizer state. Differentiating the denominator is essential; detaching it removes the tangent projection and defines another backward rule.

### Explicit sphere factors

Represent each column f directly on \(\|f\|=\sqrt M\). A tangent displacement v satisfies f^T v=0; the normalization retraction is

\[
\mathcal R_f(v)=\sqrt M\frac{f+v}{\|f+v\|}.
\]

At f=1, tangent projection is \(I-11^\top/M\), so the incremental member sum is zero. At general f, it removes the component proportional to f; it does **not** generally impose a zero ordinary member mean. At a Rademacher column it removes the signed common scaling component. For M=1 there is no continuous tangent degree of freedom.

Projecting a gradient before Adam is insufficient. If f=(1,1), h=(1,-1), and a diagonal preconditioner is diag(1,2), then h is tangent but Dh=(1,-2) is not. Project the actual displacement after all preconditioning/momentum operations, and retract. Persistent tangent momentum requires a declared transport between tangent spaces. None of these steps supplies a natural-gradient or gauge-invariant optimizer automatically.

### Post-update canonicalization with absorption

Canonicalizing updated factors and absorbing scales into W preserves the just-produced functions. Later Euclidean/Adam steps can differ, since the representative and optimizer state changed. Merely resetting moments introduces another optimizer change; naive elementwise moment rescaling generally does not transport a nonlinear, coupled, dimension-reducing canonicalization. A local invertible coordinate change would require the corresponding transformed vector/dual/metric laws. The canonical map here has a gauge kernel, so there is no full inverse Jacobian on redundant coordinates.

Post-training canonicalization alone cannot improve predictions in exact arithmetic. No training experiment is needed to establish this analytic negative control. Finite-precision differences would be implementation effects, not a quality mechanism.

## 5. Relative-only ensemble-minus-own correction

This is an analyzable representative of the parent's follow-up idea, **not a selection for implementation**. Separate it from always enforcing RMS1: it preserves each column's norm after the ordinary own-loss update, which may differ from1 and may change across training steps.

Let F be the declared aggregate of own-member losses and A the declared ensemble prediction loss. The factor correction is \(h=\nabla_\phi(A-F)\), with exactly the same loss reductions, labels, pooling and scale in all controls. Define a two-stage rule explicitly:

1. Apply the existing own-loss Adam update to all declared trainable blocks, obtaining \(\theta^+\), including private factors. Its moments are advanced using own-loss gradients only.
2. At \(\theta^+\), evaluate h for private factors only. This chosen post-own evaluation point is part of the rule; taking h before the own step defines a different rule.
3. Form a correction displacement d from h. A minimal declared choice is \(d=-\eta_c h\), with no correction momentum/state. If a fixed own-Adam preconditioner is instead used, declare \(d=-\eta_c D_t h\), its epsilon and state source. Then project d, not just h.
4. For each nonzero r/s/c cross-member column f at \(\theta^+\), set

\[
Q_f=ff^\top/\|f\|^2,\quad P_f=I-Q_f,\quad
v=P_fd,\qquad
f^{\mathrm{new}}=\|f\|\frac{f+v}{\|f+v\|}.
\]

The shared matrices/body and prediction boundaries receive only the ordinary own update; the additional correction does not change them. This preserves post-own column RMS exactly. Correction moments are not fed into own Adam in this representative, so the state semantics are clear. Introducing a separate adaptive correction optimizer needs explicit moment and tangent transport semantics.

Because v is tangent,

\[
\|f+v\|^2=\|f\|^2+\|v\|^2,
\quad
f^{\mathrm{new}}-f
=v-\frac{\|v\|^2}{2\|f\|^2}f+O(\|v\|^3/\|f\|^2).
\]

The correction is relative to first order; the necessary finite-step retraction has a second-order radial term. Norm preservation is exact, while absence of any radial displacement is not. Zero columns require an explicitly shared rule across controls and cannot use this expression.

The tangent complement is defined by the chosen member-column Euclidean metric, not by the full function-map kernel. Other gauge directions can remain within these tangents, including coordinated changes among r/s columns. Calling this a complete removal of gauge credit would therefore be too strong.

### Norm preservation does not preserve the mean function

Even f^T v=0 is a parameter-space constraint, not a constraint on the pooled predictor. Its first-order change is \(M^{-1}\sum_m J_m\delta\phi_m\), which need not vanish. For non-unit factors, the tangent condition does not even make the unweighted parameter sum constant.

An exact two-member ReLU counterexample starts with r=(1,1), W=1, s=1, outside bias −1 and input x=1. The mean output is initially0. A tangent displacement (epsilon,−epsilon), 0<epsilon<1, retracted to the same RMS1 gives

\[
r'=\frac{(1+\varepsilon,1-\varepsilon)}{\sqrt{1+\varepsilon^2}},
\qquad
\bar z'=\frac12\left(\frac{1+\varepsilon}{\sqrt{1+\varepsilon^2}}-1\right)>0.
\]

This is a symbolic counterexample, not a model fixture or numerical run. Relative-only updates can alter graph attention, hidden activations, logits and the pooled predictor. Their being relative is not an ensemble-risk preservation guarantee.

Projection/retraction of A−F credit is known constrained-optimization machinery. The correction is not generally the gradient of the own risk, pooled risk, or the complete mixed update. No general descent, stability, calibration, diversity, or weakest-member competence conclusion follows. Ordinary ownAdam alone also gives no such guarantee.

## 6. Regularization and Bayesian limits

Parameter L2 penalties are gauge dependent: \(\|W\|^2+\|R\|^2+\|S\|^2\) changes under the exact absorption. On fixed RMS1 spheres the r/s L2 norms are constants, so their constrained regularization role changes. Decoupled AdamW decay also depends on which blocks and representatives are decayed; projection or retraction can remove a radial factor decay while shared-W decay remains active. Reusing the same numeric decay coefficients does not mean the same function regularizer.

Rank-1 BNNs put distributions/priors on r/s, point-estimate W, and optimize expected likelihood plus factor KL and a W prior. Those terms are not gauge invariant. A correct change of probability coordinates transports the complete prior/posterior and density/Jacobian information; normalizing sampled factors across members while absorbing their random RMS into W can make formerly deterministic W random and couple factors/samples. Keeping the native independent family and KL unchanged generally changes the Bayesian model. RMS normalization of deterministic factors does not inherit Rank1BNN posterior or uncertainty guarantees.

The post-own norm-preserving credit diagnostic can be assessed as a deterministic optimizer rule without making a Bayesian claim. Even then, its own-stage decay, correction-stage state, loss coefficients, and zero-column treatment must be matched.

## 7. Primary overlap and read limits

Saved notes were checked before new retrieval. Retained primary method passages were revisited only for this named scale/gradient question. Mechanically retained HTML is not a whole-paper read. No numerical literature result was reproduced, author-code audit was performed, or global absence claim was certified.

| Primary/version and inspected scope | Established overlap | Limit for this proposal |
|---|---|---|
| [BatchEnsemble,2002.06715v2](https://arxiv.org/abs/2002.06715v2), retained §3.1 Eqs.1–5 | Shared W and private input/output rank-one scaling; feature-scaling algebra. | No claim that the inspected passage specifies this exact common-coordinate RMS gauge. BE itself is prior. |
| [Rank-1 Bayesian Neural Networks,2005.07186v2](https://arxiv.org/html/2005.07186v2), retained §3.1 expected-likelihood/KL/prior passages; saved initialization conclusions | Distributions over rank-one factors, deterministic W, factor priors and initialization choices. | A normalization is not an unchanged VI model without distribution transport. |
| [TabM,2410.24210v3](https://arxiv.org/pdf/2410.24210v3), retained method PDF pp.4–6; saved competence assessment | Outside private bias; random first adapter and later identity multiplicative adapters; own-member supervision and collective selection. | Unit/±1 starts can be preserved; useful weak submodels do not imply a standalone competence guarantee. |
| [Weight Normalization,1602.07868v3](https://arxiv.org/abs/1602.07868v3), retained §2–2.1 Eqs.2–4 | Explicit magnitude/direction reparameterization, projected derivative, raw-radius/effective-step effects; distinguishes reparameterized training from post-step normalization. | Applies across weight coordinates rather than BE members; generic normalization machinery is established. |
| [DoRA,2402.09353v1](https://arxiv.org/html/2402.09353v1), fresh §§4.1–4.3 Eqs.5–11 | Normalized direction plus learned magnitude, low-rank directional update, initial function preservation; projected derivative. | Its §4.3 detached-norm variant removes the derivative projection. Do not conflate normalized forward and tangent backward. Additive pretrained adapter is a different factor model. |
| [Path-SGD,1506.02617v1](https://arxiv.org/html/1506.02617v1), fresh §§2 and4, rescaling definition and Eqs.6–9/Theorem4.1 proof | Function-equivalent ReLU rescalings can yield different SGD updates; path geometry gives a rescaling-invariant rule. | Its network graph is a computation DAG; it is not evidence of a data-graph-specific failure or this exact BE constraint. Exact v2 HTML returned404; v1 is the inspected version. |
| [LoRA-Pro,2407.18242v1](https://arxiv.org/html/2407.18242v1), fresh §§3.1–3.3 Eqs.1–14 | Factor updates induce a distinct effective weight update; modifies factor gradients to approximate a full-weight gradient. | Full-rank assumptions and additive low-rank map differ; finite-step/Adam and BE competence conclusions do not transfer. Main method text inspected, proof/Adam appendix not audited. |
| [LoRA Done RITE,2410.20625v1](https://arxiv.org/html/2410.20625v1), fresh §2 definitions/scale discussion and §3.1–3.2 Eqs.9–17 plus algorithm opening | Distinguishes equivalent factorization from optimizer invariance; matrix preconditioner; adjusts first/second moments for changing bases. | Its invertible rank-basis transformations differ from this partial BE gauge. Its theorems are not inherited by sphere projection or unchanged Adam moments. Proofs/convergence and empirical tables not audited. |
| [Revisiting Natural Gradient for Deep Networks,1301.3584v1](https://arxiv.org/html/1301.3584v1), fresh §2 Eqs.1–4 | Fisher/KL geometry aims at parameterization-invariant infinitesimal function changes, including redundant parameter symmetries. | Euclidean member-sphere projection is not a Fisher metric. Singular gauges, metric approximations, damping and finite steps need their own treatment. |

The most direct generic competitors are weight normalization and constrained tangent/retraction updates; the most direct conceptual counterargument to a normalization novelty claim is the established distinction between same function and same optimizer. LoRA-Pro/RITE support examining effective updates and state transport. They do not prove this deterministic BE remedy effective. No graph specificity, compute advantage, implementation work, adoption, or source modification is offered as novelty.

## 8. Necessary failure mechanism and matched controls

Two hypotheses must be separated.

**Gauge/conditioning hypothesis:** redundant common-coordinate scale routes and a gauge-sensitive optimizer materially distort effective updates, limiting useful relative adaptation or destabilizing learning. Evidence must go beyond raw factor RMS: it needs effective A_m/function changes, the update geometry in a declared representative, and competence consequences. Gauge fixing might help that failure, leave it unchanged, or remove a useful adaptive learning-rate mechanism.

**Private-credit interpretation hypothesis:** gains attributed to pooled private credit could be produced by common feature reweighting rather than relative member steering. This is a narrower attribution question. It does not require claiming that common scaling is intrinsically harmful. The relative-only intervention tests a restriction of a credit policy, not a function-equivalent relabeling of its unrestricted updates.

For the second hypothesis, use the same own update, correction h, evaluation point and displacement generator d in all arms. For each nonzero post-own column f:

| Control | Additional private update | Question answered |
|---|---|---|
| Own only | None | Does any correction help beyond the ordinary own stage? |
| Unrestricted | f+d | Reference correction with both common and relative components. |
| Common only | f+Q_f d | Can direct common feature scaling explain the benefit? |
| Relative only | \(\|f\|(f+P_f d)/\|f+P_f d\|\) | Does norm-preserving relative correction retain the benefit? |

All three corrections use the same raw d and coefficient. Their retained step sizes and finite-step functions differ. Record projected component sizes and effective function/map changes; do not quietly enlarge a small projected component to equal the full correction. If a matched-step sensitivity is needed, declare it separately with zero-component handling and match in a stated metric. A positive common-only scale leaves member coordinate ratios unchanged but can still change nonlinear member/pool functions. A common-only arm that also retracts to the original norm would cancel its positive radial correction and is an invalid control.

For the outside-bias BE map, if all common-only corrections give r'_mj=t_j r_mj and s'_mi=u_i s_mi, their finite-step layer effect is exactly the one obtained by keeping r/s fixed and replacing W by diag(u)Wdiag(t), with outside biases unchanged. For a whole post-affine LN factor c, the equivalent common update is gamma'=t⊙gamma and beta'=t⊙beta. This is the precise sense in which the common-only arm can reveal shared feature reweighting implemented through parameters named private. It is not a claim that the reweighted functions are unchanged.

For the first hypothesis, an RMS1 forward-normalized optimizer or an explicit sphere optimizer must be chosen as one complete rule. Compare to the same unrestricted BE function class from identical effective initial functions, and to a generic magnitude/direction or scale-aware preconditioning comparator. Canonicalization-only after fitting supplies the analytic function-preservation control. Gauge-equivalent initial representatives can probe optimizer sensitivity, but representative scales and transformed initial state must be prescribed; a scale grid is not a scientific mechanism.

Keep architecture, eligible factor sites, bias/norm semantics, starting member functions, loss reductions/coefficients, label exposure, batches/order, own optimizer schedule/state policy, stopping/selection, readout/pooling, random seeds, and evaluation boundaries matched. The additional correction cannot quietly change supervision, private head exposure, or selector criteria. Regularization must be reported as a potentially changed mechanism, or represented by a common function-level penalty. A comparison with the same numeric learning rate alone does not rule out a generic effective-step-size explanation; a small declared rate/step sensitivity may be needed if that is the remaining ambiguity.

Measure member competence and pooled competence separately using the already declared task metrics. Error diversity or weak-member gains alone are insufficient; an intervention can weaken useful members or change calibration/pooling. Parameter-space dot products and radial fractions are representative-dependent diagnostics, not intrinsic causal evidence. Even a successful matched contrast supports its exact optimizer/routing intervention; it does not certify a graph-specific mechanism or universal competence preservation.

Falsifiers include: common-only reproduces the unrestricted gain; relative-only loses it under matched step semantics; generic normalization/preconditioning reproduces the result; effective updates show no meaningful gauge sensitivity; or relative corrections reduce competence enough to eliminate pooled utility. No outcome of these controls is predicted here.

## 9. Disposition and retained evidence

Reject a broad claim that RMS normalization itself is new or repairs weak BE members. The exact partial gauge algebra is valid. The relative-credit decomposition is feasible as a declared deterministic optimization experiment and useful for attribution, but has no adopted code, measured benefit, or theoretical quality guarantee in this assessment.

`READ_SCOPES.json` binds the exact source ranges; `INSPECTED_PASSAGES.json` retains the selected passages; `INPUT_BINDINGS.json` binds saved notes and the public factor source. Retrieval/search records distinguish metadata discovery, failed access, raw retention and semantic reads. `MANIFEST.json` hashes only this newly created folder. The old notes, seals, indices, runner source and root status are unchanged.
