# Conditional native decision-residual control

7 October 2026. **An attributed control remains worth specifying, despite no new primitive.** Native prediction residuals resolve the hidden-code blindness, but negative variance has its own pool-invisible failure. The preferred quality control is the already published GNCL member/actual-pool risk mixture, pulled into existing internal private factors while shared W learns own loss. No strength, implementation, compute or novelty is adopted.

The sealed two-loss counterexample and limits remain in `shared_backbone_decision_visible_contrast_assessment_20261007_v1`. The expensive3October warm/frozen-W graph-removal/guard proposal is not repeated here.

## 1. Precise regularizer

At the current weights, obtain dropout-off **native-graph served member probabilities** on a fixed complete TRAIN target panel of N objects. Use the actual class probabilities for multiclass classification, and one positive-class probability for binary BCE; do not normalize a scalar per object. Labels define only class groups already available to training. No validation/test labels, new graph view or learned projection enters this auxiliary.

Let e_m(v)=p_m(v)−y_v, with one-hot y for multiclass and scalar y for binary. Center within each TRAIN class:

`R_m(v)=e_m(v)−mean_{u:y_u=y_v} e_m(u)`.

Since the target is constant within a class, this equals class-centered p_m. Flatten all target/output coordinates into one R_m; singleton classes contribute zero here and retain ordinary supervision. Let `R̄=M⁻¹Σ_m R_m` and use

`V=(MN)⁻¹Σ_m ||R_m−R̄||²`, `D_native=−V`.

Minimizing D rewards within-class, between-member prediction variance. It preserves amplitudes, avoids response-norm division/floors, and is well-defined for binaryM4. Binary residual vectors live across N targets rather than in a one-dimensional normalized wrong-class direction. This is not a bare Gram determinant; no ADP rank/epsilon repair is needed. With probabilities, V is bounded: at most1/4 for a scalar binary output, and at most1−1/C for C-class probability vectors. Class-centering is an orthogonal row projection and cannot increase that bound.

Variance strength is **unadopted**. The current .05 hidden coefficient has no units/gradient equivalence to this statistic. Jointly minimizing an unconstrained multiplier of D≤0 can drive it upward. No coefficient search or claimed optimal value is introduced.

## 2. Exact ancestry and the counterexample resolution

Write `Q_m=N⁻¹||R_m||²` and `Q_pool=N⁻¹||R̄||²`. Then

`V=mean_m Q_m−Q_pool`.

For these quadratic residual losses, `mean Q_m−λV=(1−λ)mean Q_m+λQ_pool`. That is the familiar member/pool interpolation underlying negative-correlation and joint-ensemble objectives. Retaining native own **CE/BCE** makes the combined objective a different surrogate mixture, not a new principle or an exact Brier-risk identity for CE. This is the decisive prior attribution.

If a hidden transformation leaves native member probabilities unchanged, it leaves R and V exactly unchanged. In the supplied construction, every member predicts `W_a a(v)`, so all R_m agree and V=0 regardless of the simplex code's amplitude. The new auxiliary cannot report success from that code. It is therefore a genuine **change of the optimized object** from the present hidden contrast, while remaining an established functional-diversity family.

It does not force useful decisions. V can increase through confidence variation or different wrong probabilities while the pool retains common errors. It ignores within-class mean prediction differences; centering does not freeze those means under parameter updates. At exactly identical predictions, ∇V=0, so it cannot spontaneously break symmetry. The existing fixed BE initialization/stochastic own training supplies only a possible starting asymmetry, not a guarantee. Private capacity and common feature restrictions remain.

## 3. Native visibility is not pooled usefulness; prefer a direct risk control

For valid probabilities, `p'_m(v)=p̄(v)+α[p_m(v)−p̄(v)]` keeps the mean-probability pool exactly unchanged. It scales `R'_m−R̄=α(R_m−R̄)`, so V grows as α² while member CE can worsen. Native prediction visibility is therefore weaker than useful pooled diversity. Binary normalization avoidance does not fix this failure.

For a raw-logit pool, `z'_m=z̄+α(z_m−z̄)` likewise preserves the served mean scores while member predictions can change; the probability-variance scaling is not generally α² because sigmoid/softmax is nonlinear. Pooling contracts must remain explicit.

[GNCL, §4.1 Eq.5](https://arxiv.org/html/2011.02952v2#S4.E5) supplies the more meaningful already inspected quality control:

`J_GNCL=(1−λ) L_mean+λ L_pool`, `λ∈[0,1]`.

Use the **actual task pooling and supervised loss**, not a quadratic surrogate relabeled as CE. WikiCS uses `p̄=mean_m softmax(z_m)` and `L_pool=−mean_v log p̄_y(v)`. Binary tasks whose declared pool is mean raw logits use `z̄=mean_m z_m` and BCE-with-logits at z̄. Collab risk is the mean positive-edge logistic loss plus the mean negative-edge logistic loss, not one mean over their combined unequal counts. Molecule risk is the uniform mean BCE-with-logits over all declared finite TRAIN target/output entries; no missing or bad target may silently be removed. Compute L_mean with the identical targets and reduction for each member, then average members. With two stochastic own views, compute the exact mixture per view and average; do not first pool views or pretend the training stochastic realization is dropout-off inference.

The source's λ range/interpolation is established ancestry. **No λ value is adopted here**, and .05 is not imported from hidden contrast. At λ=0 the current shared model still has tied weights; it does not become the independent architecture of GNCL's untied limiting description. At λ=1 a pool-only private objective can permit weak/colluding members; there is no universal gain guarantee.

Nor should λ simply be jointly minimized as another inner parameter: under convex CE/BCE and the correct averaging contract, `L_pool≤L_mean`, so `∂J/∂λ=L_pool−L_mean≤0` favors the pool-only endpoint. A strength/selection protocol remains a separate prospective choice.

With fixed pool predictions and actual member CE, increasing antithetic member spread cannot improve J_GNCL for λ<1 merely by worsening member risk; the variance reward can have that incentive. At λ=1 the mixture is indifferent to fixed-pool changes. This is a direct improvement in what the quality objective measures, not a diversity or generalization theorem. No ranking-MRR guarantee follows from its BCE surrogate.

## 4. Parameter permissions and the simplest conditional comparison

Declare θ as shared weights, φ_m as **existing internal** member factors eligible for steering, and ψ_m as other private parameters, including any private classifier parameters. No factor, head, router or teacher is added. Let L_mean be the exact own-loss mean defined above, and A the existing alignment auxiliary. A matched conditional native control is

`g_θ=∇_θ L_mean`, `g_ψ=∇_ψ L_mean`,

`g_φ=∇_φ[J_GNCL+.05 A]`.

The private derivative propagates through the real predictor and shared operations at the old state, holding θ and ψ fixed **for the auxiliary/mixed private derivative**. Detaching an upstream hidden tensor would remove the desired pullback into earlier private factors. Include all true cross-member φ dependencies. Shared W keeps learning own supervised loss; the private block receives the established member/pool objective.

This is an attributed mixed block update, generally not the gradient of one unchanged scalar objective over all parameters. It has no ordinary-gradient convergence or competence guarantee. Pool risk can be computed from existing member logits/probabilities without a new graph view, but obtaining different shared/private derivatives may require extra backward work. Charge that work. A deterministic full-TRAIN V panel is an optional diagnostic, not an uncharged training term in this preferred control.

A later representative WikiCS comparison would fix matched conditions before outcomes:

1. Own loss plus A steering only φ, with no additional member/pool mixture.
2. The same shared/private permissions, with the current hidden residual contrast steering only φ.
3. The same shared/private permissions, with J_GNCL steering φ and no hidden residual term.

The current running hidden-full arm is useful context, but it has a permission confound: auxiliaries reach shared W there. The matched hidden-private control is necessary. Matching permissions isolates that confound; native GNCL versus hidden InfoNCE also changes loss geometry/strength, so it does not isolate every mathematical difference between their objects.

Use the same fixed TRAIN identities, own-loss reductions, stochastic views, initialization, paired seeds and checkpoint selection. Choose one λ only in a separately adopted prospective protocol, using no new heldout labels or grid here. Retain a capable stochastic-view-matched single and packed untied bank as utility references. There is no new warm acquisition, graph augmentation, response-energy guard or backtracking campaign in this control.

A full-TRAIN decision-residual panel must center across the entire target population; normalizing scalar per-object logits is excluded for binaryM4. A microbatch that breaks grouping changes its estimator. Native residual V can diagnose prediction-visible spread, but served utility and exact common wrong-class/negative identities decide whether it is useful. Better mean-member MRR or lower hidden loss is insufficient. For ranking, bind the actual score pool, positive/negative supervision and competitor identities before claiming a task-specific interpretation.

Failure modes remain pool-fixed member worsening under a variance reward, pool-only collusion, confidence-only variation, exact-symmetry zero gradient, private capacity limits, TRAIN overfitting and unequal gradient opportunities/costs. Neither native visibility nor the GNCL mixture guarantees graph-evidence diversity or heldout gain. A failed fixed utility comparison does not authorize selecting another coefficient or old response/guard campaign after outcomes.

**Disposition:** keep the direct GNCL/private-pullback comparison as a conditional attributed quality control; keep centered native variance as a diagnostic or separately declared surrogate with its failure explicit. New primitive and methodological novelty are not claimed. No new family, source code, dataset/test/GPU/77 work, strength choice or running-family edit follows from this note.

Scope: saved GNCL Eq.5/λ range and NCL/member-pool, DICE/CDLG/FoRDE/function-space and earlier3October conclusions reused. New source reads/retrievals, full-paper credits, author-code scopes, raw outcomes and experiments: zero.
