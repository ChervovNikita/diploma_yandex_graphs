# Count-conditioned joint identifiability: bounded source assessment

## Plain explanation

Giving both side counts tells the auxiliary how many TRAIN-observed counterpart links are hidden, but leaves it to predict which residual candidates carry them. A single conditional-Bernoulli component learns relative slot preferences. J_K adds a uniform latent member whose preferences must explain both sides together. J_K_sep allows each side to choose its member independently. This is a local identity-assignment contrast. Its normalization and mixture ingredients are classical; no novelty or accuracy conclusion follows.

The graph-specific candidate signal is agreement of the same member's relative missing-counterpart-link preferences on two native residual neighborhoods, using visible TRAIN graph context/features. Count conditioning removes direct count prediction and common-side logit offsets. It leaves degree, record multiplicity, masking and query-sampling cues. It does not create a graph-wide latent state or make TRAIN zeros verified nonlinks.

## 1. Fixed-parameter contrast and gradients

Fix visible context C, side counts K=(k_L,k_R), M positive components, finite logits and feasible patterns. Write u_m=q_mL(z_L|C,K), v_m=q_mR(z_R|C,K), U=sum_m u_m, V=sum_m v_m, A=sum_m u_m v_m. Let d=max(n_L+n_R,1), the existing per-query denominator.

q_J=A/M; q_sep=UV/M^2.

r_Lm=u_m/U; r_Rm=v_m/V; r_Jm=u_m v_m/A.

**Exact observed-pair identity:**

(J_K-J_K_sep)_query = -log[M sum_m r_Lm r_Rm]/d.

Equivalently q_J/q_sep=1+Cov_uniform(u_m,v_m)/(mean(u) mean(v)). Positive likelihood alignment favors J_K on this pair; negative alignment favors J_K_sep. There is no general ordering. Since sum r_L r_R <=1, the gap is bounded below by -log(M)/d; no universal finite upper bound holds as logits range over arbitrary finite values.

For conditional inclusion probability pi_msi and teacher bit z_si,

partial J_K/partial eta_msi = r_Jm (pi_msi-z_si)/d;

partial J_K_sep/partial eta_mLi = r_Lm (pi_mLi-z_Li)/d,

with r_Rm on the right. Hence the fixed-logit gradient difference on side s is (r_Jm-r_sm)(pi_msi-z_si)/d. This explains the coupling mechanism more exactly than member spread. Shared-parameter gradients include the corresponding neural Jacobians and all query reductions.

At fixed logits, whole-side marginals are exactly matched. After separate fits, the logits and side marginals can differ. An end-of-fit arm difference therefore combines training dynamics, marginal competence and downstream transfer; it cannot be labeled a pure measured coupling effect. Score both laws at each fitted parameter state to separate the local law switch from changes learned during fitting.

An observed log-ratio is not mutual information. Its expectation under q_J is the model's conditional mutual information because q_sep is q_J's product of marginals. Under a distinct teacher distribution P, the expected NLL gap is KL(P||q_J)-KL(P||q_sep), with no fixed sign. It equals minus true conditional mutual information only when q_J=P and the product law has P's marginals. One observed graph and overlapping mask contexts do not establish those conditions.

## 2. Exact equality, identifiability and collapse conditions

Let Umat(a,m)=q_mL(a), Vmat(b,m)=q_mR(b) over all feasible side patterns; center every row over members to get Uc and Vc. Then

q_J(a,b)-q_sep(a,b) = [Uc Vc^T](a,b)/M.

**The full laws are equal iff Uc Vc^T=0.** Sufficient cases are M=1; either side has a unique feasible pattern; or either side's component conditional laws all coincide. For M=2 the difference is (u_1-u_2)(v_1-v_2)^T/4, so full-law equality requires one side's laws to coincide. For larger M orthogonal centered variation can cancel even while both sides vary. Covariance zero on the single observed pair is only a scalar loss equality, not full-law equality.

Unique-pattern sides are exactly k=0 or k=n, including n=k=0. Their loss and logit gradients are zero; their likelihood is1 for every member. If at most one side is informative (0<k<n), J_K and J_K_sep are the same objective and gradient for that query for all logits. Nontrivial coupling can therefore arise only from queries with both sides informative. Two informative sides are necessary, not sufficient.

A coincident-law parameter point needs care: if only the left component laws coincide and the right ones differ, q_J=q_sep, but left gradients can differ (r_J=r_R versus r_L=1/M). Pointwise equality is not an identity over neighboring independently changeable logits. If both sides' laws coincide across members, all responsibilities are uniform and J_K, J_K_sep and W_K have equal losses and logit gradients there. Their informative gradients are generally nonzero. Symmetric components remain symmetric only under correspondingly tied/symmetric parameterization, initialization, optimizer history, target gradients and stochastic draws; collapse is permitted, not inevitable.

For one side with finite logits and 0<k<n, every feasible pattern has positive mass and 0<pi_i<1. Thus every individual teacher-slot gradient pi_i-z_i is nonzero. The Hessian is Cov_q(Z), with exactly the common-offset null direction for this full-support fixed-count family. Adding any member/side constant to logits leaves its law unchanged. A parameter influencing only such a constant has zero conditional gradient. Parameters entering only a potential g(C,k_L,k_R) also have zero gradient after conditioning both counts. Shared nonlinear parameter updates need not preserve raw serving scores merely because output gradients sum to zero.

Other exact parameter-gradient zeros require the neural Jacobian to annihilate the conditional gradient, tied cancellations, zero masks/detachment, or cancellation across the batch. Finite informative logits do not by themselves yield zero per-slot gradients. Responsibility starvation and saturation can make gradients very small; exact zeros from floating-point underflow/rounding are numerical events requiring evidence, not the finite-logit theorem.

## 3. What component permutations do and do not show

A common permutation of member indices on both sides is an exact J_K symmetry. J_K_sep is invariant to independent side permutations. Member labels are therefore never individually identifiable.

For a relative permutation sigma of right-side indices, q_sigma=(1/M) sum_m u_m v_sigma(m). Averaging **likelihoods** uniformly over every permutation gives

E_sigma q_sigma = (sum_m u_m)(sum_m v_m)/M^2 = q_sep.

This follows because each v_j occupies each position with probability1/M. If relative pairing is unknown and truly integrated under that uniform prior, the resulting joint density is exactly the independent-side density. Independent uniformly randomized labels destroy the intended pairing in that marginal model.

However E_sigma[-log q_sigma] >= -log E_sigma[q_sigma] = J_K_sep*d. Averaging shuffled losses is a different training objective with a Jensen penalty, not an implementation of J_K_sep. A fixed independent permutation changes pairing but is arbitrary and can be relearned; a fixed common permutation changes nothing. No training permutation is proposed here.

The existing objective mixes separately inside each query and then sums log losses: product_query [(1/M) sum_m q_m(query)]. It does not use [(1/M) sum_m product_query q_m(query)], which would posit one member for a whole set/graph. Reusing network parameters across queries couples optimization, but supplies no graph-wide latent-member likelihood or cross-query consistency theorem.

## 4. What the actual TRAIN construction can confound

The saved native source masks each selected 65536-record batch before symmetrization/deduplication. A remaining duplicate record can retain the same edge. One masked graph serves every positive query and its indexed sampled-negative query in that batch; negatives exclude complete raw/TRAIN observed edges. The iterator uses17 full batches and drops its declared tail. A future conditional adapter must preserve this exact construction and all-query reduction; this assessment does not modify it.

For query (u,v), left residual candidate w is visibly adjacent to u but not v. Its teacher bit is1 exactly when (v,w) belongs to complete TRAIN. For a full-TRAIN common neighbor w, visible leg states (u-w,v-w)=(1,0) produce a left positive, (0,1) a right positive, (1,1) an excluded visible common neighbor, and (0,0) no residual slot. Thus mask allocation of wedge legs itself creates coupled side labels. For an observed positive query these are triangle legs; for a sampled nonedge query they are full-TRAIN common-neighbor wedges.

Fixed-size record masks are not independent edge Bernoulli masks. Ignoring duplicates and forced query inclusion, two distinct one-record edges have Cov(mask_e,mask_f)=-p(1-p)/(N-1), p=batch_size/N. An edge with multiple records disappears only when all of its records are selected. Positive-query conditioning, shared edges/endpoints, common contexts and duplicate survival introduce further structure. These facts can explain pairing, count strata or apparent specialization without an identified graph-wide cause.

Conditioning both k values removes the number of positive residuals, but does not generally remove identity biases from multiplicity, degrees, support selection or visible embeddings. Conversely, if P(z_L,z_R|C,K) factorizes and its side marginals are representable, extra coupling has no population log-score advantage over their product. If hidden identity is exchangeable within both supports given C,K with no cross-side association, the uniform fixed-count law is optimal. The source audit cannot determine either population condition from construction alone.

The native masking therefore does not invalidate the algebraic matched-side contrast. It limits its interpretation to this source/mask/query regime and can remove nearly all informative-two-side signal. Large raw query counts are not independent graph samples. Record-level resampling or query-wise confidence intervals cannot establish graph-level replication.

## 5. Capable single and remaining falsifiable claim

An unrestricted fixed-count autoregressive single can represent any strictly positive finite-support joint law by chain rule: at each nonforced prefix, choose the exact conditional next-bit probability; force bits only at exhausted/full remaining budget. It can therefore represent any J_K distribution without retaining M target predictors. This is an expressivity fact for an unrestricted conditional function, not a capacity theorem for the frozen width64 MLP.

The saved S_K prototype has a523->64->1 head, visible centered unary/context features and per-side mean selected-embedding prefixes. That prefix summary can collide for different selected subsets and is not generally the full M-dimensional mixture posterior. Its finite head/backbone has no proved universal dominance over J_K, although its saved matching-side oracle demonstrates a nontrivial proposed capability by source design. No oracle was executed here. Counts/prefix bits remain auxiliary-only; native count-free serving is unchanged. A count-only C_mu potential cancels after both counts are fixed and cannot replace this identity-sensitive control.

A qualified S_K matching conditional competence and served ranking would defeat a necessity claim for retaining predictive members in that tested regime. S_K failure, infeasibility or missing native qualification would not establish bank necessity: features, order, optimization, target capacity, parameter cost and runtime differ. A gain over J_K_sep alone can reflect responsibility-induced changes in learned marginals or regularization. A conditional-NLL gain without served ranking benefit supports only an auxiliary reconstruction result. Any favorable one-seed finding remains exploratory and requires separately fixed replication.

The bounded remaining hypothesis is: under the preserved full-TRAIN native masking regime, same-member two-sided conditional identity supervision supplies useful auxiliary gradients beyond matched independent-side and capable-single controls, and those gradients improve count-free served ranking. All parts can fail. The attached diagnostics make those failures observable; they authorize no fit, source amendment, novelty or accuracy claim.
