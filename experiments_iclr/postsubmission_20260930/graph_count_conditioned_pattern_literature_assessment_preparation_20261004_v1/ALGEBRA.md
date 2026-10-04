# Algebraic assessment, no numerical execution

Fix visible context `C`, member `m`, side `s`, finite slot logits `eta_msi`, support size `n_s`, and observed count `k_s`. Teacher count is used only in the auxiliary label branch. All sums below retain every slot and every feasible subset.

## 1. Conditional Bernoulli and direct gradient

Let `w_i=exp(eta_i)` and `e_k(w)=sum_{|A|=k} prod_{i in A} w_i`. Independent Bernoulli probabilities with these odds satisfy

`p(z)=exp(eta·z)/prod_i(1+exp(eta_i))`.

Conditioning on `sum_i z_i=k` cancels the product denominator:

`q(z | C,k)=1[|z|=k] exp(eta·z)/e_k(exp eta)`.

For a valid teacher pattern, `ell=log e_k(exp eta)−eta·z`. Its gradient is `pi_i−z_i`, where `pi_i=E_q[Z_i]`. Since `sum_i pi_i=k=sum_i z_i`, the gradient sums to zero. Adding `a` to every side logit multiplies numerator and denominator by `exp(a k)` and leaves `q` unchanged. At `k=0` or `k=n` there is one supported pattern, so `ell=0` and every slot gradient is zero, including the empty side `n=k=0`.

The stable forward recursion is `E_0^(j)=1` and `E_r^(j)=E_r^(j−1)+exp(eta_j)E_(r−1)^(j−1)`, restricted to `r≤min(j,k)`. In log space its sum is logaddexp. The complement identity is

`log e_k(exp eta)=sum_i eta_i + log e_(n−k)(exp(−eta))`.

Thus a normalizer can use `O(n min(k,n−k))` scalar recurrence operations, plus input processing; a rolling forward table uses `O(min(k,n−k))` storage. Autograd, marginals, checkpointing, neural graph state and heterogeneous-batch packing have additional work/storage. No large-batch time, memory, numerical or native-kernel feasibility follows from this arithmetic bound. Zero-supervision sides still pay their preserved native neural schedule.

## 2. Mixture, memberwise loss and limits

Write `a_m=q_m(z_L|k_L)q_m(z_R|k_R)` and `ell_m=−log a_m`.

`J_K=−log[(1/M)sum_m exp(−ell_m)]`,

`W_K=(1/M)sum_m ell_m`,

`r_m=exp(−ell_m)/sum_j exp(−ell_j)`.

Then `dJ_K/deta_msi=r_m(pi_msi−z_si)` while `dW_K/deta_msi=(pi_msi−z_si)/M`. Shared-parameter gradients sum these paths. Jensen gives `J_K≤W_K`; also `min_m ell_m≤J_K≤min_m ell_m+log M`. Their training contrast is soft assignment versus component competence. Identical component laws give identical losses and uniform responsibilities, so symmetry/collapse is allowed. A lower mixture NLL or larger member spread is not a diversity or ranking guarantee.

Within a member, subset-swap odds are `q(A−{i}+{j})/q(A)=exp(eta_j−eta_i)`. The other selected identities do not enter this ratio. Between sides, the matrix indexed by feasible patterns is `sum_m u_m v_m^T/M`, with nonnegative rank at most `M`. If one side has a unique pattern, no nontrivial cross-side association remains.

For an analytic example only, take `n_L=n_R=2`, `k_L=k_R=1`, and two equally weighted members with logits `(a,−a)` on both sides and `(−a,a)` on both sides respectively. Put `p=sigmoid(2a)`. Each marginal side is uniform; the shared mixture assigns each matching identity pair probability `(p²+(1−p)²)/2` and each mismatching pair `p(1−p)`. Independent side mixtures assign every pair `1/4`. Counts are identical throughout. This is a representation argument; no numerical fixture or predictive test was run.

## 3. Uniformly mixed conditionals differ from conditioning a mixture

Let the unconditioned component be `p_m(z_L,z_R)=p_m(K_L,K_R)q_m(z_L,z_R|K_L,K_R)` and its prior member weight be `1/M`. Conditioning the original mixture gives

`p_mix(z | K)=sum_m alpha_m(K)q_m(z|K)`,

`alpha_m(K)=p_m(K)/sum_j p_j(K)`.

The proposed `J_K` instead fixes weights at `1/M`. The two are equal when count evidence gives equal weights, or accidentally when the conditional components coincide, but are not generally equal. A correct old-mixture decomposition is

`−log p_mix(z)=−log p_mix(K)−log p_mix(z|K)`

with these reweighted components. Substituting `J_K` into that equality would be false. Under the old mixture, member-specific common logit offsets can change `alpha_m(K)` even though every component's conditional law is unchanged.

## 4. Exact cancellation of C_mu

The frozen density is `p_C(z|C) ∝ exp(eta_L·z_L+eta_R·z_R+g_C(K_L,K_R))`. Holding both counts fixed yields

`p_C(z|C,k_L,k_R)= exp(eta_L·z_L)/e_(k_L)(exp eta_L) × exp(eta_R·z_R)/e_(k_R)(exp eta_R)`.

`g_C` cancels, even if its visible context depends differentiably on native logits or features. Its direct and contextual derivative paths cancel algebraically. The head's slot map only supplies this count potential, so every parameter used solely by that head gets zero conditional-NLL gradient. This statement requires conditioning on both side counts: fixing only their sum generally leaves a nonconstant `g_C(k_L,K−k_L)`.

Frozen serving uses detached unary marginals from the unconditioned count law. Target BCE therefore supplies no path through the count head; the specified weight decay is zero. Replacing the whole count-head auxiliary by this conditional NLL would leave that head without its old training signal. This does not diagnose the actual frozen C_mu experiment, whose auxiliary remains unconditioned and unchanged.

## 5. Conditional analogue of F_P

The actual frozen F_P likelihood is the product of per-slot pooled probabilities, not the mean component NLL. Define `bar_p_si=(1/M)sum_m sigmoid(eta_msi)` and `tilde_eta_si=log[bar_p_si/(1−bar_p_si)]`. Conditioning that factorial pooled law on both counts gives

`q_FK(z|K)=prod_s exp(tilde_eta_s·z_s)/e_(k_s)(exp tilde_eta_s)`.

It is one conditional-Bernoulli law per side. It has fixed-budget dependence within each side but no additional cross-side mixture association. Its normalization and pooling order differ from both `W_K` and `J_K`. In particular `tilde_eta` is neither mean member logit nor mean conditional marginal logit. Its conditional unary marginals generally differ from the mixture's conditional marginals, and member-specific common logit shifts can affect it through probability pooling even though they leave every component conditional law unchanged. J_K versus F_K therefore changes more than cross-side association.

For a shared-member-specific comparison use `q_sep=[(1/M)sum_m q_mL][(1/M)sum_m q_mR]`. This matches each whole per-side marginal law and all its unary marginals at fixed parameters. `J_K−F_K` includes pooling/conditioning order, within-side multimodality and cross-side association; `J_K−J_K_sep` more narrowly isolates use of one member identity across sides. Neither replaces a capable identity-sensitive single or a target-ranking comparison.

## 6. Serving and graph claims

Conditional training discards information identifying common side offsets and count probabilities. Zero-sum output gradients do not constrain all parameter updates to preserve unconditioned predictions in a shared nonlinear network. The teacher pattern is observed TRAIN incidence under a mask, not true missing-graph structure. The target predictor uses native completion and pooled raw target logits, not the auxiliary conditional law, and does not know teacher counts at serving. Local density accuracy, repeated-context agreement, or component responsibilities therefore cannot by themselves establish a coherent graph posterior or a useful served completion.
