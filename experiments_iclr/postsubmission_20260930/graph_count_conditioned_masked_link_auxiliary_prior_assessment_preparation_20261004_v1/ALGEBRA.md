# Exact support and categorical reductions

Symbolic analysis only, 4 October2026. Inherited conditional-Bernoulli/mixture algebra is bound to the sealed preceding assessment. No numerical execution or predictive claim.

## Fixed-count binary law

For one side with n canonical binary slots, z in {0,1}^n, k=sum_i z_i and member logits eta_m,

q_m(z|k)=1[sum z=k] exp(eta_m dot z)/e_k(exp eta_m),

where e_k(w)=sum_{S:|S|=k} product_{i in S} w_i. Loss ell_m=log e_k(exp eta_m)-eta_m dot z. Its gradient is pi_mi-z_i, where pi_mi is its conditional inclusion marginal. Sum_i pi_mi=k, so the gradient sums to zero. A common side logit shift cancels. At k=0 and k=n the only feasible pattern has probability1 and loss/gradient0.

This is an established conditional Bernoulli law. It is also a diagonal k-DPP: set L=diag(exp eta_m), so det L_S=product_{i in S} exp eta_mi and Eq.175 of the monograph is exactly q_m. A non-diagonal kernel changes the set law; it is not a generic arbitrary-association comparator.

ESP recurrence E_j(t)=E_{j-1}(t)+w_j E_{j-1}(t-1), with E_0(0)=1, gives scalar O(nk) work. Complementing the pattern uses -eta and n-k and gives O(n min(k,n-k)) scalar work. These are arithmetic descriptions, not native batch memory/runtime qualification. Stable log-domain implementation and differentiation need their own static/numerical checks outside this packet.

## Two sides and one component identity

Write a_m=q_mL(z_L|k_L), b_m=q_mR(z_R|k_R). The proposed probability is Q=(1/M)sum_m a_m b_m and J_K=-log Q. Responsibilities r_m=a_m b_m/sum_j a_j b_j give gradient r_m(pi_msi-z_si) on either side. This is ordinary posterior responsibility weighting for a mixture.

J_K_sep=-log[(1/M)sum_m a_m]-log[(1/M)sum_m b_m]. At fixed logits, both models have identical full per-side distributions and therefore identical unary marginals. The shared model uses one member index; the separate model draws two independent indices. The difference is cross-side association. Their gradients differ (joint r_m versus side-specific responsibilities), so separately trained models need not retain equal per-side competence.

W_K=(1/M)sum_m[-log(a_m b_m)] fits every member to the pattern instead. Jensen gives J_K<=W_K at fixed parameters, but says nothing about held-out ranking after training. J_K_sep and J_K have no universal ordering. If all members coincide, the shared/separate distinction vanishes. If one side has an extreme count, its factor is1 in every member and J_K=J_K_sep for that query.

Index feasible left/right patterns as A,B. Q_AB=(1/M)sum_m q_mL(A)q_mR(B) has nonnegative rank<=M. Within a component, exchanging selected slot i for unselected slot j changes odds by exp(eta_j-eta_i), independently of the other selected identities. A mixture relaxes this but is not an arbitrary fixed-count set law or globally consistent graph posterior.

## k=1 is categorical, exactly

For observed positive slot i,

q_m(i|1)=exp eta_mi/sum_j exp eta_mj=softmax(eta_m)_i,

ell_m=logsumexp eta_m-eta_mi.

This is ordinary categorical cross-entropy. With scores eta=s/tau and one designated positive among a candidate pool it has the InfoNCE formula. For M=1 and k_L=k_R=1, J_K is the sum of two such losses. This is not a new objective.

For M>1 and observed pair (i,j),

Q_ij=(1/M)sum_m softmax(eta_mL)_i softmax(eta_mR)_j.

Define pair score B_ij=log Q_ij. Sum_{i,j} exp B_ij=1, so

J_K=-B_ij=logsumexp_{a,b} B_ab-B_ij.

It is ordinary categorical cross-entropy over a Cartesian pair alphabet, with a mixture-of-products parametrization. The representational restriction and shared responsibility coupling differ from two separate side losses. Neither the generic categorical loss nor mixture-responsibility operation is new. CPC's Eq.4 supports the formula; its density-ratio/MI reading needs one positive from the conditional and proposal negatives as described there. Candidate supports and masked TRAIN observation zeros do not automatically satisfy that sampling interpretation.

The same categorical observation applies to k=n-1: if j is the omitted slot,

q_m(z excluding j|n-1)=exp(-eta_mj)/sum_a exp(-eta_ma).

Consequently k>1 alone is insufficient evidence of multielement subset behavior. The genuine subset stratum has min(k,n-k)>1. If n<=3, every informative count is categorical or complement-categorical. Cross-side mixture coupling can still matter in those strata, but it is categorical pair association.

## k>1: multinomial and multi-positive alternatives are different laws

For a binary selected set S of size k, the recommendation multinomial log likelihood (up to an eta-independent combinatorial constant) is sum_{i in S} eta_i-k logsumexp eta. Equivalently it sums k categorical log probabilities. It allocates mass to count vectors permitting repeated items. The fixed-count binary-subset NLL is instead log e_k(exp eta)-sum_{i in S} eta_i, normalized only over distinct k-subsets. They agree for k=1 and differ generally for 1<k<n.

Conditioning that multinomial further on every item count being at most1 recovers the same conditional Bernoulli law: its binary-set mass is k! product_{i in S} pi_i, its total binary-set mass is k! e_k(pi), and normalization gives product_{i in S} pi_i/e_k(pi)=exp(sum_{i in S} eta_i)/e_k(exp eta). This is an own symbolic relation, not a claim that the inspected MultiVAE implementation applies that conditioning.

An aggregated multi-positive contrastive loss -log[sum_{i in S} exp eta_i/sum_j exp eta_j] trains membership in a positive group; it is not the likelihood of the exact k-subset. Sequential without-replacement softmax selection (Plackett-Luce) is also generally different: its denominators depend on the selected prefix/order. It must not be silently substituted for ESP normalization. Any fixed-support model can still be viewed as categorical CE over its feasible subsets; the distinction is support and parametrization, not the invention of cross-entropy.

## Conditioning and pooling order

Uniformly mixing already conditioned member laws gives J_K. Conditioning the old unconditioned mixture instead gives weights proportional to each member's probability of the observed counts. These distributions are generally different. A conditional F_P analogue that first pools slot Bernoulli probabilities and then conditions also has different per-side marginals; it is not the J_K_sep matched-distribution control. A potential depending only on (k_L,k_R) cancels once both counts are conditioned, so the old C_mu count potential supplies no identity-sensitive capability in this comparison.

The teachers are observed TRAIN incidence/counts under the prescribed masks. Serving remains pooled raw target logits with the existing completion/decoder. The conditional auxiliary is not the served probability law. Shift invariance, count removal, starvation/collapse, local overlap inconsistency and target-risk mismatch block any unconditional transfer guarantee.
