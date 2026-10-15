# Analytical limits of the implemented C3 contrast

These are source-level derivations for the frozen criterion, not experimental results, new literature claims or a proposed method replacement. Let w_i denote the existing nonnegative anchor/reduction weight, common across routes and arms; for NCN/NCNC its total may be two rather than one. All identities apply rowwise before this reduction.

## 1. Target linearity and entropy

Write C(q,p)=-sum_j q_j log p_j, qbar=(qA+qB)/2, and groups A={0,1}, B={2,3}.

    A_common = (1/4) sum_m sum_i w_i C(qbar_i,p_mi)
    A_split  = (1/4) [sum_(m in A) sum_i w_i C(qA_i,p_mi)
                     + sum_(m in B) sum_i w_i C(qB_i,p_mi)].

C is linear in its first argument: C(qbar,p)=(C(qA,p)+C(qB,p))/2. With d_i=qA_i-qB_i,

    A_split - A_common
      = -(1/8) sum_i w_i d_i dot (log p_0i + log p_1i - log p_2i - log p_3i).

Define the assignment-alignment statistic

    T = sum_i w_i d_i dot [mean_(m in A) log p_mi - mean_(m in B) log p_mi].

Then A_split-A_common=-T/4 at the same bank. T>0 measures correspondence between the fixed target assignment and donor likelihoods. It is directly rewarded by SPLIT, so it is a fit diagnostic, not an independent predictive success criterion. A four-by-two KL matrix is more interpretable than this scalar alone.

For each row,

    C(q,p)=H(q)+KL(q||p)
    H(qbar)-[H(qA)+H(qB)]/2 = JS(qA,qB) >= 0.

The unconstrained common-target entropy floor therefore exceeds the average split-target floor. Subtract the actual target entropy before comparing fit residuals. This removes a reporting constant; it does not make target geometry or optimization difficulty equal. All donor masses are strictly positive on the identical compatible support, so the entropy difference is not a support-count difference in current C3.

If s_m are donor logits, dC/ds_m=p_m-q_m. Since mean_m q_m=qbar, mean donor-score cotangents of COMMON and SPLIT agree at any fixed bank. For shared parameters theta and route-specific score Jacobians J_m,

    grad_theta(A_split-A_common)
      = (1/4) sum_m sum_i w_i J_mi^T (qbar_i-q_mi).

This cancels if the relevant route Jacobians are equal. Equal p_m alone is insufficient. Private parameter derivatives need not cancel. The shared/private optimizer trajectory is not made identical by target linearity; no claim about Adam, stochastic layers or later states follows.

## 2. A common target may be outside the scorer's family

For one compatible class/relation block, let K_ij=exp(cos(h_i,h_j)/tau) for i!=j and K_ii=0. K is symmetric. If Z_i=sum_j K_ij, then

    p_ij=K_ij/Z_i,
    Z_i p_ij = K_ij = Z_j p_ji.

Thus every such donor kernel is reversible. In particular, for any three distinct eligible objects,

    p_12 p_23 p_31 = p_21 p_32 p_13.

Both factual kernels individually have this property. Their arithmetic mean need not, because the two normalizing/stationary measures can differ. Satisfying reversibility is only a necessary condition: temperature, Gram positivity, dimension and the native architecture impose additional restrictions.

### Exact three-object example, valid at the frozen tau=0.2

Choose unit context-A vectors with Gram matrix

    G_A = [[1,       0,       tau*log(3)],
           [0,       1,       tau*log(2)],
           [tau*log(3), tau*log(2), 1]].

This is a valid positive-definite Gram: it is symmetric and strictly diagonally dominant, since every off-diagonal row sum is at most tau*log(6)<1. Three-dimensional vectors therefore realize it. Let context-B vectors be three orthonormal vectors. Exponentiated off-diagonal affinities are A: K12=1, K23=2, K13=3; B: every Kij=1.

The row targets are

    qA = [[0, 1/4, 3/4], [1/3, 0, 2/3], [3/5, 2/5, 0]],
    qB = [[0, 1/2, 1/2], [1/2, 0, 1/2], [1/2, 1/2, 0]],
    qbar = [[0, 3/8, 5/8], [5/12, 0, 7/12], [11/20, 9/20, 0]].

For qbar the forward cycle product is (3/8)(7/12)(11/20)=77/640. The reverse product is (5/12)(9/20)(5/8)=75/640. They differ by 1/320. Consequently no symmetric cosine-softmax scorer can fit this common target exactly, at any prehead dimension. Each constituent target is individually representable by that scorer family. This does not prove simultaneous realizability by a particular shared native architecture, nor that the actual data have a material obstruction.

**Practical diagnostic:** for already preserved TRAIN q rows, inspect the target three-cycle log residual

    R_ijk = log qbar_ij + log qbar_jk + log qbar_ki
            - log qbar_ji - log qbar_kj - log qbar_ik.

Use a fixed, identity-based subset of triples within compatible blocks if enumerating all triples is too costly; record that sampling law and actual cost. Report magnitudes and coverage, not merely nonzero floating-point tests. No triples exist for blocks of size two. This is target analysis and creates no model arm. Large residuals limit a split-over-common claim to the full objective comparison; they do not isolate complementary prediction learning. Do not silently replace arithmetic mixing with score/geometric mixing to repair the frozen control.

## 3. Signal and collapse qualifications

- With no donor the implementation omits the row. With exactly one donor, p=q=1 and both CE and its gradient are zero. An eligible-row count alone overstates nontrivial supervision.
- If all normalized nonzero preheads within a compatible class/relation block coincide, every supported cosine is 1 and its derivative with respect to each nonzero prehead is zero. Nonuniform q can yield a nonzero score cotangent but zero backpropagated auxiliary gradient at this state. C3 does not guarantee preventing or escaping directional within-class collapse.
- The penalty constrains normalized within-class relationships, not radial norms, other-class geometry or directly the native label logits. There can be auxiliary-fitting directions unused by the downstream prediction map. Which such directions are available is architecture-dependent; their existence in a fitted bank has not been measured here.
- qA!=qB can still be a tiny perturbation, can emphasize a nuisance, or can be largely unrealizable within the native shared representation. Fixed .05 credit and zero extra model passes establish neither signal strength nor predictive benefit.

The exact identities above do not use a dataset, model import, numerical optimizer or synthetic fit. No executable experiment was run.
