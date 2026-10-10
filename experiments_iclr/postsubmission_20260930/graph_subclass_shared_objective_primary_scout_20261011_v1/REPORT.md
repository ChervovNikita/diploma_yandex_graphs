# Graph contexts, within-class geometry and shared objective allocation

**Disposition:** two bounded primary documents scoped: one new method and one recent weighted-objective analysis. Zero whole papers, proofs, author implementations or experimental results adopted. One conditional hypothesis/control is retained; no implementation or fit is admitted. The running context source is preserved.

## Answers

Different class-compatible similarity relations can be assigned to different private projection spaces of one encoder. **MLCL explicitly does this.** This permits a shared representation to support more than one auxiliary geometry, but does not demonstrate useful graph-defined modes, preservation of predictive competence, or an advantage from serving several predictive members.

The closest inspected source for **different positive relations allocated to shared private heads** is now MLCL. The saved AMCL scope already supplies private contrastive heads, but its inspected positive relation remains common. MLCL's heads are auxiliary: its hierarchical experiment discards them and serves one representation through a linear classifier. Saved SBCL/C3GNN/SoftTriple already establish subclass or multiple-center ingredients in their respective settings. These are attributed building blocks.

## New bounded sources

1. Ghanooni, Pajoum, Rawal, Fellenz, Duy and Kloft. [Multi-level Supervised Contrastive Learning](https://arxiv.org/html/2502.02202v1), arXiv:2502.02202v1, 4 February 2025. Read **§§3.1–3.4**, retained blocks21–53. Shared encoder `f`, private `g_h`, independently defined positive sets `P_h(i)`, weighted sum of per-head SupCon objectives, and per-head temperatures are explicit in Eqs1–3. Given subclass/superclass labels or aspect labels determine the relations. Its global head also weights positives by a supplied label-overlap similarity. The hierarchical setting removes projectors and trains a classifier on the frozen common representation; the text setting combines CE with the auxiliary heads. No inferred graph contexts or predictive ensemble bank is established in this scope.

2. Vock, Duchesnay and Dufumier. [A Unified Geometric Framework for Weighted Contrastive Learning](https://arxiv.org/html/2605.13943v1), arXiv:2605.13943v1, 13 May 2026. Read **§§3.1–3.8**, retained blocks20–81. The objective already allows arbitrary nonnegative pair weights. Its analysis assumes symmetric weights and symmetric similarities, relates positive weights to `log W` target scores, and distinguishes Euclidean from spherical realizability. Soft SupCon retains constant within-class weights and its stated ideal still collapses each class. The continuous-label section defines y-Aware Euclidean weights and X-CLR exponential cosine teacher weights, including a spherical teacher/student construction. This is direct ancestry for continuous teacher affinities, not a new graph learner nomination.

The second preprint's existence statements contain notation inconsistencies: some write `s=exp(metric)` although the displayed objective already exponentiates `s`, and one writes an unsquared norm after defining squared distances. Proofs were not read. This packet uses its objective definitions and the explicit elementary ratio/Gram conditions below; it does not import those theorem statements verbatim. Published-version equivalence is unverified for both sources.

## The current membership rule does not require within-class differences

Let every normalized hidden vector in class `c` equal a prototype `u_c`. With `k_c=min(16,n_c−1)` uniform selected positives, the masked loss of a class-c anchor is

\[
\log\left[k_c e^{1/\tau}+\sum_{a\ne c}n_a e^{u_c^\top u_a/\tau}\right]-1/\tau.
\]

Which classmates the context selected has disappeared. All four contexts have the same value at this configuration. This establishes a permissive configuration, not the actual outcome or global optimum of the mixed native objective. In addition, the derivative of cosine between nonzero parallel vectors is zero. Target-dependent same-class gradients can vanish even for nonuniform weights. Cross-class and CE gradients can remain nonzero.

## What continuous weights change—and their limits

For fixed normalized graph signatures `u_m`, define within-class weights

\[
w^m_{ij}=\exp((u^m_i)^\top u^m_j/\tau_T),\qquad
Q^m_{ij}=w^m_{ij}/\sum_{k\in C_i}w^m_{ik},
\]

where `C_i` contains every other TRAIN classmate. Cross-class target weights are zero. Although `W` is symmetric within a class, row-normalized `Q` need not be symmetric; its class restriction is reversible with row-sum stationary mass.

With all other TRAIN nodes in the denominator, let `a_i` be the student softmax conditional on `C_i`. The complete loss decomposes as

\[
L_i=-\log P_i(C_i)+H(Q_i)+\mathrm{KL}(Q_i\|a_i).
\]

A collapsed class makes `a_i` uniform, so nonuniform `Q_i` leaves a positive conditional KL term. The target requests unequal pair ratios. However, changing geometry can worsen the class-mass term, and CE adds another tradeoff. Therefore nonuniform targets alone do **not** prove a noncollapsed global optimum for the bounded cosine/mixed/native objective. They also do not guarantee SGD escapes parallel-vector degeneracy.

For free normalized vectors, matching teacher ratios requires a Gram matrix `G` satisfying

\[
G_{ij}=(\tau/\tau_T)K^m_{ij}+b_c\quad (i\ne j,\ y_i=y_j=c),
\]

with a shared within-class constant, global symmetry, positive semidefiniteness, unit diagonal and rank at most the hidden width. Cross-class entries must admit a compatible completion. Copying normalized teacher vectors is sufficient for conditional ratios when temperatures match and teacher rank fits the student width. At lower student temperature, the spherical construction with an additional constant coordinate gives another sufficient case when rank plus one fits. No teacher ranks or model realizability were inspected here.

Hard cross-class zeros cannot equal a finite all-TRAIN softmax, whose probabilities are positive. Full entropic equality is unavailable for this exact support at finite cosine temperature. Conditional ratio feasibility is narrower than full objective feasibility.

## One retained conditional hypothesis/control

**Class-conditioned continuous graph-context teacher weights versus uniform all-classmate weights.** If root later admits a comparison, preserve native SAGE, one factual forward, own CE, auxiliary scale0.05, student/teacher temperatures0.2 and the four fixed signatures. Replace top-16 membership by continuous weights over every classmate. The decisive control uses uniform weights over exactly the same classmates and the identical complete denominator. There is no adaptive teacher, projector, additional parameter or served forward.

Useful evidence would be better complete served predictions than the uniform control while preserving mean/worst-member competence and NLL. A shared-recipient interpretation additionally requires capable ordinary/factorized singles, genuine independent I4 and common-objective shared references, each equipped with the complete four-objective mixture. Lower auxiliary KL or more representation spread is insufficient. No benefit versus the uniform control, or competence/NLL harm, refutes this fixed utility hypothesis. Gains matched by those capable references do not establish a shared-recipient advantage.

For this proposed common full denominator, averaging the four targets equals averaging their complete outside-log objectives. That identity does not validate union-mask shortcuts in the frozen masked family.

The minimal three-seed candidate/control pair would require six fresh shared-bank fits if admitted. Teacher affinities are a one-time computation; the loss retains the same580×580 score dimensions as existing full-denominator arms. Capable references require additional fresh fits. No wall-time estimate or scientific operation is adopted.

An earlier private-projection comparator draft is preserved under `superseded/` and is explicitly withdrawn from the retained queue. MLCL remains the strongest objective-allocation prior and a potential capable comparator requirement, not a second proposed family.

## Custody and accounting

See `READ_SCOPES.json` for versioned bytes, hashes and exact boundaries; `REUSED_BINDINGS.json` for saved conclusions checked first; `GEOMETRY_AND_COLLAPSE_LIMITS.json` for precise algebraic limits; and `ONE_CONDITIONAL_CONTROL.json` for the sole retained conditional comparison. Discovery-only biological perturbation and emotion-task abstracts were not promoted to method reads. HTTP429 metadata failures are preserved. No scientific host, current prediction payload, TEST, model import, fit, canonical ledger or manuscript was touched.
