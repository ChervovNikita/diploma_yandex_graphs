# Quality-preserving diversity after the completed MolBACE pilot

15 October 2026. **Recommendation: add no new pilot or diversity loss.** The
useful next contrast is already prepared: the same private input correction
trained with own-member risk F versus `.5F+.5L_mean`. Class-conditional
decorrelation is a different objective from current alignment/variance, but
the retained literature and completed findings do not yet justify adding it.
“Quality-preserving” should describe an observed outcome, not a promised property.

This note uses completed selected-VALID reports and existing source/literature
records. No tensors, checkpoints, models, TEST payloads or remote hosts were
opened. No implementation, training, roster change or novelty claim follows.

## 1. What the completed findings support

The three-seed MolBACE/GIN study has one scaffold VALID population already used
for checkpoint selection. Its original complete analysis and the subsequent
complete aggregation sensitivity report give the following **mean-logit** results:

| Method | VALID AUC mean ± sample SD (%) | Mean BCE |
|---|---:|---:|
| Native M1 | 73.321 ± 1.172 | .779 |
| Genuine I4 | 74.847 ± .500 | .890 |
| Base GNNM | 67.778 ± 4.390 | .719 |
| Alignment + variance | 71.172 ± 2.657 | .551 |
| Private input correction | 71.758 ± 3.907 | .941 |
| Both ingredients | 69.280 ± 1.801 | .882 |
| Private BN follow-up | 73.614 ± 1.057 | 1.086 |

The combination's AUC interaction is negative in all three seeds (mean −5.87
points); its reported approximate interval includes zero. This motivates
separating ingredients, not diagnosing gradient interference. Private BN has
the strongest observed shared-method mean AUC, while geometry has much lower
BCE. A competence constraint based only on BCE could prefer a worse ranker.
Neither metric supplies a universal ordering of these methods.

The complete fixed-aggregation analysis also found that replacing mean logits
with mean probabilities improves correction BCE from .941 to .638 while its
AUC changes from 71.758% to 71.587%. Thus probability-loss improvement can come
from the readout without new useful ranking information. The serving rule must
remain fixed in the proposed contrast below. These are descriptive, selected
validation findings, not confirmation or transfer evidence. [E1–E3]

The retained seed101 diagnostic is narrower evidence: an unusually disagreeing
route had weak AUC and a large feature norm, while almost every graph had some
correct route. Opposing class predictions can create that oracle coverage.
The report does not establish that private corrections need more repulsion,
that all-member errors are conditionally redundant, or that BN caused the
weakness. No three-seed gradient/decorrelation diagnosis is invented here. [E4]

## 2. Conditional residuals: a legitimate distinction with no quality guarantee

Current C1 aligns same-class live preheads and penalizes low **within-class
sample variance within each member**. It does not penalize cross-member
conditional dependence. Its variance penalty can have zero derivative at
exact collapse. A new cross-member conditional covariance/independence term
would therefore change the objective; it is not an alias of C1. [S1]

The motivation already has direct ancestry: Ramé and Cord's **DICE** preserves
task information while targeting conditional feature redundancy. Its actual
conditional joint/product discriminator and bottleneck/CR ablation differ
from a centered-cosine or covariance penalty. The retained implementation
review also identifies estimator/sampling/scale and state-policy qualifications.
No DICE graph benefit or conditional-independence guarantee transfers to a
four-route live shared backbone. [P1]

Three simple checks limit a new residual proposal:

1. **Class-centering binary prediction errors removes the label value.** With
   `e_mi=p_mi−y_i`, within class c,
   `e_mi−mean_(j:y_j=c)e_mj = p_mi−mean_(j:y_j=c)p_mj`.
   Labels select the group, but the centered value contains no residual
   correctness target. Zero conditional covariance is compatible with identical
   constant, uninformative predictors. Correlation normalization or a variance
   floor changes this proxy; neither creates a competence theorem.
2. **Bank-centered repulsion can reward a change invisible to serving.** Let
   `r_m=H_y(z_m−zbar)`, where H_y centers example rows within observed TRAIN
   classes. Since `sum_m r_m=0`,
   `sum_(m<n)<r_m,r_n> = −.5 sum_m ||r_m||²`.
   Minimizing this signed pair sum merely rewards residual energy. Scaling
   `z_m→zbar+a(z_m−zbar)` changes that energy while preserving the entire
   mean-logit predictor. A squared covariance penalty is a different objective;
   it still needs protection against collapse and a demonstrated task benefit.
3. **Hidden residual separation need not affect decisions.** Changes in a
   classifier's hidden nullspace can change hidden distances without changing
   logits. Conditional second moments are not conditional mutual information,
   and removing shared within-class variation can remove useful shared evidence.
   No direct implication for AUC/BCE follows. [P1,P2]

The binary setting also blocks a literal ADP wrong-class log-determinant port:
after excluding the true class, each positive normalized vector has one entry,
so four members' Gram matrix has rank one. Its determinant is zero. Ridge or
pseudodeterminant variants would be different objectives. This is the retained
ADP dimensional argument specialized to MolBACE, not a new literature result. [P3]

## 3. Prediction-quality constraints: define the protected quantity

Ordinary per-route supervised risk, the C6 route-weight floor and positive
background supervision encourage competence; none guarantees improvement over
M1/I4 or even over the starting bank. Conversely, requiring every route to beat
M1 would discard potentially useful weaker members, as retained TabM findings
already show. Quality must be assessed jointly with the actual served pool. [P4,S2]

A tempting teacher-free guard is pointwise
`BCE(y,z_m) ≤ BCE(y,zbar)` for every member. **At zero slack this forces identical
binary logits.** For y=1 BCE is strictly decreasing, so every z_m must be at least
their mean; all must equal it. For y=0 the inequalities reverse and give the
same conclusion. Detaching the mean changes derivatives, not this feasible set.
Nonzero slack introduces a new choice; it does not certify heldout ranking.

Aggregate TRAIN-risk bounds against a prior state avoid that pointwise collapse
but protect only their specified examples, loss and reference. They do not
protect AUC or an unseen scaffold population. Finite Adam proposals, moment
state and accept/reject policies also need their own semantics; the retained
guarded DICE/FoRDE source review explicitly treats such guards as changes to the
learner. No teacher, rollback controller, tolerance grid or constraint solver
is nominated by this note. [P1]

## 4. One falsifiable contrast, already in the source inventory

**Hypothesis:** with exactly the same four private nonlinear input corrections
and shared native body, directing part of training at the deployed mean improves
served ranking without merely trading away member probability quality.

| Arm | Existing source identity | Training/readout |
|---|---|---|
| Control | `portable_input_residual_be_v1` in `portable_private_input_v2` | F; mean raw logits |
| Method | `interaction_input_meanJ` in `interaction_node_molecular_v1` | J=.5F+.5L_mean; mean raw logits |

Both retain one live shared native backbone, four BE routes, the same rank-cap-8
zero-up input residual, source categorical inputs, native batch/reductions and
optimizer, and no teacher or added readout parameters. All blocks receive each
arm's scalar objective. Match the BN policy, seeds, split, initialization,
source controller and selection contract prospectively. **The old completed
molecular pilot is motivation, not automatically a byte-compatible control:**
the interaction package explicitly distinguishes its new states from the old
pilot. Do not retrospectively combine the best BN/geometry settings. [S2–S4]

For binary mean-logit serving, suppressing the unchanged regularizer,

```
F = L_mean + D,
D = mean_m softplus(z_m) − softplus(mean_m z_m) >= 0,
J = L_mean + .5D.
```

Thus this exact existing contrast halves the own/pool Jensen penalty at a fixed
state. It does not add explicit repulsion. The ancestry is GNCL's own/pool
interpolation and the retained loss-matched diversity identity. Pooled training
can also exploit compensating weak members—the retained learner-collusion
warning—so retaining `.5F` is encouragement, not a quality guarantee. [P2,P5,P6]

**Falsifier:** if the complete paired comparison does not improve the declared
served VALID AUC relative to its matched F control, reduced conditional
correlation, a larger Jensen gap or stronger TRAIN fit does not rescue the
hypothesis. If AUC improves while served BCE/member competence deteriorate,
report that trade-off; do not call it quality preservation. Report all paired
seed differences, served BCE, per-route AUC/BCE, mean/worst member quality and
the exact F/L/D bookkeeping at the same selected state. The stronger utility
claim still needs competent M1 and I4 references. No numerical success margin,
coefficient search, new selector or TEST opening is added here.

**Disposition:** retain this existing comparison and the existing alignment-only
and BN controls. Add no separate “quality-preserving residual decorrelation”
pilot on this evidence. The conditional-dependence question is distinct but
currently lacks a diagnosed predictive deficit and a validated quality proxy;
the actionable own/pool hypothesis already has source coverage.

## Reading accounting and citations

**New primary papers: 0. New full-paper reads: 0. New primary/code retrievals: 0.**
Existing exact conditional-diversity and own/pool sources already address the
question; no additional paper was needed. Named prior results below are reused
from retained scoped conclusions/reviews, not recounted as fresh full-text or
abstract reads. Current local source inspected: the method tables and scalar
loss paths, without imports/execution. Exact local receipts/read scopes are in
`SOURCES.json` and `READ_SCOPE.json`.

- **P1:** Ramé and Cord, *DICE: Diversity in Deep Ensembles via Conditional
  Redundancy Adversarial Estimation*, ICLR 2021, [2101.05544v1](https://arxiv.org/html/2101.05544v1),
  §2.1–2.2, Eq1/5; retained source review adds Appendix B/F.2 and distinguishes
  the CR ablation from full DICE. FoRDE comparison/guard limits are from the
  retained 3 October independent source review and corrected v2 specification.
- **P2:** Wood et al., *A Unified Theory of Diversity in Ensemble Learning*,
  JMLR 24(359):1–49, 2023, [primary](https://jmlr.org/papers/v24/23-0041.html),
  retained pp9–14/Corollary9 scope; no fresh theorem or proof audit.
- **P3:** Pang et al., *Improving Adversarial Robustness via Promoting Ensemble
  Diversity*, [1901.08846v3](https://arxiv.org/html/1901.08846v3), §§3.1–3.3/Eq3–6,
  retained wrong-class Gram dimensional assessment.
- **P4:** Gorishniy et al., *TabM: Advancing Tabular Deep Learning with
  Parameter-Efficient Ensembling*, [2410.24210v3](https://arxiv.org/html/2410.24210v3),
  retained §5.1 member/collective-quality scope; no graph-performance transfer.
- **P5:** *Generalized Negative Correlation Learning for Deep Ensembling*,
  [2011.02952v2](https://arxiv.org/html/2011.02952v2#S4.E5), Eq5, reused from the
  retained decision-diversity assessment; no new primary read.
- **P6:** Jeffares, Liu, Crabbé and van der Schaar, *Joint Training of Deep
  Ensembles Fails Due to Learner Collusion*, [2301.11323v1](https://arxiv.org/html/2301.11323v1),
  retained §5/Eq3 and Appendix E scope; no blanket failure theorem for this bank.

E1–E4 and S1–S4 resolve to the exact local evidence/source records in `SOURCES.json`.
