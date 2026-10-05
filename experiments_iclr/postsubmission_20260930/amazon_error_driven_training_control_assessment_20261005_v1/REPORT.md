# One attributed training control from the Amazon error diagnosis

5 October 2026. Source/literature/theory only. This report uses root's completed three-split VALID error summary and saved conclusions; it opens no metric shard, label/logit/tensor/checkpoint payload or remote session. It proposes no new GPU cohort, implementation, tuning grid or change to the running private-transfer pilot. The next hypothesis below is an **attributed training control**, not a new method or novelty gap.

## What the completed diagnosis motivates

Root reports all nine native split/predictor panels and 44,514 aggregate rows. The following are root-supplied three-split descriptive summaries, not independently recomputed findings:

| Quantity | Shared4 | Independent4 |
|---|---:|---:|
| Mean member accuracy | 52.2402% | 52.5859% |
| Pooling gain over mean member accuracy | +0.1633 percentage points | +0.5907 percentage points |
| Mean six-pair error correlation | 0.9292 | 0.7304 |
| All four members wrong | 44.55% | 35.05% |
| Erroneous prediction confidence | 92.5966% | 81.0860% |

The shared-minus-independent accuracy gap is -0.7730 percentage points: -0.3457 from mean-member accuracy and -0.4274 from pooling gain. Root's FP64 Brier decomposition gives a +0.110735 shared disadvantage: +0.011436 mean-member Brier difference and +0.099299 from reduced probability ambiguity/averaging benefit.

This motivates checking both member competence and ensemble complementarity. It does not identify a causal sharing collapse. Confidence/calibration, initialization, optimization, private capacity and base selection can contribute. Error correlation is descriptive; all-member errors are not proof of missing score information. The separately processed capable single and independent ensemble in the ongoing fixed CPU screen remain decisive practical references. Their outcomes may remove the need for a new training route.

## One next training hypothesis

**Retain direct native supervision of every member, while adding a fixed task-error coupling term that reduces common probabilistic errors; accept complementarity only when member strength and complete served quality are retained.** Apply this first as an ordinary shared-ensemble training control. Keep the actual Amazon architecture and its arithmetic probability-pool serving endpoint fixed. Do not add a gate, hidden export, graph view or new private architecture in the same hypothesis.

On the existing permitted FIT population F, define

    p_m(v) = softmax(z_m(v)),  Y_v = onehot(y_v),  e_m(v) = p_m(v)-Y_v,
    L_own = mean_(v in F,m) CE(z_m(v),y_v),
    R = mean_(v in F) [1/choose(M,2) * sum_(m<n) e_m(v)^T e_n(v)].

One candidate control objective is L_own + lambda*R, with one prospectively fixed nonnegative coefficient. Every route keeps its own CE. The coupling uses all FIT targets, not selected VALID mistakes or favored classes/degree bins. No VALID error mask, TEST label or score-derived selection enters training. A later design must fix the coefficient, optimizer, complete horizon, selector and quality margins before new outcomes; this assessment allocates none.

R is an **uncentered cross-error moment**, not Pearson correlation. For one-hot Y and simplex probabilities, the true-class residual components are nonpositive and every other component nonnegative, so each same-node pair inner product is nonnegative. Its reduction can result from genuinely smaller member errors or from redistribution of errors. It does not ensure lower binary double fault, lower conditional correlation or better accuracy. In particular, keeping own CE in the objective is an anchor, not a theorem that member quality is preserved.

### Exact own/pool Brier equivalence

Let B_own=mean_(v,m)||e_m(v)||^2 and B_pool=mean_v||mean_m e_m(v)||^2. Elementary expansion gives

    B_pool = B_own/M + (M-1)*R/M,
    R = (M*B_pool-B_own)/(M-1).

For M=4, R=(4*B_pool-B_own)/3. Therefore the proposed CE-plus-R objective is exactly CE_own + lambda*(4*B_pool-B_own)/3. If the competence anchor is Brier itself, then

    B_own + lambda*R
      = [1-lambda/(M-1)]*B_own + [lambda*M/(M-1)]*B_pool.

For 0<=lambda<=M-1 this is, up to a positive common scale, an ordinary scalar own/pool Brier interpolation. Outside that range the own-Brier coefficient is negative and the competence tradeoff becomes especially important. This algebra is the reason to call it an **NCL/GNCL-style training control**. Relabeling it as a covariance loss, graph-diversity loss or new complementarity principle would not create novelty. An algebraically identical GNCL comparator is the same arm and should not be fitted twice.

The saved ambiguity identity is B_pool=B_own-A, where A=mean_(v,m)||p_m(v)-p_bar(v)||^2. Thus increased ambiguity helps Brier only relative to the simultaneous change in member Brier. The observed shared disadvantage is a useful diagnosis of that tradeoff; it does not prove that maximizing A will repair it. CE, Brier, top-1 accuracy and calibration remain different endpoints.

## Inherited primitives and decisive controls

Saved index72 and active supplements were consulted before any retrieval. No missing primary method was needed, so new primary/full-paper reading counts are zero.

| Saved precedent | What is inherited; relevant limit |
|---|---|
| GNCL, arXiv:2011.02952v2, saved Eq.5 conclusion; Dynamic NCL, DOI:10.12792/iciae2026.015 | Scalar own/pool and adaptive strength/diversity tradeoffs are prior. The dynamic paper's printed ambiguities remain unresolved; do not port its literal equations. |
| FoRDE, arXiv:2306.02775v3, saved paper/author-source correction | True-label input-gradient kernel repulsion is prior. The selected author recipe uses raw true-class logits; graph differentiation cost and numerical qualification cannot be inherited from images. Input-gradient separation does not certify complementary task errors. |
| DICE, arXiv:2101.05544v1, saved method/Appendix F.2 | Label-conditioned latent redundancy minimization, its discriminator and noisy deterministic ablation are prior. A useful representation objective still needs competent optimization and full quality evaluation. |
| Learner Collusion, arXiv:2301.11323v1; TabM/BatchEnsemble and saved mixed-block analysis | Joint own/pool objectives, member competence, shared/private factors and block-dependent use of established losses are prior. Pooled fitting can exploit compensating errors; an own-loss anchor offers no universal protection. |
| BMAML/ANIL/MLDG/OML/SELAR and retained private-transfer conclusions | Private learning, meta-gradients, persistent state and recomputation have direct ancestry. Their conjunction in the current pilot remains an unconfirmed configuration/utility question. |

A later comparison must preserve these decisive references without weakening them:

- **Ordinary separately trained independent four:** full encoders/heads, independent initializations, complete competent schedules and declared selection opportunities. A jointly trained four-model arm is not a substitute for this reference.
- **Capable modern single:** fully trainable nonlinear predictor with an adequate paid schedule and generic probability/calibration regularization where relevant. It cannot be reduced to the shared ensemble's tiny factor-only private block.
- **Ordinary shared own-loss training and the exact NCL/GNCL control:** same architecture/data/serving/selection, with no duplicate fit for identical own/pool algebra. Apply the same proposed coupling to an untied independent architecture if attributing a benefit to sharing; a generic regularizer can help both.
- **Competent ordinary training regularization:** native regularized CE plus a simple probability-fit/calibration control is necessary because confidence alone can explain gains. Source-qualified DICE/FoRDE are established alternatives if a future claim specifically asserts superior diversity learning; match permitted information and disclose their different optimization and full costs. Failed auxiliary learning or copied image schedules cannot support superiority.

These are control requirements, not a new cohort or grid. The already running CPU processors and private-transfer family keep their existing frozen rules. Lower correlation without complete pooled NLL/Brier/accuracy improvement is not success. Improvement purchased by weaker members supports a strength/diversity tradeoff, not preservation of competence. Replicated complete comparisons and prospectively frozen confirmation are needed for a quality claim; reused VALID remains retrospective development.

## What a graph/shared-core-specific contribution would need beyond this control

The diagnosis does not supply a missing novel operator. A meaningful extension would have to specify **how graph-conditioned private learning changes the shared representation's ability to support competent, complementary corrections**, beyond the scalar own/pool risk tradeoff above. Degree-dependent coefficients, hidden centering, generic repulsion or an added router alone do not establish that mechanism.

It would need a concrete dependency on graph context/learning that cannot be removed by rewriting its objective as ordinary NCL, plus a reason this dependency retains target-relevant private correction capacity. The graph condition should have a matched ablation with the same permitted labels, exposure, support and selection. A capable single and an untied version using the same operation must test generic learning versus the benefit of a shared ensemble. Member-strength and pooled-gain decompositions should show the claimed correction, not merely feature spread. Exact source/prior comparison would still be required for novelty; a different conjunction or a positive score alone is insufficient.

The running private-transfer pilot is the existing concrete attempt in this direction: it learns a shared core through endpoint-separated private supervised updates, then recomputes committed private state. Its live/detached and endpoint/random contrasts test learning credit and graph-conditioned supervision; its capable single, untied four and ordinary references test attribution. Keep that pilot unchanged. It uses NCN/link objectives and mean raw-logit serving, while the Amazon diagnosis concerns node classification and arithmetic probability pooling. Neither this Brier coupling nor Amazon percentages automatically transfer to its ranking endpoint. If the capable single benefits equally, the evidence favors generic training regularization; if untied four wins, tying remains a quality compromise.

**Disposition:** use the diagnosis to interpret the current CPU screen and private-transfer results, and retain the simple error-coupled objective only as an attributed training control. It is insufficient for the user's novelty objective and does not admit a new GPU cohort. Root alone decides later execution. No canonical, manuscript or current-study artifacts were edited.
