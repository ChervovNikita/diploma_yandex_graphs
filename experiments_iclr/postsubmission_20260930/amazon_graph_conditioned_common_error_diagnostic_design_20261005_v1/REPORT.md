# Graph-conditioned common-error diagnostic before a training proposal

**Status:** sealed conceptual design; source, saved conclusions and theory only. No diagnostic was executed. No training method, hidden-state extraction, GPU cohort or compute admission is proposed by this packet. The running private-transfer pilot remains unchanged. Root alone decides execution.

## 1. Decision and motivation

The useful next question is whether the shared bank has an excess of *the same wrong class recurring across graph neighbors* after controlling local member weakness and confidence. The existing error summary establishes highly correlated errors and reduced pooling benefit. It does not establish where those errors recur, whether their wrong competitor agrees across nearby nodes, or whether a graph-dependent shared-core training mechanism produced them.

Specify one paired, matched neighborhood recurrence contrast. Its primary population contains VALID targets where both four-member banks are unanimously wrong at the **same** competitor. This fixes target correctness, member-correct counts and wrong competitor for the two banks. Match each eligible FIT neighbor to a FIT nonneighbor with the same true class, public degree band, both banks' member-correct counts, native-single correctness and nearby confidence. Use the identical targets, neighbors and controls for both banks and the native single.

This is an observational diagnostic on selected checkpoints. A positive contrast would identify excess graph-conditioned identical-class errors in this population. It would not identify message copying, oversmoothing, representation collapse, causal harm from sharing, or shared-core gradient interference. A negative or unsupported result would not prove that all graph/shared-core training mechanisms fail.

## 2. Evidence already available

These outcomes were supplied by root and are not recomputed here. The original analysis covered three splits and three native predictors, with 44,514 aggregate rows. Its exact output shards were not opened for this note.

| Original selected-VALID quantity | Shared four | Independent four |
|---|---:|---:|
| Mean member accuracy | 52.2402% | 52.5859% |
| Pooling gain | +0.1633 pp | +0.5907 pp |
| Mean six-pair error correlation | .9292 | .7304 |
| All four members wrong | 44.55% | 35.05% |
| Erroneous confidence | 92.5966% | 81.0860% |

The shared accuracy difference is −.7730 pp, decomposed into −.3457 pp in mean member quality and −.4274 pp in pooling gain. The shared-minus-independent Brier difference is +.110735, comprising +.011436 in mean member Brier and +.099299 in the reduced ambiguity contribution. These selected-VALID descriptions do not establish causal sharing collapse.

Root also supplied a completed frozen aggregation screen that was still under root audit at the time of this note: processed single 53.1276%, processed shared 52.5450%, processed independent 53.2528%; shared fails the full gate. The final root audit packet governs that screen. Its negative result rejects those frozen operators and budget, and does not demonstrate that every score/graph function must fail or that target information is absent. The strongest processed independent and single remain quality references; the proposed diagnostic does not replace their quality gate.

## 3. Artifact and label-custody preconditions

1. Preserve the original split identities, selected checkpoints, member identities, native scoring, class ordering, precision and tie rule. Individual members and the native single use raw FP32-logit argmax. A four-member native pool uses argmax of the mean FP32 softmax probabilities. Do not substitute mean-logit pooling or apply a calibration wrapper for the diagnostic.
2. Use the authenticated public graph and its already bound canonical undirected edge recipe. A neighborhood excludes self edges and duplicate undirected edges. Public degree uses this complete public graph, including nodes whose labels are unavailable. If its existing recipe differs, root must bind the recipe before any diagnostic; do not silently change the graph interpretation after viewing results.
3. Targets are the complete original VALID set. Anchors and controls are the original allowed FIT set. They require explicitly authenticated, authorized FIT-only score views for all nine native predictors and an allowed FIT-only label projection. FIT IDs and class ordering must also be bound.
4. The unchanged V2 custody loader authenticates the original closure and exposes **VALID-only** compact labels/views for the existing analysis. That does not by itself authorize a FIT label read or a broader score view. A full official TRAIN pack may also contain TRAIN-control labels. Do not decode it on the assumption that FIT authority covers the pack. Root must establish a safe, authenticated projection before this diagnostic can run.
5. No TRAIN-control or TEST labels, feature tensors, hidden states, checkpoint payloads, Jacobians or training replay are needed for this diagnostic. Public topology can be used without hidden labels. This packet has read none of those payloads.

FIT predictions are in-sample predictions from the selected models. Therefore an absence of FIT errors can leave this diagnostic without support. A support failure is not evidence that VALID errors contain no graph pattern. The graph may be transductive, and checkpoint selection is retrospective; neither fact gives a causal intervention.

## 4. Preserve the complete error partition

For each VALID target, retain the native single's correctness, both banks' member-correct counts (0–4), both banks' confidence summaries and the following mutually exclusive native-pool strata:

| Pool status | Required retained distinctions |
|---|---|
| Both correct | Complete denominator and member-correct distributions |
| Shared wrong, independent correct | Shared loss; shared wrong competitor and shared erroneous unanimity |
| Shared correct, independent wrong | Shared recovery; independent wrong competitor and independent erroneous unanimity |
| Both wrong, same competitor | Both / shared-only / independent-only / neither erroneously unanimous |
| Both wrong, different competitors | Both / shared-only / independent-only / neither erroneously unanimous |

“Erroneously unanimous” means that all four raw-logit member argmaxes equal the bank's wrong competitor. The primary stratum is the **both-wrong, same-competitor, both-unanimous** cell. Report it as a fraction of all VALID targets and of all shared losses/errors. It is intentionally narrow and cannot by itself explain the full accuracy gap or shared-only errors.

Keep every stratum's splitwise denominator and matched-support accounting, including empty strata. Do not select a favorable stratum, pool away different wrong competitors, change the target population, or promote a secondary stratum after observing the primary result. The other strata provide the complete descriptive context; this note specifies only one primary recurrence contrast.

## 5. One frozen matched-neighborhood contrast

### Events and eligible neighbors

For a bank f in {shared, independent}, let a_fm(x) be member m's native raw-logit argmax and let p_fm(x) be its FP32 softmax vector. Let k_f(x) be the number of members whose argmax equals the allowed label y_x. Define

`U_f(x,c) = 1{a_f1(x) = a_f2(x) = a_f3(x) = a_f4(x) = c and c != y_x}`.

For the native single, `U_0(x,c) = 1{a_0(x) = c and c != y_x}`. Define confidence `q_f(x) = mean_m max_j p_fm,j(x)`, and native-single correctness `k_0(x) = 1{a_0(x) = y_x}`. These definitions do not use a fitted ensemble weight or a learned diagnostic.

For primary target v, write its common wrong competitor as c_v. Eligible neighbors are exactly the FIT anchors u adjacent to v with **y_u = y_v**. This defines a same-true-class confusion recurrence estimand. It conditions away label homophily within the analyzed anchor pairs and does not describe heterophilous edges. Since c_v differs from y_v, a prediction of c_v is wrong for every eligible anchor and control.

### One matching recipe

For each eligible neighbor u, candidate controls a are FIT nodes that are not adjacent to v and have:

- `y_a = y_u`;
- the same public degree band `D(x) = floor(log2(1 + degree(x)))`;
- exactly the same `(k_shared, k_independent, k_0)`;
- `abs(q_shared(a) - q_shared(u)) <= .05` and `abs(q_independent(a) - q_independent(u)) <= .05`.

Use one deterministic greedy 1:1 match, with no control reuse within a target. Process neighbors in the node-ID hash order under the fixed salt `common-error-diagnostic-20261005-v1`; choose the unused admissible control minimizing the sum of the two absolute confidence differences, with a salted node-ID hash tie break. The salt, degree bands and .05 caliper are fixed by this note. Control reuse across different targets is permitted and must be reported. No matching decision may use c_v-specific unanimity, a particular wrong-class identity at the anchor/control, or the eventual recurrence contrast.

Matching both banks' correct-member counts controls coarse competence for both families on the same anchor pair. The confidence caliper controls coarse confidence; it does not produce exact exchangeability or remove all calibration differences. Report actual within-pair confidence differences and count signatures. The raw-argmax event itself is unchanged by a positive scalar temperature for each member, except numerical/tie qualifications. Thus a change in confidence scale alone cannot turn a fixed raw unanimous competitor into a different one.

Unmatched neighbors remain unmatched. Do not widen the caliper, merge degree bands, drop the single-correctness match, move to VALID anchors, or alter the FIT population to rescue support. Fixed matching uses allowed labels for retrospective analysis and is not a serving rule.

### Estimand, weights and references

For matched pairs (u,a) of target v, calculate the bank-specific graph-minus-nonneighbor recurrence:

`E_f(v) = mean_(u,a matched at v) [U_f(u,c_v) - U_f(a,c_v)]`.

The splitwise primary contrast is

`Delta_s = mean_(matched primary targets v in split s) [E_shared(v) - E_independent(v)]`.

Give each matched target equal weight, independently of its degree or number of matched pairs. Report the unweighted mean of the three splitwise contrasts only alongside all three values. Every bank and the single use the identical target/pair set. Also report E_shared, E_independent and E_0 on that set; the single is a reference for graph-conditioned class difficulty, not a directly exchangeable four-member-unanimity event.

Report raw neighboring and matched-nonneighbor recurrence separately. A difference of differences can otherwise hide whether the pattern comes from neighbors, controls, or both. Retain native single correctness at the primary targets. For context, retain the complete stratum table and original whole-VALID quality results.

### Support and descriptive reporting

For each split, report: all primary targets; targets with at least one eligible neighbor; targets with at least one match; eligible and matched edge counts; unmatched edge counts by the fixed signature; unique neighboring FIT anchors; unique FIT controls; maximum control reuse across targets; and actual confidence distances. Report all denominators before reporting Delta.

The frozen provisional descriptive-support floor is at least **100 matched primary targets, 100 distinct neighboring FIT anchors and 100 distinct FIT controls in every split**. If any split falls below any floor, label the primary diagnostic **INSUFFICIENT_SUPPORT**, still show its descriptive counts/contrast, and retain the fixed recipe. These floors prevent a tiny supported subset from being presented as a supported mechanism result; they are not a power calculation or an inferential guarantee. Passing them establishes neither representativeness nor causality.

Edges, anchors, controls and splits can overlap. Do not report an independent-node or independent-edge p-value. This note requests descriptive splitwise estimates and support only. A significance claim would require a separately justified dependence/uncertainty design frozen before its outcomes, not an opportunistic bootstrap. No new test grid or adaptive analysis is admitted here.

## 6. Interpretation that the artifacts can support

| Observed outcome of the fixed design | Supported interpretation | Limit |
|---|---|---|
| Positive shared-minus-independent Delta with adequate reported support | Shared has excess neighbor-specific recurrence of the fixed wrong competitor, conditional on this primary population and these matches | Residual feature/community difficulty, graph context, selection and other unmeasured differences remain |
| Shared and independent/single show similar neighbor recurrence | A graph-conditioned difficulty pattern is compatible with ordinary predictors; sharing specificity is weakened | Single and four-member unanimity have different marginals; equality does not prove a common cause |
| Delta is near zero despite the previously high global error correlation | The global correlation need not imply excess graph-conditioned recurrence after these competence/confidence controls | Does not quantify how much weakness/calibration caused the accuracy gap; matching is coarse and supported targets are selected |
| No excess with adequate support | No evidence for this specific recurrence pattern in the supported same-class FIT-neighborhood population | Does not cover shared-only VALID losses, heterophilous edges, unsupported nodes or other graph functions |
| Insufficient support | Allowed FIT outputs cannot sustain this comparison under the frozen recipe | Do not infer absence of a graph mechanism or missing target information |

The complete partition is needed because “all four wrong” is not “all four predict the same wrong class,” and a high pairwise error correlation can arise from shared difficult nodes without a common competitor. The primary contrast deliberately holds a common target competitor fixed and compares its recurrence against same-node-class nonneighbor baselines. It is more discriminating than a global error-correlation or confidence summary, while remaining observational.

Even a positive result would not show that edges carried that wrong class during training or inference. Matched nonneighbors need not be exchangeable with neighbors; they may share two-hop context, and communities/features can remain unbalanced. Degree and allowed labels cannot supply the absent features or counterfactual graph intervention. This design can motivate a bounded mechanism question, not answer it.

## 7. Why shared-core gradient interference remains unidentified

Allowed labels and saved logits can supply the cross-entropy **logit cotangent** `r_m(v) = p_m(v) - Y_v` for each labeled node. The shared-parameter gradient is

`g_theta,m(v) = J_theta,m(v)^T r_m(v)`,

where J is the member's derivative of logits with respect to the actual shared parameters. The saved outputs contain r, but not J, its layer paths, loss weighting or parameter metric. A residual cosine/correlation therefore does not identify the sign or magnitude of a shared-parameter gradient inner product. At a fixed output value, different local derivatives can yield aligned, orthogonal or opposed parameter gradients while preserving that output. Output similarity alone cannot resolve the missing derivative.

Hidden states would add representation observations, but hidden states alone also lack those parameter Jacobians and objective paths. Any claim of current shared-gradient conflict needs separately authorized source/checkpoint replay with per-member VJPs or equivalent gradients of the actual loss with respect to the same shared parameter set. The parameter set, loss, label support, stochastic state and gradient inner product/metric must be bound. A checkpoint-local replay would establish only local behavior at that checkpoint. A claim that such interference caused the learned errors additionally needs a training intervention/trajectory argument and controls; it cannot be read backward from selected logits.

No gradient cosine surrogate, NCL algebra or learned output weighting is proposed here.

## 8. Evidence needed before a graph/shared-core training extension

Before a new GPU method proposal, the hypothesis must have a source-defined graph-conditioned private-learning dependency and a reason that this dependency changes the shared-core update. “Graph errors correlate” is not that dependency.

A credible later proposal would need all of the following, with any replay or experiment separately admitted by root:

1. **Graph specificity:** an operation-level comparison that removes or perturbs the graph condition while preserving labels, exposure, support, private-update opportunities and comparable capacity. Compare it with a generic context dependency of the same scope; a graph-conditioned operation should earn its graph claim.
2. **Sharing specificity:** ordinary independent four, a capable single, and the same operation with untied cores. A shared bank should improve the complete served quality gate while preserving member strength. A weaker control or a selected error-cell rescue is insufficient.
3. **Mechanism evidence at the claimed level:** actual shared-parameter gradients/Jacobians for a gradient-interference claim, or source-defined graph paths plus an intervention for a message claim. Saved-logit recurrence and hidden-state similarity alone do not identify either mechanism.
4. **Complete outcomes:** all frozen endpoints and the strongest processed independent/single references, together with support and cost. Better recurrence/diversity is an explanatory observation, not a substitute for served quality.
5. **Closest-prior comparison:** a precise distinction from established graph residual correction, graph-aware calibration, propagated-bank attention/label utilization, local-pattern expert routing and generic diversity objectives. A positive diagnostic supplies no novelty clearance.

The running private-transfer pilot already has its own frozen endpoint/random, live/detached, single, untied-four and ordinary references. Its link/ranking endpoint and fixed mean-raw-logit serving differ from this Amazon probability-pool diagnostic. Do not change that pilot or transfer Amazon Brier/probability findings to it by analogy.

## 9. Saved literature reused; no new primary reading

The relevant saved conclusions were sufficient for this note:

- **C&S, arXiv:2010.13993v2:** propagation of labeled prediction residuals is established. Spatial error recurrence is not a novel primitive.
- **GATS, arXiv:2210.06391v1:** neighbor logit similarity, confidence and distance to training nodes inform calibration. Argmax preservation depends on a positive scalar temperature and the implementation/tie qualifications.
- **Calibration analysis, arXiv:2206.01570v1:** neighbor-prediction ratio informs ratio-binned calibration; true neighbor labels occur in explanatory analysis. Label-conditioned diagnostic value is not a serving authorization.
- **GAMLP, arXiv:2108.10097v3:** propagated feature banks, attention and label utilization are relevant ancestry.
- **MoE-NP, arXiv:2412.00418v3:** random-walk/edge/degree patterns and expert routing are relevant ancestry.
- **Saved round-five alignment conclusion:** cross-node member correspondence concerns members, not hidden channels; centered private deviations can hide common backbone errors. Familiar covariance does not establish information or novelty.

Sources are bound in SOURCE_BINDINGS.json. No paper was retrieved or reopened, no full-paper reading certification was added, and no canonical literature index/manuscript was edited. The current decision is **DIAGNOSTIC_DESIGN_ONLY; NO_IDENTIFIED_TRAINING_MECHANISM_OR_NEW_COMPUTE_ADMISSION**.
