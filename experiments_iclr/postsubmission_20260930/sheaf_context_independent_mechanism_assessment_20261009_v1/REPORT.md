# Independent assessment of the shared sheaf context mechanism

Date: 2026-10-09. Scope: a proposed learning rule, assessed by static source inspection and algebra. This is not a completed-paper review, experimental result, or publication verdict. No author claim of novelty or utility is evidence. Exact reading scopes are recorded in `READ_SCOPES.json`.

## Finding

The analytic no-message reference is correct for the original general NSD recurrence, and a geometry-only auxiliary recipient is mathematically coherent. The proposed contrast supplies **teacher-dependent logit adjustment**. On the proposed binary task its private-parameter gradient is also **exactly an adaptive weighting of ordinary context CE**. A gain over the specified fixed teacher-error weighting would identify a different weighting policy, not a new graph-evidence statistic. The fixed 64-query panel and teacher trained on those queries can make TRAIN correction competence misleading; transfer and source dependence need explicit prospective checks.

## 1. The exact native reference and gradient recipient

The original general model constructs a pointwise stem with input dropout, `lin1`, optional ELU, hidden dropout and optional `lin12`, then stores it as `x0` [N1:259–270]. Each layer transforms the current state, multiplies by the live sparse operator, optionally applies ELU, and updates

`H_(l+1) = C_l H_l - ELU(message_l)`,

where `C_l = diag(1+tanh(epsilon_l))` repeated over nodes [N1:271–289]. Zeroing every propagated term gives `H_L^0 = (product_l C_l) H_stem`; ELU(0)=0, and the shared linear classifier gives the stated analytic logits [N1:294–296; D1:27–35]. The stem must be the exact realization captured from that forward. Re-running the stem with fresh dropout would not be the same reference. Subtracting native log probabilities instead of logits is valid because the difference is a class-constant shift.

With private factors only in incidence learners, this null has no direct dependence on η. For each auxiliary forward, holding θ constant while differentiating the entire live context path into η yields the specified update [D1:55–61; D2:37–41]. Blocking θ parameters must not detach higher hidden states, maps, normalization or logits: those contain recurrent paths into earlier η. Detaching only the null also does not implement the stated recipient, because the context logits depend on shared θ. The teacher and frozen assignment have no derivative. η receives both mean-own full-task NLL and the scaled auxiliary gradient.

This is a restricted update rule, not joint gradient descent on `F + lambda*J` in all parameters. Auxiliary changes to η can still alter later θ updates through F. Independence of the immediate null from η does not provide a general optimization or generalization guarantee. No implementation or numerical gradients were inspected or executed.

## 2. The contrast has an exact conventional description

For one query and one captured native realization, let `z` be context logits, `z0` null logits and `r` frozen teacher probabilities. With τ=1,

`J = CE(z + a, y)`, where `a = log(r) - z0`.

Equivalently, up to a class-constant shift, `a = log(r) - log(p_null)`. Since a is independent of η during the update, this is ordinary supervised CE with an offset; all graph-dependent information still comes from the original context forward. If `r=p_null`, it is exactly context CE. The design itself records this identity [D1:65–73], but the identity, rather than that narrative, establishes the conclusion.

There is a stronger reduction specific to the binary Tolokers task. Define `p_ctx=softmax(z)`, `p_corr=softmax(z+a)` and

`alpha_q = [1-p_corr(y_q)] / [1-p_ctx(y_q)]`.

For finite logits, the two binary CE logit gradients are parallel, and

`grad_eta J_q = stopgrad(alpha_q) * grad_eta CE(z_q,y_q)`.

Thus a dynamically weighted ordinary context CE, using the displayed detached weight, gives exactly the same private gradient at every state. With the same starting point, assignment, stochastic draws and optimizer, it describes the same η update. This is an algebraic equivalence, not an empirical conjecture. It need not hold as a scalar-weight equivalence for arbitrary multiclass offsets.

The specified ordinary-context comparator uses the different, frozen weight `w_q=1-r(y_q)` [D2:51–54]. It is not gradient matched. For example, at d=0 with a uniform binary null, the candidate's correct-class logit-gradient magnitude is w_q, while the weighted native CE magnitude is w_q/2 at the same λ. Away from d=0 the discrepancy varies with the current logits. A candidate win over that comparator could reflect effective gradient strength or the adaptive weighting curve. It cannot distinguish contrast from conventional logit adjustment or adaptive hardness weighting. An exact detached-weight implementation is useful for qualification, but supplies no distinct scientific comparator.

At evaluation, every auxiliary query has zero input features. The original stem is pointwise and shared; dropout is off. Consequently `z0(q)` is identical across those queries, contexts and members. The null subtraction there is a common class offset, while `log r(x_q)` provides the query-dependent local-feature offset. During training the null can also vary because of the captured dropout realization [D1:19–21; N1:259–270].

## 3. Prediction difference is not source-feature evidence

The null removes the epsilon residual bypass, but it does not subtract a prediction on an all-zero-feature **live graph**. `lin1` and `lin12` are ordinary biased linear layers [N1:243–246]. Zero raw rows can therefore have nonzero stems. The live operator contains both diagonal blocks and off-diagonal blocks [N2:310–335]. Block-degree normalization uses incident-map sums and augmented inverse square roots, with training jitter and clamping [N2:275–301]. Hence d can include transformed constant/bias signals and structural class priors. With normalized irregular support or stochastic training states, source features can all be zero without forcing d=0.

The ordered endpoint map learner is a linear map plus activation of current endpoint states [N3:28–59]; it is not an externally supplied evidence measurement. State-dependent normalization can also transmit dependence through neighboring map degrees, so a graph-distance≤L eligibility rule is a conservative panel definition, not an exact statement of all feature dependencies [N2:285–290,316–320].

A minimal source test is a paired **all-zero-source context** replacement: set every feature row to zero while retaining the original Q_k, labels, teacher, assignment, live topology, recipient and context-forward budget. If it recovers the candidate's full-input or transferred correction gain, source-feature evidence is unnecessary. The existing within-context feature-identity permutation is useful but does not test this same question [D3:16]. The right-feature-matrices-zero sanity test checks that the propagation path is unused; it does not show that surviving propagation uses informative source features [D1:77].

## 4. Why the panel and teacher can mislead

The same fixed 64 labelled TRAIN queries are used to allocate contexts after warmup and repeatedly train the auxiliary loss; all are already targets of the primary F [D1:19–23,49–59; D2:25–35]. An endpoint-conditioned graph model can distinguish repeated queries through their neighborhoods. Increasing their TRAIN competence, or observing assignment-dependent scores on them, supplies no out-of-panel evidence. With as few as eight eligible queries and only a requirement that both classes occur, some context scores can be dominated by a handful of targets. Equal 32/32 stratification also changes the auxiliary class prior if the authentic TRAIN distribution is imbalanced; report that policy rather than treating it as representative NLL/calibration evidence.

The teacher is frozen, but it was trained on these same query labels [D1:37; D2:21]. It may predict them very confidently through in-sample fitting even when its unseen feature-only predictions are weak. At d=0, auxiliary gradient magnitude depends directly on this in-sample error probability. A teacher that memorizes Q suppresses the intended residual request; a confidently wrong or miscalibrated teacher amplifies selected queries. Good full-VALID teacher performance alone does not establish representative teacher residuals on Q. These issues are label reuse and generalization concerns, not extra-truth leakage.

The planned new VALID queries are a useful transfer test [D2:103; D3:14]. Their eligibility, query population, coverage and class/patch reporting should be fixed explicitly before training. Use all eligible VALID queries, or a prospective sufficiently large panel, rather than a favorable small subset. These logits can be read from the already required full context forwards. VALID scores obtained after selecting models on full-input VALID remain exploratory evidence, as the design already acknowledges [D2:35,107].

## 5. Minimal decisive revision

1. Describe J directly as geometry-recipient teacher-offset CE, or binary adaptive context-example weighting. Drop a separate contrast mechanism claim; no experiment can separate two algebraically identical descriptions. If retaining the fixed teacher-hardness arm, state that its auxiliary gradients are not matched and record their magnitudes before attribution.
2. For the auxiliary teacher, prospectively exclude Q from its TRAIN fitting, or use cross-fitted predictions on Q. Freeze the architecture/selection rule and charge the extra teacher work. A full-TRAIN teacher can remain the ordinary feature-only benchmark. Preserve the same auxiliary teacher for every compared arm.
3. Add the paired all-zero-source replacement above and fix an out-of-panel VALID context assay with explicit coverage. Require useful original-input improvement and transfer beyond the fixed TRAIN targets; do not use TRAIN context scores to claim learned graph evidence.

After these changes the remaining question is useful geometry-recipient context specialization at a measured budget. It is not the existence of private sheaf ensembling, a Bayesian posterior, or a new evidence contrast. The saved closest prior already identifies shared-network sampled-sheaf ensembling and coherent-mode/structured-latent alternatives [P1:5–9,34–50]. Its controls address persistence, private geometry and joint multi-transport capacity [P1:80–90]; this static assessment establishes none of their empirical performance.

## Supporting sources

All paths below are local. Cited ranges indicate the portions supporting the corresponding statements, not whole-paper reading credit.

| ID | Source and supporting lines |
|---|---|
| D1 | `shared_sheaf_context_contrast_design_20261009_v1/METHOD.md`: 9–13, 19–45, 49–79 |
| D2 | `shared_sheaf_context_contrast_design_20261009_v1/EXPERIMENT.json`: 7–41, 43–81, 101–108 |
| D3 | `shared_sheaf_context_contrast_design_20261009_v1/FALSIFIERS.md`: 7–18 |
| N1 | `private_sheaf_native_source_design_20261009_v1/source_custody/nsd/models/disc_models.py`: 206–296, especially 243–246, 259–296 |
| N2 | `private_sheaf_native_source_design_20261009_v1/source_custody/nsd/models/laplacian_builders.py`: 262–336 |
| N3 | `private_sheaf_native_source_design_20261009_v1/source_custody/nsd/models/sheaf_models.py`: 14–108 |
| P1 | `shared_body_private_geometry_closest_prior_20261009_v1/REPORT.md`: 5–11, 34–50, 64–70, 80–98 |

All source paths are relative to `/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/`. No dataset, checkpoint, scientific outcome contents, server or numerical/model execution was accessed. Existing design/source/ledger/manuscript files were not edited; this separate packet is the only output.
