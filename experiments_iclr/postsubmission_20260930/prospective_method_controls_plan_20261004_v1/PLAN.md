# Smallest prospective comparison for correlated NCNC completion

4 October 2026. Source-only plan. No implementation, fit, data reader, result, checkpoint, SSH, VALID/TEST traversal or execution release is supplied. Input source identities and reading scopes are bound in `INPUT_BINDINGS.json`.

## 1. Next comparison

Prepare **J, F and an autoregressive structured single (A)** on the complete native Collab task, with three fresh, prospectively fixed paired seed blocks. J/F retain the frozen four-member, width-64 shared-encoder family: identical 43,790 parameters, native target BCE, detached completion, labels, masks, optimizer and mean-raw-logit serving. Only the auxiliary differs. This is the clean capacity-matched contrast; its difference can still operate through representation learning or regularization.

A is a capable joint-distribution quality comparator. Use it before claiming that four associated target predictors are necessary. The existing count control C and covariance control S remain subsequent explanation controls: C restricts spatial dependence within count pairs; S restricts the target interface to moments. Neither exhausts full-context structured singles.

Keep 100 epochs, 17 batches of 65,536, all 100 complete official VALID shared-negative Hits@50 selector candidates, first exact tie, and two complete selected-state replays. Select using served ranking; auxiliary fit and favorable subgroups cannot substitute for it. No TEST stage is proposed.

## 2. Implement the common contract first

Adapt new J/F training to the prepared replay provider. Persist actual negative pairs and full permutation/tail, then qualify equality of every record mask, masked graph, query and ordered common/left/right support. Old hashes cannot reconstruct old realizations. These are new repetitions of the frozen hypothesis, with model-independent stream RNG.

All arms receive the same permitted features, masked TRAIN graph and native slots, without caps. Construct complete-TRAIN observation labels only after enumeration; zero means unobserved in TRAIN. Teacher state, record IDs and receipt bookkeeping remain outside predictor features. Separate dropout, replay and structural-draw RNG. Preserve empty queries and the original positive/negative reductions.

## 3. Implement one full-context joint single

A has one fresh native width-64 encoder/scorer and one target decoder. Add a two-block causal transformer: width 64, four attention heads, feed-forward width 128. Each residual token contains its allowed slot/context features, left/right role and native calibrated scorer logit. Use the native deterministic order, all left then all right, without learned sorting or truncation.

Define a proper law `p_A(Z|C)=product_i Bernoulli(Z_i; sigmoid(eta_i + delta_i(C,Z_<i)))`. Teacher forcing supplies previous observation targets **only to the density-training branch**; no teacher prefix enters target ranking. The auxiliary is exact chain-rule NLL divided by residual count, with lambda 1 and the same query reductions. This changes the conditional density construction, while preserving the teacher and source information.

For target training and serving, draw four complete residual vectors ancestrally using an exact causal KV cache. Apply weights `1.05 Z_i` to every native residual slot and reuse the same nonlinear target decoder for all four draws. Detach the density/draw route from main BCE; retain ordinary encoder/decoder main gradients. Train mean BCE over draw logits and serve their mean raw logit. Fix validation draw keys by unlabeled query identity; charge all draws. Four draws approximate a nonlinear decoder expectation and require numerical qualification.

This is a declared adaptation, not a parameter-matched ablation: it changes soft completion to binary completion, adds a head and changes the main prediction approximation. Report its actual parameters and work. A win or loss alone does not identify higher-order information.

## 4. Falsify the proposed mechanism

The defensible question after PIFM is whether **component-wide observation-pattern supervision improves ranking through member-specific completion/decoder association in this shared-backbone learner**. Joint graph refinement itself is already supplied by the comparator. A finite mixture is also one distribution, and a grouped single can reproduce the bank exactly; “ensemble necessity” is not established by its naming.

At selected J/F states, reuse one fixed scorer bank and hold recipient features/decoders fixed. Score own association, the three cyclic completion reassignments, and pooled clamped weights using the prepared routes. Compare the association penalty in J with that in F. A penalty shared equally by F is evidence of ordinary specialization, not a distinctive J mechanism. These interventions never select checkpoints; charge all complete candidate scores.

On one predeclared common TRAIN diagnostic trace, record joint NLL, marginal NLL/Brier separately for observed-positive and source-zero slots, and responsibility/spread summaries. Add one identity-aware rooted-triangle closure diagnostic using exact native slots, with eligibility fixed from visible context. Masks/queries are dependent observations, not independent graphs. Coherence improvement caused only by dense zeros is insufficient.

A persuasive restricted result requires reproducible served J-over-F improvement, useful source-pattern competence, and a differential association effect. Matching A would establish a practical implementation alternative rather than a need for multiple target predictors. Even J beating A leaves capacity/readout optimization explanations; C/S and further matched interventions are needed before attributing gains specifically beyond counts/covariance. Three seed blocks are a screen, not a precise population-effect estimate. Freeze effect reporting and any later replication rule before fitting.

Before structural attribution, also predeclare F strength controls at lambda 0.25 and 4, with identical architecture, trace and selector. Lambda-1 F remains the primary contrast. The extra F selection budget is conservative and charged: if either strength control matches J, generic auxiliary regularization remains a viable explanation. This brackets strength without pretending to match gradient directions.

## 5. PIFM adaptation where qualification is tractable

Prepare a **separate Cora HeaRT comparison of J/F/A/P**, not a cross-dataset comparison of Cora PIFM scores with Collab results. Generalize and qualify the replay geometry and feature adapter for all four arms. Use supplied positive splits and complete per-positive hard-negative rows, preserving order, ties and deliberately included observed TRAIN candidates.

P uses a fresh, source-pinned depth-one NCNC prior and the public flow architecture. Implement TRAIN-only truth/interpolants, positive and sampled-negative seed training, one label-independent candidate-centered context per pair, deterministic 96-node caps and complete coverage of center entries. Score that center entry after flow. Rebuild permitted features at the declared masking granularity. Missing coverage cannot default to 0.5. Preserve observed TRAIN anchors without candidate-label-dependent removal.

Freeze K=1, 100 prior epochs and 100 flow epochs, seed-entry MSE, and 100 complete HeaRT VALID selectors for each fitted stage with first exact tie and two selected-state replays. These are prospective adaptation budgets, not native paper reproduction. Charge prior selection, caching, all graph/feature work and flow selection. Its whole-subgraph support/cap differs from J/F; report that explicitly. Collab PIFM remains deferred until complete candidate coverage and resources are qualified.

## 6. Work estimate and preparation gates

The three-arm Collab screen is **9 fits, 15,300 optimizer updates, 900 selectors and 918 ordinary complete VALID evaluations**, before interventions and qualification. Source geometry gives 222,822,400 training query rows per fit, or 2,005,401,600 across nine fits. Each J/F fit pays 1,700 shared encoder passes and 3,400 grouped decoder query passes with four member paths, plus auxiliary checkpoint recomputation. Four extra association routes on six selected J/F states add 24 complete candidate scorings; shared context reuse must be accounted for explicitly.

The two F strength controls add six fits: the expanded suite totals **15 fits, 25,500 updates, 1,500 selectors and 1,530 ordinary VALID evaluations**, before qualification/interventions.

For support size r, A's dense causal training and cached ancestral attention require quadratic pair work; serving adds four length-r sequential generations and four target decodes. Actual degrees, parameter count, memory and wall time are unmeasured.

Compact replay integer payload is approximately **2.64–4.39 GiB per 100-epoch seed stream** under the provider's admitted planning cases; three retained streams require 7.91–13.18 GiB plus headers/receipts. A provider full-epoch audit pays 51 graph rebuilds and 102 support enumerations; each 100-epoch generation pays 1,700 rebuilds and 3,400 enumerations, before fit regeneration. Do not store all residual coordinates by default.

P pays separate prior and flow fits: with B_N/B_P batches per epoch, `100(B_N+B_P)` updates per seed. A capped context has 9,216 adjacency cells; K=1 pays one flow pass per context. Multiply by actual candidate coverage, prior work and selectors. No timing or Collab feasibility estimate is available.

Implement source-frozen adapters, exact conditional-likelihood/sampling checks on fabricated supports, detachment/RNG checks, actual replay equality, complete-batch resource qualification and selected-state replay before any fits. Report every failed attempt and actual paid work; a resource failure cannot silently authorize truncation.
