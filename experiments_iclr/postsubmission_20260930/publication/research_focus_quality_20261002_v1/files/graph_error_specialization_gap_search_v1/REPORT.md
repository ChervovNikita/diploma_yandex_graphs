# Occasional graph-error refreshes: a bounded follow-up question

Literature/source analysis, 2 October 2026. Index v19 was consulted first. Two genuinely new primary papers received scoped method reads; no retained primary was reopened. No current study outcomes, labels, arrays, checkpoints, models, remote hosts or GPU were accessed. This packet changes no active source, protocol, index, ledger, status or manuscript.

## Decision

**A small conditional follow-up survives: test one occasional, safeguarded graph-error-directed private-factor refresh before proposing persistent training.** The operation has a precise *frozen local* prediction objective. It does not generally follow a fixed global CE objective, and the initializer's simple projection/cancellation formulas do not extend unchanged to divergent routes.

This is an attributed extension of supervised graph-error filtering, projected private-factor updates and shared-model ensembling. The argument for considering it is narrow: the current initializer only supplies graph-error orientation once, while ordinary mean-member CE does not enforce its retention. An occasional refreshed signal can test whether retention matters. Neither actual orientation decay nor utility has been observed in this packet. The already saved post-closure orientation diagnostic should determine whether this question is worth its cost. If there is no meaningful orientation to retain, or no useful finite realization, there is no reason to escalate to repeated refreshes.

No new architecture or theory breakthrough is required for the screen. No new learner, calibration guarantee, posterior interpretation or global novelty is established by it. No execution is authorized.

## Two new primary method reads

### Self-Error Adjustment, 2025

Rui Zou, **Self-Error Adjustment: Theory and Practice of Balancing Individual Performance and Diversity in Ensemble Learning**, [2508.04948v1](https://arxiv.org/html/2508.04948v1), 7 August 2025. Read saved blocks 18–42 (§III-A method/surrounding caption) and 147–160 (§IV-A setup). Zero full-paper/proof reads; author source and full Algorithm 1 listing were not audited.

SEA defines, for mean squared-error ensembling, a complementary prediction `g_i = M t - sum_(j!=i) f_j`, and adjusts the self-error term with

`e_i = (f_i-t)^2 - 2k(f_i-t)(g_i-t)`.

Its square-completed form uses a complementary target. This is direct prior for directing members using task errors rather than maximizing arbitrary output spread. Its classification experiments convert labels to one-hot **regression** targets, using MLP/LeNet and parameter grids; they do not establish a shared-factor CE recipe.

The derivation drops terms involving other learners and adds a complementary-target constant under separate-learner updates. With shared parameters those terms can depend on the shared weights. A literal sum of the printed losses with gradients through every complementary target is a different update. Consequently, neither the independent-coordinate simplification nor the reported adjustable-method bounds can be silently transferred to the GNNM shared body. We did not inspect source to resolve its implementation's target detachment. This narrows, rather than justifies, a direct SEA port.

### Controllable normalization ensembles, 2026

Mihai Suteu and Ovidiu Serban, **Controllable Diversity in Normalization-Based Implicit Ensembles via Softmax-Temperature Modulation**, [2607.23860v1](https://arxiv.org/html/2607.23860v1), 26 July 2026. Read blocks 19–50 (§§3.1–3.4), 52–59 (§§4.1–4.2, including Table 1 context), and 108–130 (Appendices A.3–A.5). Zero full-paper/proof reads; author code and figure pixels were not read.

The method shares convolution/attention weights, replicates normalization and heads, bounds per-member scales with a sigmoid, and continuously adds

`lambda sum_(layer,feature,member) log softmax_member(sigmoid(gamma)/tau)`.

This is direct prior for **ongoing** diversity control in a compact shared backbone; persistence itself is not a new principle. The regularizer acts on parameter importances. Its correspondence with useful predictive diversity is empirical on the inspected vision/text settings, not a graph-error or CE-descent guarantee. Appendix A.5 clarifies that finite-temperature owner corners persist and the observed allocation arises from balancing the task loss; temperature does not establish a universal desirable graph allocation. Appendix A.3's output-preserving conversion requires positive pretrained scales, introduces reconstructing scale/bias, and reinitializes member heads. It cannot simply be applied to unconstrained existing factors while claiming the warm function and optimizer contract remain unchanged.

All members still perform complete forward passes (§3.1). Table 1 reports roughly four single-model multiply-add counts for four members despite compact parameters. Those author-reported figures are not current host timings or a low-cost graph qualification. The paper's modulation/backbone uncertainty distinction is a structural interpretation; it does not confer calibrated uncertainty on a graph refresh.

## What the proposed graph-specific operation adds

The prospective signal uses the fixed released topology to filter the **current pooled TRAIN CE cotangent**, then remasks to TRAIN. The resulting four detached band-contrast cotangents drive the actual route Jacobians on the admitted private slice. A joint projection removes the route mean and each actual route's loss-gradient component; a separate pooled-gradient constraint can remove the extra first-order pooled CE effect. The shared dense weights receive their ordinary mean-member-CE updates during continuation and are frozen during the refresh. All other route-private state is fixed during its function evaluations.

This differs from SEA's squared-error complementary target and from normalization-importance repulsion. It also builds directly on the already known graph residual/Bernstein ingredients and the registered initializer. The source-level increment is **recomputed, graph-conditioned constrained prediction directions at a divergent actual state**, not a new transport operator, generic gradient projection or diversity regularizer.

`REFRESH_ANALYSIS.md` gives the local objective, the small joint projection, counterexamples to inherited cancellation/descent arguments, approximation assumptions and the absence of a general fixed CE potential. It identifies why repeated independent centering/projection, the identity-only initializer closure, and the earlier Gram-coloring random control cannot be reused unchanged after route divergence.

## A representative small falsifier

Use one freshly declared follow-up on the already bound 512-coordinate PolyFormer-Mono/Squirrel slice, with three seeds and **one pulse** at a fixed continuation update, such as update 200. Every seed starts from a fresh identical declared prelude; the intervention state is fixed by update count, not selected from validation outcomes. Existing or running-study fitted state is not inherited. This is an exploratory, previously exposed task, not confirmation on a new population.

Fork three arms from that same prospective actual state:

1. Common private descent only.
2. Common descent plus original-topology graph contrast directions.
3. Common descent plus node-permuted-topology contrast directions.

Use the same active slice, gradients, joint constraints, joint radius, frozen outside state and **one shared alpha**, selected by a bounded paired training-only search. All three must meet mean member CE Armijo decrease and nonincrease of pooled TRAIN CE from their own common starting baseline. Report each member's CE change; the aggregate gate does not guarantee improvement of every member. Retain nonconstructible ranks, zero signals and paired search failures without resampling or outcome-based retries.

Compare the finite graph-extra signed contrast against the common-only candidate at that same alpha, and report incremental JVP/finite geometry. This checks whether the finite pulse realizes its local target; success is partly by construction and is not independent task evidence. Resume the same native continuation budget and validation selector, copied named Adam state and matched RNG. Declare that the pulse is a manual displacement with moments preserved, not an Adam-equivalent training step. The closure, AD and state correspondence at the divergent state need a new source/qualification contract before any run.

The utility question is whether the graph pulse improves later pooled NLL beyond both common descent and the alignment-null pulse at the same total declared training budget, with accuracy, member quality, branch failures and costs retained. One possible predeclared practical screen is a mean NLL improvement of at least 0.01 nats over both controls with no more than 0.5 percentage-point mean accuracy loss; these are proposed practical constants, not theorem-derived thresholds or a significance test. Freeze them before the follow-up's outcomes. A failure to beat either control, inability to realize the finite target, or excessive paired failures/cost is reason to stop this extension. More spread alone cannot rescue it.

The permuted direction may have different predictive gains even at equal parameter radius. Thus a favorable result supports the complete graph-conditioned pulse, not an isolated graph orientation claim at matched output Gram. The earlier common-Jacobian Gram-coloring construction does not preserve all route-specific constraints here. A stronger control would be additional work, not an inherited solved step.

## Full cost, including the hidden work

The 4×4 or 5×5 Gram solve is negligible beside whole-model AD and the finite checks. At one identical fork state, a sequential low-memory construction can require up to:

- Four deterministic output forwards to form the common pooled cotangent, then four route VJP primals.
- Sixteen VJP pullbacks: four member gradients, four pooled-signal gradients, and four cotangents for each of the two topologies. The pooled constraint reuses these gradients; no full parameter Jacobian is stored.
- Eight JVP calls for incremental functional checks across the two non-common branches.
- Three sparse graph products per topology, plus normalization/permutation and all buffers.
- Up to **72 complete route forwards** for six paired line-search trials over three four-route candidates. A failed pair receives no hidden extra fallback search.

Qualification, installation checks, clone peaks, graph filtering, shared prelude, all rejected candidates, resumed continuation, optimizer state, caches and serving must also be charged. Actual AD pass costs and native model peaks remain unqualified. More frequent refreshes multiply this expense and can interrupt fused/batched training. Start with one pulse; do not assume that the tiny Gram matrix makes a persistent scheme cheap.

## Search and read limits

Six bounded arXiv metadata queries and one eight-result OpenAlex query supplied the discovery inventory. Two selected primary versions are absent from v19's normalized groups; title, authors, date and explicit version markers were verified from saved arXiv pages. Other entries remain abstract/metadata screens only, including BatchEnsemble-collapse diagnostics, NeuroTrails, EDO and an agree–disagree spiking-head method. No method result from those screens is adopted. No broad survey or author repository was opened.

All public HTTP requests in this search succeeded. An optional BeautifulSoup parser was unavailable; a saved stdlib parser replaced it. An attempted Algorithm 1 figure extraction returned no compatible listing, so no complete pseudocode inspection is claimed. These local extraction limitations establish neither source absence nor novelty. The bounded search is not exhaustive; no lack of an exact-title match is used as evidence of an unoccupied gap.

Read accounting: **2 new scoped primary methods, 0 new full reads, 0 retained-primary rereads, 0 author-code reads/executions**. Exact ranges, primary hashes, passage IDs, metadata, retrieval logs and rejected-discovery dispositions accompany this report. Cumulative project read totals remain uncertified.
