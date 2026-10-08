# Quality mechanism and first-fit decision for the coupled label corrector

**For root, 8 October 2026.** Saved-literature/contract assessment only. No source change, new download, experiment, score access, server action or canonical mutation. This note assesses the actual sealed core, rather than assuming four independent private heads.

## Actual learning mechanism

The native feature backbone supplies detached H/base logits. The corrector shares learned Q/K/V/output matrices and a class-label embedding, while four BE factor rows produce route-specific attention and value/readout transformations. Shared corrector parameters receive mean-four own corrected CE on the same commonly masked TRAIN queries. Q/K input factors have standard Rademacher starts; the other factors start at one and the shared output matrix starts at zero. These are coupled predictors: every route changes the common correction function learned by the others.

The falsifiable hypothesis is **shared estimation of label compatibility across several private feature-conditioned attention paths**. With limited supervised anchors, one jointly learned label/value basis and attention geometry might estimate reusable relationships more reliably than four separately estimated bases. Private factors may retain different useful neighbor weightings within that basis. This can improve member competence while leaving alternatives for the served mean probabilities.

This is a bias/variance and optimization hypothesis, not a new observation source or new ensemble principle. UniMP, C&S, GMNN and GAMLP already establish graph label context/correction; BE and shared/private learning establish the coupling ingredients. A same-context capable single can observe and exploit every label used here. Sharing can overconstrain incompatible relations; an independent ensemble can learn the same shared solution or better ones. Stop-gradient, masking and concurrent fitting do not establish novelty.

## Own CE does not demand complementary evidence

For a target y, write `rho_m = p_m(y) / sum_l p_l(y)`. The established identity is:

`mean_m[-log p_m(y)] = -log(mean_m p_m(y)) + KL(uniform || rho)`.

Own supervision rewards competent routes and penalizes imbalance in their true-class probabilities. It need not reward different useful mistakes. Shared matrices trained on the same labels may favor one common easy label pattern, and private routes can become nearly interchangeable. Equal true-class probabilities do not imply equal wrong-class distributions, so the identity is not a collapse theorem. Useful complementarity and protected member quality must be measured.

The actual shared-matrix update can change each route's marginal learning outcome; it is therefore materially different from averaging separately trained correction heads. Conversely, averaged gradients alone are not evidence that estimation variance falls or that useful alternatives are preserved.

## When graph context can overcome redundancy

If feature-based members share a wrong rival c, changing only their nonnegative probability weights cannot make truth outrank c. Label-conditioned corrections change the predictive function. For a common base error, at least one route must satisfy

`delta_m(y) - delta_m(c) > z0(c) - z0(y)`

to overturn that rival in the served probability pool. This is necessary for that rival, not sufficient for a correct final prediction. The mean probability margin must beat every competing class; wrong-confidence changes and newly introduced errors can defeat repairs.

Such a change is rational when permitted neighboring labels contain target-relevant information, H helps distinguish relevant from misleading neighbors, and the TRAIN query signal teaches that relation. Fixed C&S can already exploit broad label smoothness; learned query-dependent attention could help only where that fixed correction is inadequate. Strong UniMP/GAMLP or a same-context single may already learn the needed conditional weighting.

If no neighboring anchor supplies a visible label value, this core's correction is exactly zero. If label neighborhoods are uninformative, misleading or indistinguishable under the available features, different factors cannot manufacture the missing evidence. Arbitrary BE spread can produce different errors or confidence without producing a correct alternative. Here the native feature map is preserved at the boundary, but an inaccurate residual can still harm every served prediction.

## Confounds that cannot be ignored

- A gain over a feature-only model can be ordinary label propagation, additional supervised context or added correction capacity; it does not establish an ensemble benefit.
- Four own-CE paths change optimization opportunities and parameter coupling relative to one path. A weak or narrow single is insufficient.
- Native backbone differences, selector/restoration policies and route/private optimizer scaling can account for contrasts. Detachment alone does not preserve an identical native trajectory.
- Sparse anchors dilute values under the all-neighbor denominator; native TRAIN overconfidence can make correction feedback weak. Mask noise is shared across routes, and drifting H can hamper learning.
- Inverse inclusion preserves restricted message/logit first moments, not probabilities, CE, pooled risk or generalization. Common query masks must hide every target label in every route and context field.

## One representative low-grid experiment

After native integration/visibility qualification, use **one complete official node-classification task, native full training schedules, three predetermined paired optimizer blocks, fixed corruption/scaling and fixed selectors**. Choose the task from source suitability and allowed label-context reach before comparing outcomes; do not use a favorable subgroup or truncate the graph. Existing WikiCS could supply development evidence only; it is not newly untouched confirmation. UniMP's full-batch ogbn-arxiv scope supplies another source-natural candidate, whose history/roles root must audit before choosing.

The first mechanistic screen is the source suite's **candidate four-route coupled corrector, capable same-context single, and four untied correctors on the same native backbone trajectory**: nine complete correction configurations. Give every method identical permitted labels, common query exclusions, context/scaling opportunities and competent fitting. This is not a coefficient, initializer or denominator grid. The untied-corrector comparison isolates learned correction coupling; it is **not** an independently acquired four-GNN baseline. Report base predictions from their native states where custody/selection matches.

After a positive screen, a **fully independently acquired native-backbone plus same-operation corrector four**, faithful C&S and a strong source-native label-aware reference remain required before a superiority claim. The live-correction-gradient control remains necessary before attributing utility to gradient isolation. These are retained obligations, not new code or expanded admission here; the suite agent owns implementation.

Charge all complete native/correction work, masking, selection and failures. The three correction alternatives could reuse one current native forward per block/update only if an owner explicitly qualifies isolated RNG/buffers/optimizer/restoration and matched coherent snapshots; that reuse is not implemented by this note. Fully independent GNN comparisons additionally pay for four separate native trajectories per block. Existing qualified references may be reused only under matching custody. No GPU-hour or latency forecast is made.

## Conservative forecast and go/no-go

The most plausible first gain is stronger member predictions from label context. Additional pool gain over a capable label-aware single is uncertain; beating a competent same-operation independent ensemble is harder. The core is worth **one qualified representative screen**, not a broad rollout or an acceptance claim.

Go only when native capture/restoration/selection is coherent, every route's target-label visibility is verified, and complete-task label context can reach the assessed population. An absence of label context or disconnected correction feedback is a reason to fix integration, not evidence for accuracy. No scientific fit is admitted by this note.

Stop ensemble-mechanism promotion if the first complete candidate does not improve on the capable single and untied correctors, or gains only by sacrificing the declared member/risk safeguards or selecting favorable blocks. If a single/C&S explains the gain, retain an attributed label-correction utility result. If independently learned correction bases match, the shared-corrector explanation is unsupported. A later failure against the full ordinary GNN ensemble closes the broader superiority claim. A positive development screen still needs unused confirmation; it does not justify changing the fixed denominator, adding a diversity loss or predicting acceptance.

## Counterfactual variance limit, not a claim about this core

For **uncoupled iid private heads** with identical marginal training recipes, conditional on a fixed common mask schedule, averaging heads on one random backbone B has excess expected Brier risk over fully independent backbone/head averaging of `(1-1/M) E||E[p|B]-E[p]||^2`. This nonnegative difference is standard total variance and explains why backbone tying alone supplies no automatic quality advantage. The actual core jointly learns shared correction matrices from all routes and does **not** satisfy those iid-head assumptions. The formula predicts neither its superiority nor its inferiority, and is not a theoretical contribution. Unequal selectors or coupled training also invalidate its direct application.
