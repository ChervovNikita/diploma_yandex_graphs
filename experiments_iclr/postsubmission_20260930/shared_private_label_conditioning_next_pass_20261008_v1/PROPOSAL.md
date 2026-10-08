# Private label-conditioned corrections on a concurrently learned feature backbone

8 October 2026. Independent conceptual next pass. This is one conditional, attributed utility hypothesis—not an implementation, frozen execution protocol, scientific admission, novelty clearance or demonstrated improvement.

## Why this is worth a bounded test

Private predictors need useful correct alternatives, not only separated hidden coordinates. A graph's permitted TRAIN labels can be used as explicit prediction context, rather than solely as parameter-fitting targets. This is established graph learning. The narrower hypothesis is that several private label-evidence correctors can learn useful alternative predictions on one shared feature backbone **without letting their early shortcuts change that backbone's native feature-learning updates**.

This does not add labels or establish an information-theoretic advantage. It changes how already permitted labels enter prediction. It is potentially meaningful as a complete learning recipe if a strong single with identical context, ordinary independent paths and established label-propagation methods cannot account for its quality. The broad superiority/acceptance goal remains unachieved.

## One changed prediction and learning operation

Train one source-native feature-only graph model from scratch. Its actual native forward supplies node states `H` and base logits `z0`. In parallel, train four private, one-block graph-attention correctors. Queries and keys come from `H`; **message values come only from visible TRAIN-label embeddings**, with zero values at nodes whose labels are unavailable. Use the complete allowed graph, native edge/self conventions for the base, and observed nonself neighbor edges for the correction. A bias-free value/output path makes the correction exactly zero when its entire neighborhood has no visible label values. It therefore changes the base logits through label context, rather than adding another unconstrained raw-feature propagation branch.

For correction learning use:

`z_m = stopgrad(z0) + delta_m(stopgrad(H), G, visible_TRAIN_labels)`.

The backbone receives only its ordinary feature-only supervised loss and native optimizer update. Each private corrector receives its own corrected-prediction CE on the commonly masked query nodes. At serving, use the actual final backbone and correctors, and the fixed mean of member class probabilities. No learned router, contrastive objective, peer target, pooled training objective, virtual private adaptation, pretrained independent teachers or second compression fit is part of this proposal. Different ordinary private attention draws/stochastic paths are allowed; they are not a new initialization contribution. A zero final residual output is an established optional construction convention to qualify prospectively, not evidence of initial competence.

The core and correctors learn concurrently from the beginning. This avoids a separate donor-acquisition stage, but is not assumed to be cheaper or better. The shared representation changes as the correctors learn; that drift can hurt them. Stopping correction gradients preserves the **native update policy**, conditional on unchanged native RNG/buffers/batches/optimizer semantics. It does not guarantee competent final members, complementary mistakes, lower risk or superiority. Identical correctors and identical paths still preserve exact copied symmetry.

## Target-label visibility for every route

Let `A` be the allowed TRAIN-labeled node set. Draw a common query mask `Q` before the forward. **Every label in Q is zeroed in every route's literal label inputs and every label-derived context field.** Supervise corrected predictions on Q. All four routes receive the same remaining labels `A minus Q`; there are no member-specific bootstrap label subsets in this proposal. Masking a target only in its own route would be insufficient if another route contributing to a pooled prediction saw that label.

Recompute label-conditioned fields after masking; no full-label cached propagation, stale recurrent label state, pseudo-label field containing hidden truths, or neighbor-label lookup may bypass this exclusion. The feature-only base's ordinary supervision on A is standard parameter fitting, not literal label input. At VALID/TEST inference, only A supplies observed labels; heldout labels never become inputs or updates. Give all comparator models the same allowed labels and common masking opportunities.

This is initially a transductive node-classification hypothesis. On an unseen graph with no permitted label anchors, the specified correction vanishes. Heterogeneous labels, inductive adaptation and link-query labels each require their own visibility contract; no extension to those tasks is admitted here.

## Closest ancestry and the actual difference

| Known method/family | Already established | Remaining question here |
| --- | --- | --- |
| **UniMP**, arXiv:2009.03509v5 §§3.2–3.3 | Label embeddings enter feature/label message passing; common randomly masked labels are predicted; all allowed labels are used at inference. | Private label-only correction paths around a concurrent, gradient-isolated native feature predictor. This is an attributed composition, not a new label trick. |
| **Correct-and-Smooth**, saved v2/code conclusions | Diffuse allowed-label residuals `Y minus probability`, add a correction to base predictions, then perform separate smoothing. | Learned feature-conditioned label-message attention and several supervised correction paths. Additive correction and graph label evidence are direct prior. |
| **GMNN/GAMLP**, saved scoped conclusions | Learned graph label dependencies and separate feature/label processing; reliable label utilization/propagation. | This particular private correction bank and stopped-backbone learning policy need empirical isolation. No new conditional-label principle is claimed. |
| **Existing staged WikiCS private graph residual family** | One acquired frozen donor, private raw-X/H graph residuals, own CE/pool Brier, staged/interleaved controls and a capable joint-path single. | Its private paths read raw features and cached feature states, not commonly query-masked literal labels. The distinguishing operation is the label-only message-value interface and its visibility contract. Concurrent fitting is secondary; schedule renaming alone would add no contribution. |
| Shared/private ensembles and selective backward treatment | Private late paths, stop-gradient/block routing and member supervision are established. | Whether this graph-context composition earns useful sharing/ensemble quality under capable references. |

No exact complete equivalent was established in these bounded scopes. That is not an absence proof or novelty certificate. The possible contribution is a concrete, reproducibly useful graph learning recipe and a demonstrated benefit of private prediction paths, not unprecedented individual primitives.

## Exact falsifiers and representative work

A complete representative task with competent native schedules and several predetermined paired blocks is required. A suitable source-natural candidate is full ogbn-arxiv, which UniMP evaluates using full-batch training/inference; root must check historical exposure and choose the actual task before comparative outcomes. It is not declared unused by this note.

The decisive ensemble falsifier is a capable **single** receiving the identical common label context and feature prior, with a sufficiently wide/multihead correction block and joint nonlinear readout, matched supervision and selection opportunities. If it matches the four-route bank, the ensemble-specific explanation is unsupported. A separately trained ordinary four-model ensemble with the identical correction operation is the sharing falsifier. Strong ordinary feature-only single/ensemble and source-faithful UniMP/GAMLP or other competent label-aware references remain required.

Also compare the identical four-route correction architecture with live correction gradients into the backbone. A match closes the gradient-isolation explanation. Apply faithful C&S where its graph/label assumptions and source contract fit; its own target exclusions must hold when predicting masked TRAIN queries. If C&S explains the gain, do not promote a new correction contribution. A same-capacity label-free correction reference is needed to distinguish label context from additional private capacity. These are scientific comparison obligations, not an expanded frozen protocol or permission to launch.

Report full-population served accuracy/NLL/Brier, mean and worst member quality, coverage, repairs and introduced errors, common wrong rivals and complete costs. No label-distance subgroup, favorable seed or increased disagreement can rescue a failed aggregate comparison. A positive exploratory result requires unused confirmation and a complete-recipe prior assessment.

Per update, the candidate pays **one native base forward/backward plus four correction forward/backwards**, label-mask/context construction and all selection evaluations. It cannot cache one final H while the backbone is learning. Serving pays one native feature forward plus four correction forwards. An untied same-operation four pays four native base paths plus four correctors. A four-cell mechanism comparison (candidate, capable same-context single, untied same-operation four, live-gradient bank) over three paired blocks would be 12 complete configurations and **21 native backbone training trajectories**, plus their correction work. Additional required native label-aware/no-label/C&S references are extra; 12 is not a complete paper budget. No measured GPU-hours, runtime advantage or resource shortage is claimed.

## Disposition

Retain this one conditional utility hypothesis for root's comparison with complete current families. Do not implement, launch, modify a running family or claim methodological novelty from masks, labels or update preservation alone. If a current supported direction requires references/confirmation, finish that first.
