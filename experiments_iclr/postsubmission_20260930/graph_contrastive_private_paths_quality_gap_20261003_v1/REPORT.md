# Contrastive supervision of private graph paths

3 October 2026. One prospective quality candidate; no execution, data/tensors/outcomes, source edits or manuscript verdict. Index_v31 and saved contrastive/label-relevant conclusions were consulted first. GENN's sealed access packet is unchanged.

## Decision

**Retain one attributed extension for a representative prospective pilot:** condition functional diversity on the target class and neighborhood composition, measure it through class-probability responses to controlled neighborhood evidence removal, and update only existing private intermediate paths under explicit competence and response-energy constraints.

This is an extension of conditional redundancy reduction, task-relevant sensitivity diversity, graph views and constrained adaptation. It is not a new contrastive primitive. Its unresolved scientific question is whether constrained changes to private graph paths produce useful complementarity that survives a competent modern multiscale encoder, rather than feature-coordinate separation, augmentation alone or weaker members. The prior filter review's zero-pilot decision concerned unspecific private graph-filter allocation; this candidate defines a distinct supervision/update experiment without changing that archived decision.

## What the saved sources establish

| Saved primary conclusion | Relevant scope and consequence |
|---|---|
| DICE, arXiv:2101.05544v1 | §2.1 Eq.1: suppresses conditional feature redundancy \(I(Z_1;Z_2\mid Y)\) while retaining label information. Class conditioning and supervised redundancy reduction are direct ancestry. Exact MI is invariant to invertible feature transformations; it must not be caricatured as raw cosine repulsion. |
| CDLG, arXiv:2306.11344v1 | §III-C: same node/channel across views positive, same node/different channels negative; neighborhood routing/channel projections feed one concatenated predictor. Graph channel contrast is prior. The saved PDF check identifies a printed negative-loss ambiguity; a repaired implementation needs explicit qualification. |
| GNCL, arXiv:2011.02952v2 | §4.1 Eq.5: trades mean member loss against collective loss. Competence/diversity trade-offs are established. |
| ADP, arXiv:1901.08846v3 | §§3.1–3.3: non-target probability diversity plus entropy and member CE. Uses probability pooling; its bare determinant is invalid if \(M>C-1\). It is not a guarantee for restricted shared paths/mean logits. |
| FoRDE, arXiv:2306.02775v3 | §3.2 normalized true-label input-gradient kernel; §3.4 biased minibatch approximation and §3.5 differentiation cost. Simply moving a gradient penalty to edges is insufficient as this extension's delta. |
| Training Diverse Graph Experts, arXiv:2510.18370v1 | §§2.1–2.3, App.B.2: independent expert training, filter/direction/init/data variation, heldout fusion. Different graph evidence is established; disagreement can reflect one weak expert. |
| Wood et al., JMLR 24(359), 23-0041 | Saved PDF pp.9–14: loss-matched diversity decomposition does not establish that maximizing diversity improves risk. |
| PolyFormer, arXiv:2407.14459v1; TFE-GNN, DOI:10.52202/079017-2966 | Saved method/source scopes establish competent polynomial-token and low/high-pass multiscale controls. The shared encoder need not discard graph-frequency evidence. |

Exact source versions, prior passage coordinates and hashes are in `REUSED_REFERENCES.json` and `READ_SCOPES.json`. No primary PDF/HTML was reopened. Archived ADP quotations were incidentally exposed while checking the source-custody JSON and are recorded separately. No new search, deliberately selected primary method scope, full-paper certification or author-source audit was needed to formulate this conditional experiment. GENN remains an unresolved closest prior; its missing source is not evidence that the proposed combination is absent.

## One precise candidate

Use an M=4 all-layer BatchEnsemble version of a qualified PolyFormer backbone, retaining complete member token-attention/FFN trajectories. Stage A trains a competent ordinary ensemble with native mean-member CE. Stage B copies the same warm function into every arm, freezes common weights, stems, classifier heads and running statistics, and updates only the **already present private factors of intermediate attention/FFN maps**. Every Stage B objective and guard uses dropout off with gradients enabled where required, so response separation cannot be random-mask noise. No new predictor, projection head, filter parameter or inference gate is added. Every member predictor has the same parameter tensors; helper-estimator parameters and active-parameter differences in separate controls are disclosed and charged.

Within official TRAIN, fix an 80/20 fit/control role split before training. Fit labels train the ordinary losses and construct two auxiliary graph views: remove a fixed 10% sample of edges between fit-role labeled nodes of the same class, or of different classes. Preserve the native graph at serving. Use the same graph normalization and complete graph-feature token computation for each view. These are **supervised evidence-removal probes**, not claimed label-preserving causal interventions. No VALIDATION/TEST label defines a mask, group, reference or acceptance constraint.

For a labeled target \(v\), define its group \(s(v)=(y_v,g_v)\), where \(g_v\) indicates whether at least half its fit-labeled neighbors have class \(y_v\). Targets without fit-labeled neighbors use a separate fallback. Groups with fewer than 32 fit targets collapse to class-only, then global if required; this predeclared fallback never drops their native supervision. The masks, sampling seed and rules are fixed, not selected by later utility.

Let \(p_m^0(v)\) be member probabilities on the native graph and \(p_m^+(v),p_m^-(v)\) the two probes. Form the class-anchored response

\[
A_m(v)=\operatorname{Concat}[p_m^0(v)-p_m^+(v),\;p_m^0(v)-p_m^-(v)].
\]

Within each group, center across targets and flatten to \(R_{m,s}\); let \(U_{m,s}=R_{m,s}/\|R_{m,s}\|_2\). Penalize conditional response redundancy,

\[
D=\operatorname{mean}_s\operatorname{mean}_{m<n}
\langle U_{m,s},U_{n,s}\rangle^2.
\]

This is a normalized second-order conditional redundancy surrogate, **not DICE's MI estimator** and not an InfoNCE claim. It leaves the class/group mean response unrepelled. It changes the measured object from arbitrary hidden coordinates to a functionally identified response to particular graph evidence. Pairwise decorrelation is not independence or useful error complementarity.

Stage B proposes private AdamW updates minimizing native mean CE plus mean probe CE plus **0.1 D**. From the proposed complete private displacement, try scales 1, 1/2, 1/4, 1/8, then zero. Accept only if every member's deterministic native and both-probe control-role CE, globally and in supported class/group cells, is at most its copied warm reference plus **0.01 nats**. Cells with fewer than 16 control targets use class/global fallback. A zero step discards proposed parameter and optimizer-state changes; an accepted step commits the corresponding AdamW moments once. The actual guard evaluations/backtracking cost is charged. This is ordinary constrained/backtracked adaptation, not unchanged-objective gradient descent or a convergence theorem.

For active response groups, also constrain each \(\|R_{m,s}\|\) to [0.5,2] times its copied warm reference. Groups with any reference norm below 1e-5 are excluded from D for all arms; their CE remains supervised. Freeze this active set. If it covers under half the fit targets, stop the proposed mechanism screen. These safeguards prevent winning D by zero responses or unlimited response amplitude. They can forbid a legitimate move toward graph invariance and can make the problem infeasible; that is a failure condition, not a reason to relax constraints after seeing results.

## Why superficial separation cannot suffice

Compensated hidden rotations/rescalings, classifier-nullspace changes and common shifts of all logits leave every native/probe probability unchanged, hence leave A, R, U and D unchanged. The claim does not assume those transformations are reachable symmetries of the tied network. Multiplying a response vector by a positive scalar also leaves U and D unchanged; the response-energy band prevents collapse/inflation. No learned member temperature or mutable auxiliary readout is supplied. A logit-temperature change that changes probabilities is an actual predictor change, not a coordinate gauge, and must survive competence/utility checks.

The competence constraints protect only the declared TRAIN control roles and probes. Repeated guard use can overfit those roles; they are supervision, not heldout evidence. They do **not** guarantee validation/test competence. Every member must also pass the separate prospective VALIDATION competence screen below. More accepted steps, lower D or distinct neighbor responses are not success criteria.

## Plausible mechanism and operational delta

On nodes of the same class but with different labeled-neighbor composition, a common multiscale model can still learn correlated reliance on particular neighborhood evidence. Existing private intermediate factors can alter nonlinear order/channel selection while the common backbone and class heads remain fixed. Conditional functional supervision might encourage alternative uses of retained graph evidence, preserving label competence instead of repelling information every accurate member needs.

Relative to ordinary embedding repulsion, the supervised object is identified class-probability evidence response. Relative to independent diverse graph experts, members remain jointly represented by shared fixed maps, keep the same native graph/pool, and receive conditional private-path credit under feasibility constraints. Relative to merely adding parameters, the tensors and inference architecture are identical. Relative to generic graph-gradient repulsion, finite graph-removal responses, class/neighborhood conditioning, fixed intermediate adaptation and competence/energy constraints jointly define the tested operation. Each ingredient is attributed; complete published duplication has not been excluded.

It can fail through nuisance-sensitive response variation, label-derived-mask shortcuts, insufficient private capacity, sparse groups, correlated errors, frozen-backbone restriction, guard overfitting or repeated rejected steps. Graph views may be harmful on genuine heterophily. A multiscale encoder may already use all useful evidence, or the best predictor may be insensitive to these probes. Stronger unconditioned diversity or augmentation alone may explain any gain.

For the primary mean-logit pool, retain the saved exact accounting
\(\mathrm{mean}_m CE(z_m,y)=CE(\bar z,y)+A\), where
\(A=\mathrm{mean}_m\log\sum_c e^{z_{mc}}-\log\sum_c e^{\bar z_c}\).
Report changes in mean-member NLL and A separately. Larger A at the expense of member competence is insufficient, and current centered logit spread alone cannot alter a fixed mean-logit predictor.

## One representative prospective pilot

Use the **complete Amazon-ratings node-classification task**, its exact released version and native official splits. Prospectively pair split IDs 0/1/2 with optimizer seeds 17/29/43; these are three blocks on one graph, not three independent graph populations. Qualify PolyFormer and all-layer exact-member BE against their source recipes before admission. No current fitted bank is inherited. Keep one common ordinary warm bank per block and a fixed 200-update Stage B continuation with the same native validation-NLL checkpoint selector for every continuation arm.

Eight fixed continuation arms from each warm bank (one protocol, no grid):

1. Native CE only, with the same guards/energy rules.
2. Native plus probe CE, no diversity.
3. Candidate conditional functional response penalty.
4. Class-only response penalty, omitting neighborhood grouping.
5. Class/group-normalized hidden cosine repulsion at the same intermediate sites.
6. Source-qualified DICE conditional feature redundancy, same backbone/private update permission and guard roles; charge its estimator parameters/training.
7. Source-qualified FoRDE normalized true-label input-gradient diversity, same private permission and supervision; charge extra differentiation. Any graph-domain adaptation must be explicitly labeled.
8. Candidate with degree-matched label-permuted evidence-removal masks, preserving removed-edge counts; exact construction must be qualified before freezing.

All diversity arms receive the same native/probe CE terms, guards and frozen warm state; DICE/FoRDE retain a competent source-informed recipe qualified without these pilot outcomes. The fixed 0.1 normalized-surrogate coefficient is not asserted optimal. If the closest controls cannot be qualified competently, do not launch or treat them as weak stand-ins. Optional tuning is not hidden in this protocol.

Utility references: competent PolyFormer single, capacity-matched PolyFormer single, native TFE-GNN single and M=4 independently trained PolyFormer ensemble, with matched label visibility, selection and paid continuation budget. Include both raw-logit and probability pools as separately labeled endpoints; the primary pool stays mean raw logits. Offer competent packing to independent members. Charge all warm acquisition, three graph/token caches, adversarial estimators, gradient passes, guard evaluations and rejected steps. Base acquisitions alone are 33 single-member fit equivalents across three blocks, plus 24 M4 continuation banks and matched utility-reference continuation. No timing, hardware need or feasibility result is asserted; a complete-data resource preflight is required.

Primary development screen, frozen before training: candidate mean selected VALIDATION NLL improves by at least **0.01 nats** over augmentation-only and the qualified diversity controls, with the same improvement sign in all three blocks; mean accuracy/macro-F1 loss at most 0.2 percentage points; mean-member NLL increase at most 0.01 nats and worst-member increase at most 0.02. Intervals are descriptive. If class-only matches it, neighborhood conditioning is unnecessary; if permuted masks match it, the claimed evidence semantics are unsupported. If D improves without pooled utility, or guards leave under 10% of proposals accepted, stop this branch. Do not rescue it with extra sites, groups, masks, losses or seeds after outcomes.

If a qualified utility reference matches pooled quality more cheaply, the result supplies only bounded mechanism evidence, not a useful quality/cost extension. A passing screen only admits a frozen, fresh heldout confirmation, such as complete Roman-empire with native splits, new predetermined seeds and the unchanged mechanism/control set. It supplies no present quality, generality, novelty or publication verdict. Existing frozen studies are unaffected. `PILOT_SPEC.json` records this single unexecuted candidate and its qualification gates.
