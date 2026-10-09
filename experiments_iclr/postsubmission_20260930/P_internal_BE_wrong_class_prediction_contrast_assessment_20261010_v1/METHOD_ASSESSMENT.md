# Internal BE contrast: one attributed prediction-space alternative

## Decision and measured problem

Retain **one unimplemented, prospectively falsifiable contrasted-object extension**. It replaces hidden residual contrast with contrast between actual wrong-class prediction distributions, while keeping shared and boundary supervision unchanged. This is distinct from the fitted Wiki15 hidden objectives, but the prediction object is already ADP ancestry and the pairing is CDLG-style. Internal-private routing already exists. No novel graph mechanism, benefit, execution admission or complete prior exclusion is claimed.

The admitted saved Wiki15 reports establish practical decision redundancy: all 15 states have identical member top-one labels on 98.123–99.583% of 5274 development nodes, zero pooled-only rescues, and a strict common wrong rival on almost every all-member-wrong node. Combined C−P mean accuracy is −0.0695 pp and canonical S−P is −0.1959 pp. More disagreement in S/6203 accompanies weaker members and more pool harms. These are selected-development findings, not independent TEST confirmation.

They do **not** show that hidden separation succeeded, that classifier-null directions moved, or that a particular internal BE site caused the redundancy. No embeddings or trajectory were measured in the compact reports. New attention-credit/relation18 outcomes remain unopened in this assessment.

## What would repeat prior recipes

- A/S already use same-label positives and other-label negatives within each member across stochastic views. Ordinary class SupCon is not a new proposal.
- R already pairs the same member/object across views positively and other members on the same object negatively after TRAIN class centering. C combines A and R.
- `alignment_private` and `hidden_private` already apply these losses only to inventoried internal factors, with shared maps and input/head boundary factors receiving mean-own supervision. Restricting gradients or moving the same hidden objective inward is duplication.
- Class-centered hidden codes, attention/head spread, a learned auxiliary projection or a local classifier-null component can change while actual decisions remain unchanged. Saved DICE/CDLG/nullspace work already supplies that concern.
- Prediction-tangent projection, finite graph-response repulsion and InfoNCE on those responses are also saved attributed proposals. They are not reissued here.

## One extension: contrast wrong-class predictions, update internal factors

Use the existing WikiCS multiclass BE model, all TRAIN labels, four members, original graph/features and the same two native dropout forwards. No extra graph view, teacher, projector, head, private capacity or inference rule is added. Maintain the original bounded auxiliary target index rule; every TRAIN target still receives own supervision.

For TRAIN target `i`, true class `y_i`, member `m` and view `v`, let `z_mi^v` be the **actual native final logits**, and define

```
q_mi^v = softmax((z_mi,c^v) for c != y_i)
u_mi^v = q_mi^v / ||q_mi^v||_2
s_mki   = <u_mi^a, u_ki^b> / 0.2
R_out   = mean_i mean_m 0.5 * [
             -log(exp(s_mmi) / sum_k exp(s_mki))
             + the corresponding a/b-swapped term]
```

The positive is the same member/target across views; negatives are other members on that same target. Compute log denominators stably. Softmax restricted to wrong logits is exactly the conditional wrong-class distribution; after L2 normalization it equals ADP's normalized full probability vector with `y_i` removed. This is not a newly invented prediction object.

To replace only the contrasted statistic, retain the existing `.05 A` common alignment component. With `F` the unchanged two-view mean-own loss, use the audited partition:

```
shared theta, input/classifier boundary psi: gradient(F)
internal private phi: gradient(F + .05 A + .05 R_out)
```

All gradients are evaluated at the same old parameters/views and committed in one existing Adam update. Do not detach the native downstream continuation from the internal factors: it supplies their actual prediction pullback. Auxiliary gradients are requested only for `phi`; shared and boundary weights remain live for their own supervised updates. This reuses the known mixed-block routing principle; it is generally not one scalar global-gradient update and gives no Adam descent guarantee.

For the saved public WikiCS adapter, `phi` is the 56 inventoried internal private tensors and `psi` the 6 input/phase-head boundary tensors. Inactive local/global tensors retain unused gradients. Another constructor or attention partition requires its own source audit; the inventory is not silently transferable.

This is prediction-space supervision applied through internal BE paths. A raw intermediate activation is not automatically prediction relevant. If “contrast at an internal output” instead means cosine distance before native downstream maps, the classifier-null concern remains.

## Exact scope of the prediction-relevance claim

Hidden-coordinate changes that preserve the actual logits/probabilities leave `u` and this contrast unchanged. An auxiliary learned projector cannot absorb it because none exists. The loss therefore cannot be reduced by a purely classifier-null hidden code or common shift of all logits.

That is a limited invariance result, **not** a complementarity theorem. Low-mass wrong tails, wrong-rival stereotypes or harmful probability reallocations can reduce contrast while a common rival still defeats truth. The loss ignores the overall wrong-versus-true probability mass. Excluding `y` gives zero direct auxiliary derivative in its logit, but internal parameter changes also change the true logit and future own/head updates. Member competence is not protected by that exclusion.

A fixed member-specific class bias is another shortcut: internal factors acting on shared biased hidden units can generate virtual logit offsets even when classifier boundary parameters receive own gradients only. The output changes, so it is not a classifier-null gauge, but it need not encode different graph evidence. At the probability interface, assign each member truth probability `t` and its own distinct wrong class probability `1-t`. Wrong vectors have maximal disjoint spread; every member is wrong for `t<1/2`, and the uniform pool is still wrong for `t<1/(M+1)`. These are probability-level examples (or finite-softmax limits), not a claim that every tied BE model realizes them. Clearing a common rival is therefore not sufficient for rescue.

Copied deterministic members are also a symmetry stationary case; native separate dropout can break realized symmetry but guarantees no useful alternatives. No new initialization distribution or redraw is included. A zero prediction pullback at an internal site remains zero; no private-only permission repairs that structural limitation.

**Multiclass eligibility is essential.** With binary output, the one-dimensional wrong-class vector is always1 and `R_out = log M` with zero gradient. This does not supply a link-prediction or MolHIV mechanism. Independent Bernoulli labels do not become an eligible multiclass target by being stacked together. WikiCS C=10/M=4 avoids ADP's dimensional obstruction, but not numerical/symmetry degeneracy.

## Minimal competent comparison and falsification

Use one fixed candidate weight/temperature above, with no strength, site, rank, view or seed grid. Match backbone, parameters, initialization, all label visibility, two forwards, auxiliary targets, own-loss scaling, gradient permissions, optimizer/selection, full horizon and actual probability-mean serving. Charge the extra reverse sweeps and retained graphs.

Four scientific policies identify the changed object:

1. Existing alignment-private reference: `phi` gets `F + .05 A`, no residual contrast.
2. Existing hidden-private reference: `phi` gets `F + .05 A + .05 R_hidden`.
3. Proposed output-private contrast: replace only `R_hidden` with `R_out`.
4. **Complete native ADP regularizer control**, under the same declared restricted graph/private permissions and common alignment component: retain both ensemble entropy and bare wrong-vector Gram logdet,

   `F + .05 A + mean_views,targets[-alpha H(mean_m p_m) - beta logdet(U_y^T U_y)] / M`.

The `/M` expresses native summed-member CE scaling in the mean-own convention; record every reduction explicitly. Use one prospectively source-qualified alpha/beta setting, not a grid. Retaining A and restricting recipients makes this an explicitly matched graph/private adaptation of ADP, not a reproduction or adverse verdict on its published independent image networks. A determinant-only, pairwise, epsilon or pseudodeterminant proxy is not the complete native ADP objective. A copied/saturated bank can make the bare Gram singular even when M≤C−1; qualify finite computation at the declared starts, retain failure if that control cannot be executed competently, and do not silently repair it.

Also require a competent native single, a capable same-information single with the total capacity/opportunity budget, and genuine independently trained4 with disjoint complete bodies and own selectors. A jointly regularized untied version can test whether the output contrast's utility is generic, but is separately labeled and does not replace genuine independent4. Historical ordinary independent4 counts in Wiki15 have different recipe identities and cannot establish a new paired advantage.

For mechanism attribution, include one capable **common predictor plus member class-offset control**, `z_m(x)=g(x)+b_m`, with the same own/output-contrast/view/selection allowances and charged work. The common predictor retains own supervision; its explicitly declared private offsets receive the contrastive update. This is a narrower-capacity diagnostic control with a different auxiliary recipient, not a same-parameter reproduction of internal `phi`. If the candidate's served/member/rival gains do not exceed this control, a graph-evidence specialization claim is unsupported. In this control, fixed offsets cancel from every finite **logit** truth-versus-rival response to an input/graph change. Probability responses can still differ through softmax, so probability-response variation alone does not exclude this shortcut. Any attribution assay uses an already prospectively fixed panel; no separate new graph-response family is added to repair that failure.

One meaningful pairing falsifier is a prospectively fixed **within-TRAIN-class permutation of peer target identities in the negative pairs**. Keep every positive on its original member/target; keep class coordinate identities and all own supervision unchanged. This tests target-specific allocation versus a member-wide wrong-class stereotype. Report how many negatives actually change; singleton classes cannot be deranged and do not lose own supervision. If this control matches, same-object specificity is unsupported. A label shuffle that accidentally treats truth as a wrong coordinate would confound the question and is not proposed.

Use a full-population wrong-rival falsifier: record common-rival witnesses cleared and introduced, all-wrong pooled-only rescues, acquired-correct-member repairs, coverage gained/lost and introduced errors. If contrast drops but the dominant common rival persists and pooled truth margins/served predictions do not improve, reject useful diversification. Lower hidden/output loss, angular spread, rivalry churn or Jensen gap alone cannot advance the candidate.

Advancement requires a prospectively declared improvement in served task quality and full-population net repairs, while respecting fixed mean/worst-member competence limits and capable controls. All labels, states, failures and costs remain retained. No coefficients, seed, checkpoint, class or evaluated cohort is selected to rescue a negative result. Passing development remains a reason for unused confirmation, not confirmation itself.

## Closest prior and handoff

ADP is the closest prior for the exact wrong-vector geometry and output-level motivation. CDLG precedes the same-object/member-or-channel negative pairing; its printed loss caveat remains and no repaired author implementation is claimed. DICE precedes preserving class information while reducing residual redundancy, but this contrast is not conditional MI. FoRDE and function-space repulsive ensembles already diversify task-function sensitivities/evaluations. GNCL and the saved private-steering adapter establish own/shared versus auxiliary/private routing. None supplies a graph-specific novelty or competence guarantee for this composition.

**Handoff:** retain this one attributed candidate and its comparisons; first let root interpret the already completed attention-credit family. Do not consume that new evidence until an immutable compact bundle is supplied. If its existing rule already supplies useful alternatives, reassess whether this candidate is necessary. If the deficit persists, a separately reviewed source/protocol may implement the changed auxiliary object using the existing audited routing. No runtime, fits, new owner framework or scientific source is supplied here.

Reading counts: zero new primary methods, primary revisits, public requests, full-paper reads or author-code audits. Saved DICE/CDLG/ADP/FoRDE/output-diversity scopes were reused through reports and exact scope references. Only the explicitly admitted Wiki15 reports and existing project loss/permission source were consulted; no raw predictions, arrays, embeddings, checkpoints or new attention outcomes were opened.
