# Frozen hidden-feature fusion with graph context: one literature shortlist

5 October 2026. Literature/source notes only, unadopted. No model/data/checkpoint/logit/score/prediction payload, allocation, 18.77 or MacLink access; no training, model execution, source/gate/protocol/canonical/index changes. Public paper retrieval, PDF text extraction/rendering and artifact hashing only. All new artifacts are in this folder.

## Recommendation and unresolved question

Retain **one frozen graph-conditioned hidden residual readout** for a future source-qualified node-classification block. Freeze the already-computed four private routes and expose each route’s penultimate state to one small supervised readout. Supply that readout all member logits and one fixed labels-free summary of neighboring pooled probabilities and degree. Anchor its output to the existing mean logits:

`z_F(i) = mean_m z_m(i) + V ReLU(W [h_1(i),…,h_4(i), z_1(i),…,z_4(i), g(i)] + b) + a`.

One ReLU bottleneck allows joint decisions from the hidden bank; output is not restricted to a convex combination of existing member scores. Set its initial last layer/bias to zero, preserving native pooling at initialization. Fit only the readout, with no KD or base/member retraining. The initial capacity suggestion is r=32 for C≤32, otherwise r=C, fixed before outcomes. This is a concrete, ordinary composition of published ingredients, not a new ensemble principle.

The unresolved utility question is: **Do the four private hidden states retain useful label information beyond their scores, and can the fixed graph summary help a nonlinear readout use it?** The decisive reference is a parameter-budget-matched nonlinear score stacker supplied exactly the same graph context and supervision. Improvement over mean pooling alone is insufficient to isolate hidden information.

For unrestricted population log loss, adding per-node hidden bank H to member scores/context Q has optimal improvement `I(Y;H | Q)`. This criterion was not estimated here and supplies no finite-sample or generalization guarantee. A shared backbone may already have lost information that none of the private states contains; a later readout cannot recreate such information. Conversely, reduction to class scores may discard useful private features that a fused head can still use. Both cases are unresolved for the target routes.

## Closest published ingredients

Index69 was consulted first (249 paper records). ONE, PCL, Collaborative Learning, both C&S versions, Link-MoE and network stacking were already adopted scoped reads. Cross-stitch and sluice were not direct index69 paper records but already had local primary passages; they were reused. FFL was the only new primary identity method-read. `INDEX69_DEDUP.json` records the bounded dedup query/hits; this is not a global absence search.

| Method and exact scope | What it establishes here |
|---|---|
| **Cross-stitch**, 1604.03539v1, §3.3 Eqs1–3; §§4–5; saved exact activation Eq1/per-channel passage | Learned linear communication among continuing task activations. A late fused head changes placement and need not modify continuing routes. |
| **Sluice**, 1705.08142v1, §2 Eqs2–4; §3; saved subspace passages | Learned task/subspace communication and within-task shared/private subspaces. Its orthogonality penalty does not certify member-error independence; neither does centering a hidden bank. |
| **Collaborative Learning**, 1805.11761v1, complete saved method through §3.4, Eqs1–6 | Shared/multiple-head hard/soft collaborative training and backward scaling are prior. This does not establish frozen hidden-fusion utility. |
| **ONE**, 1806.04606v2, §3 pp3–5 Eqs1–7/Algorithm1; p7 §§4.3–4.4 | Shared low layers/private branches, a learned gated score teacher, CE/KD and default one-branch deployment. It establishes shared-branch ensemble training ancestry. |
| **PCL**, 2006.04147v2, saved complete method pp3–5; targeted revisit lines313–435 | Concatenated peer features plus an extra fully connected classifier learn a feature teacher, with CE/KD and EMA collaboration. Direct hidden-feature fusion precedent; its feature classifier is linear. |
| **FFL**, 1904.09058v2, §III pp2–4 Eqs1–6 and Figures1–2; supplementary p8 opening | Concatenate final feature maps, apply 3×3 depthwise then 1×1 pointwise convolution, with BN/ReLU at both layers. Same-architecture case shares low/private high layers. Mean-logit ensemble teaches fused classifier, fused classifier teaches branches, every classifier has CE and all train simultaneously. This is direct nonlinear hidden-fusion precedent. |
| **C&S**, 2010.13993v2, pp4–5 §§2.2–2.4 Eqs1–4, autoscale/fixed diffusion; pinned official code | Supervised TRAIN residual propagation, scale correction and label/prediction smoothing are prior. It is relevant to graph-correlated common error, but uses labels and is not the proposed labels-free graph summary. |
| **Link-MoE**, 2402.08583v2, saved §4 Eqs1–3/Algorithm1, §5.1 and scoped appendices; **network stacking**, 1909.07578, published Methods blocks29–35/SI pp8–9,15 | Structure-conditioned score gating and supervised nonlinear combinations of fixed graph scores are prior. Link-MoE’s collab gate fits on80% VALID and selects on20%; that extra supervision cannot be silently borrowed. |

FFL’s exact title is *Feature Fusion for Online Mutual Knowledge Distillation*, by Kim, Hyun, Chung and Kwak: ICPR2020 (DOI metadata2021), DOI10.1109/ICPR48806.2021.9412615. It is not a CVPR paper. This packet reads a bounded complete method, not the whole paper or author implementation. No published performance/runtime result is transferred. The proposed frozen vector readout uses neither FFL’s online mutual KD nor PCL’s EMA teachers; graph context is a further adaptation whose utility remains open.

## What common/private error reasoning actually permits

For node classification with scalar simplex weights applied to each complete member logit vector, a **common competitor** c satisfying `z_m,c > z_m,y` for every member will still beat y after any such pooling. Four incorrect members choosing different competitors can nevertheless pool correctly; “all members wrong” alone is not a convex-pool impossibility. A hidden/context head can escape the convex-hull restriction, provided the supplied features and supervision support the correction; it does not automatically rescue common errors.

For link ranking with candidate-dependent weights, common per-member positive/negative ordering is also insufficient to prove impossibility. A sufficient blocking bound is `max_m s_m(positive) < min_m s_m(negative)`; overlapping score ranges may allow separately gated scores to reverse the order. The shortlisted study is node classification, avoiding a silent node-residual-to-query-space port.

For probabilities P_m and the same TRAIN mask, define residual `E_m = M_train(Y − P_m)`. With any common linear graph error operator T and common scale s:

`mean_m [P_m + s T(E_m)] = P_bar + s T(M_train(Y − P_bar))`.

Thus centered private residual contrasts cancel under this plain linear correction and mean. Running that operation separately for all members adds nothing beyond pooled correction. C&S’s native autoscale, norm division, clipping and member-specific scales/choices can break this equality; no equality for the full native algorithm is claimed. The official pinned source uses `Y−probability`, then adds the propagated correction. The printed v2 fixes the Laplacian objective but still prints `Z−Y` with addition; that source mismatch stays disclosed.

Hidden-state mean/contrast coordinates also require care. If all centered contrasts are retained, mean plus contrasts is merely an invertible coordinate change of the concatenated bank. A mean is not an estimate of shared error, and hidden contrasts do not prove independent errors. Concatenation avoids assuming the channels of separately learned private states are semantically aligned.

## Bounded comparison specification

The core comparison has four arms: native mean-logit pooling; nonlinear residual head on all scores plus the same graph context; the proposed hidden/score/context head with ReLU replaced by identity; and the proposed nonlinear head. The identity control retains the same factored parameters and, with r≥C, can represent a full linear class readout. Give the score-only head the same total parameter budget by increasing its width, and report the resulting work difference. All heads get identical fusion examples/labels, normalization rule, recipe and selection budget.

Two attribution controls are needed before assigning a gain to specific ingredients: fix graph-context coordinates to a declared constant in the otherwise identical hidden head; and fit a budget-matched head on one private feature bank plus all scores and g. The strongest single-feature reference can be chosen among four identically fitted heads using only the fusion-selection role; disclose that four-head selection cost. These are component controls for one candidate, not a new architecture grid.

A capable single readout given the whole private bank, all scores and g contains this candidate; indeed the proposed aggregator is itself one such readout. That description is not grounds to reject a useful composition. It does mean that renaming the readout cannot establish a distinct ensemble-fusion principle. The controls ask which information sources and nonlinear operations earn their cost.

Use one fixed row-mean summary of pooled probabilities over the unchanged visible graph, plus degree/isolate context. Define isolates explicitly and preserve canonical topology/visibility. This adds one class-width sparse pass. It contains no residual seeds or label propagation. Pooled TRAIN-only C&S is an additional relevant graph-error baseline only after its label use, sign, operator/scales/clipping and selection role are declared and fairly matched; it is not a core labels-free-head ingredient.

## Availability, supervision and cost limits

This packet does **not** establish that penultimate states are saved or that an existing source exposes a suitable cache. If only logits were stored, later extraction requires all four member forwards and that cost must be charged. All four private routes remain computed; late fusion saves no dense member work and makes no sparse-compute claim. Account for base extraction, hidden storage, sparse context, head fits/inference and selection together.

A genuinely separate fusion-training/selection role or prospective out-of-fold base prediction recipe is the clean comparison. Splitting existing VALID after its labels already chose base checkpoints does not undo that dependence. TRAIN-output fitting is in-sample to the base and can overfit. A cached-only exploratory head fit must state those limitations; cleaner cross-fitting adds base fits and is outside a free cached-only claim. No labels, checkpoints, payloads, TEST results or current model outcomes were accessed here, and no future access/fit/confirmation is authorized.

If the hidden head does not improve on nonlinear scores plus the same context, extra hidden-bank utility is unsupported. Matching the linear feature control leaves nonlinear fusion unnecessary; matching the graph-free control leaves the graph explanation unsupported; matching the capable single-feature control leaves multiple private-bank benefit unsupported. Member spread, hidden variance or counts of rescued common errors are descriptive and cannot substitute for served-quality comparisons.

## Accounting and integrity

New complete-method scoped primary identities: **1 (FFL)**. New full-paper reads: **0**. Targeted retained primary method revisits: **2 (PCLv2 and C&Sv2)**. One retained pinned official C&S source was revisited; its propagation-loop function lines160–174 is a small new local source scope. Cross-stitch, sluice, ONE, Collaborative Learning and stacking/routing were reused from saved scoped conclusions/passages. Cumulative project totals are not recertified.

Keyword retrieval and full saved evidence/report exposure incidentally showed nearby published/historical result text; none was adopted. Exact scopes and exclusions are in `READ_SCOPES.json`; source pins in `SOURCE_BINDINGS.json`; public retrieval receipts in `RETRIEVAL.json`; the comparison suggestion in `COMPARATOR_SPEC.json`. Manifest/seal certify artifact bytes only. No source, adopted index, live study, selector, gate or canonical status was modified.
