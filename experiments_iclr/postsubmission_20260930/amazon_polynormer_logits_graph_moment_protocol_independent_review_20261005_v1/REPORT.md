# Independent review of the frozen Amazon graph-moment protocol v1

**Scope:** mathematical and protocol review of the unchanged sealed specification, without data/logits, fitting, scientific imports, SSH, agents or literature review. Findings below identify required amendments and interpretation limits; they are not an execution authorization or an outcome verdict.

## What the specification handles correctly

- Three equal outer folds give 2,041 scored nodes and 4,082 fusion-fit nodes per fold. Every scored fold is excluded from local/global moments, fallback frequencies, posteriors, analytic skips, correction and smoothing resets. Two 2,041-node inner folds produce whole-fold-excluded training features; fitting the head to the stitched targets is coherent. C&S subsequently uses B only, and its in-sample B residuals do not introduce D labels.
- Label-free all-node predictions/topology can be transductive context. The full VALID checkpoint selection still makes base predictions depend on D labels indirectly; the protocol correctly discloses this and does not call aggregator OOF an independent full-pipeline assessment. Selecting settings/operators on outer OOF results is further development reuse, also disclosed.
- Native pooling is mean probabilities, with stable native log-mixture NLL; mean raw logits is a distinct control. Member/stage/split/node/class custody is specified. The Gram is an uncentered second moment. Its pairwise identity is correct. H is row stochastic under the stated row-normalized graph, and the fixed 20-step recurrence is well defined. The positive ridge yields a strictly convex moment weight problem, and the density constraint is feasible. Operator 9 has a unique projected probability and can use a minimum-norm weight tie rule.
- Operator 8 has an expressive single residual readout and contains operator 6 at zero residual. Both complete four-member banks receive the same operator/settings budget. The retained raw outputs, cheap tuned controls and posterior projection provide useful attribution diagnostics.
- Three overlapping blocks are correctly treated as dependent development evidence. The optional df=2 descriptive t summary is labeled accordingly, and nine folds are not claimed as nine independent replicates. Practical go/no thresholds do not imply significance. Role overlap alone does not invalidate an independently isolated per-split final predictor; future target access still needs a separate contract.

## Concrete omissions and amendments

### F1 — Pin the graph-mask probability dtype before implementation

REPORT lines 17 and 33 combine native FP32 accuracy, new FP64 arithmetic, and an explicitly possible FP32/FP64 argmax discrepancy. “Frozen native pooled predicted classes” does not unambiguously choose which prediction builds the mask. A disagreement changes the graph, all propagated fields and all corrected arms.

**Required:** declare native FP32 pooled argmax as the graph mask input, retain smallest-class ties, and record the FP64 disagreement separately. Specify whether the feature degree counts the retained self-loop and pin adjacency orientation/coalescing in the implementation. Hash each actual mask/normalized operator before outcome-dependent selection.

### F2 — Add the required processed native-single comparator

V1 reports only uncorrected member-0 metrics. Four-member banks receive supervised C&S, analytic fusion and learned heads; a raw single alias does not answer the intended comparison against a strong processed single-model predictor. Operator 8 is a capable single readout of four-member information, but it still requires the four-member bank at serving.

**Required:** retain the same fixed member-0 checkpoint alias, and add a bounded processed-single baseline with the same permitted VALID labels/folds, fixed C&S and a capable score/context residual head at a comparable parameter budget. Pin its features, graph choice, settings, initialization and training-iterate rule. Charge its extra fits/propagations explicitly. No extra backbone fit or favorable-member selection is needed.

### F3 — Hypothesis-class containment does not establish fitted-control competence

The fixed final Adam iterate can have a worse B-fit objective than the initial zero residual; containment alone does not prevent the empirical control from discarding the analytic candidate. Nonconvex fitting, two coupled settings and unequal feature/width choices also do not establish that a loss by ID 8 proves intrinsic analytic superiority.

**Required:** within the existing fixed 150 updates, retain the deterministic best permitted B-fit objective iterate, including initialization, with an explicit earliest-tie rule and no D-label epoch selection. Apply a declared rule consistently to fitted controls. Record initial/final/selected Brier, penalty/objective, finite gradient information and convergence diagnostics. If a control is empirically unqualified, its loss cannot support an analytic-superiority claim. This does not require additional seeds, fits or an adaptive search.

### F4 — Make the processed-independent quality comparison operational

The v1 quality gate compares a bank's finalist to its own cheap controls and native pool. It does not define a required comparison between the best processed shared route and the best processed independent route. “Disappear against matched independent processing” leaves the comparison, threshold and consequence undefined.

**Required:** state the cross-bank quality comparison as the best processed shared pipeline versus the best processed independent pipeline, after each split's already declared isolated selection. Keep all three splits/both banks in reporting. The per-bank graphs differ because their frozen predicted classes differ: a cross-bank result is a complete-pipeline comparison and cannot isolate a pure sharing or covariance mechanism.

Within a bank, tuned IDs 4–6 may select different rho values. Their selected-route contrasts support the tuned procedure; a claim isolating local/off-diagonal terms should additionally display the already available matched-rho contrasts. No additional arm or fitting grid is needed. Specify that the ID 8/9 “match” tolerance applies to the mean three-split Brier contrast, and reconcile the REPORT's ID 9 wording with the JSON's shared 0.001 tolerance.

### F5 — Finish numerical definitions before a fusion implementation is frozen

The minimum ridge is 1e-14 when a=1e-12 and rho=0.01. An absolute 1e-10 stationarity tolerance on the unscaled objective can accept substantially different weights while the exact minimizer is unique. “Deterministic convex solver” also does not pin the active-set/tie behavior for the rank-deficient posterior projection.

**Required:** pin a deterministic four-variable solve, scale-aware KKT/primal checks, and the minimum-weight-norm projection tie procedure. Positive scaling of the moment objective, for example by a, leaves its minimizer unchanged and makes the stated tolerance meaningful. Assert matrix symmetry/PSD, finite probabilities, row sums and weight floors. Preserve failed numerical attempts and costs; fixes must preserve the scientific constants and rerun affected arms consistently.

The residual formula can underflow when multiplied probabilities are zero numerically or when the maximum h occurs on a zero-probability coordinate. Use the stable log-skip/log-sum-exp form from the original logits and weights, with finite/simplex assertions; this requires no prediction contamination. Specify NLL contamination as an exact equation, e.g. q_delta=(1-delta)q+delta/5. Also pin standardization ddof, Xavier gain/RNG implementation, Adam options and CPU numerical environment before fitting. These are reproducibility/numerical omissions, not evidence that the model or estimator is infeasible.

### F6 — Separate pilot, final-refit and historical costs

The stated 72 MLP and 36 calibration pilot fits, feature dimensions, head parameter counts, 11,756,160 promised logit bytes and 3,134,976 field bytes per seed context are correct. The final recipe nevertheless requires up to six additional selected learned fits across bank/splits, plus full-VALID fields and C&S; these are outside the 108 pilot fits. The added single comparator changes both pilot and refit totals.

V1 already implies 270 outer configuration predictions, 540 C&S H applications, 54 outer/inner seed contexts and six label-free H(native-p) applications: 600 twenty-step H applications, or 12,000 sparse multiplication steps with different field widths. Include these, per-node convex solves, buffers/features, final refits, serving and cache I/O in measured wall/RSS. A 3.1-MB field cache is not peak RSS. The four-variable problems and small FP64 heads are bounded in size, but a generic optimizer call per node could dominate; actual batch implementation/runtime is unverified.

Preserve historical bank acquisition, qualification, successful/failed native evaluator and replay costs separately. The parent's reported CPU/GPU-receipt mismatch and corrected evaluator are existing acquisition history; they require no new fusion qualification and do not supply fusion evidence.

## Final-label and claim boundary

The own-VALID-only exclusion remains the operative frozen fusion policy. Section 8's general TRAIN/VALID isolation statement must not silently add FIT or TRAIN-control labels to this recipe. The across-split practical screen may decide whether confirmation is proposed; it must not change a split's predictor using another split's outcomes, drop an unfavorable bank/split, or be portrayed as independence. No globally disjoint-node requirement is imposed by this review.

V1 therefore has a coherent retrospective fold design and useful analytic controls, but the missing strong single control and unresolved mask/control/comparison definitions should be amended before fusion outcomes. Numerical implementation choices and all added costs must be bound at the same time. No fusion result, current payload availability, resource sufficiency or final-score custody was inspected or certified.
